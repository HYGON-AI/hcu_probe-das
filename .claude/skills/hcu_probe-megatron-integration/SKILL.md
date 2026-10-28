---
name: hcu_probe-megatron-integration
description: 在 hcu_megatron(海光 HCU 版 Megatron-LM fork)训练流程中接入 hcu_probe whl 精度调试工具,训练命令加 CLI flag 即激活、0 行代码改训练脚本。讲怎么写 feature/wrapper 薄壳、cfg 模板、L1 yaml patch、命令行用法;6 个坑 + 验证见 references/runbook.md。当用户问"如何接入 hcu_probe 到 hcu_megatron / 怎么写 feature / 怎么写 wrapper / L1 dump 报 fusion ValueError / risk_level invalid / hcu-probe-config CLI / patch_optimizer_step / TORCHDYNAMO_DISABLE / megatron monitor 三通道" 时触发。
---

# hcu_probe-megatron-integration(接入教学)

把 `hcu_probe` whl 接入 hcu_megatron(海光 HCU 版 Megatron-LM fork)训练流程:**在 hcu_megatron 的 `features_manager` 下写 4 个薄壳文件,训练命令加 `--hcu-probe-config` flag 即激活,不改 megatron / 训练脚本任何一行。**

> **前置定位**:面向 **hcu_megatron 训练已跑通的人**(HCU 环境 / 模型 / 数据就绪),只讲 hcu_probe dump/monitor 接入,不含环境搭建。
>
> **本 skill 自包含**:接入要写的 4 个薄壳文件在 `scripts/`(拷进你的 hcu_megatron fork 即可,不需翻别的仓);cfg 模板在 `config/`;坑 + 验证在 `references/runbook.md`。hcu_probe whl 本身已带翻译层 `hcuprobe.integrations.megatron.*`(薄壳调它)。

## ✅ 2026-10 复验(Megatron-LM-das core_v0.18.2 + 光源镜像,先读这节)

环境:`harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron-0.18.2-ubuntu22.04-dtk2604-py3.10-20260908-0350` + GitHub `HYGON-AI/Megatron-LM-das@52b4450` + Qwen3-0.6B(官方 `examples/qwen3/train_qwen3_0.6B.sh`,TP4,单机 8 卡)。完整步骤见 `docs_hcu/zh/quick_start/megatron_quick_start.md`。

| 模式 / 工具 | 结果 |
|---|---|
| baseline | ✅ loss 11.95→11.39(10 步) |
| L0 | ✅ 8 rank dump;loss 与基线逐位一致。GBS=256 时每个采集 step 约 15 min(128 micro batch),建议 `step:[0]` 或调小 GBS |
| L1 | ✅(需 Step 4 yaml patch)。**未 patch 实测复现** `Only support fusion of gelu and swiglu` → 训练崩溃;patch 后正常,dump 不含 silu/gelu |
| monitor | ✅ 8 rank × 3 通道 × 每步 CSV,loss 不变 |
| combined(L0 + monitor) | ✅ 两者同时产出 |
| `hcuprobe compare`(两次同配置 L0) | ✅ 前向 23448 条逐位一致,差异仅在反向梯度(原子加非确定性) |
| `hcuprobe overflow_check` | ✅ 正常数据报 "Cannot find any anomaly node";注入 NaN 后准确定位到起点 `layers.10.mlp.linear_fc1` |
| `hcuprobe graph_visualize` | ✅ 单图 / 双图 `.vis.db` |
| `hcuprobe multi_acc_check`(精度预检,L1 输入) | ✅ 6709 API:pass 3028 / warning 1326(bf16 vs fp32 正常舍入,相对误差 ~2⁻⁸)/ error 0 / SKIP 2355(Distributed.*);8 卡约 12 min |

**相对旧版(dcu_megatron + llama2-7B 时代)的关键差异**:
1. 路径:`hcu_megatron/features_manager/`(旧为 `hcu_megatron/adaptor/features_manager/`),`ADAPTOR_FEATURES` 现有 15 个 Feature。
2. **`train_step` 挂点必须是 `hcu_megatron.training.training.train_step`**:Megatron-LM-das 的 `MegatronBasicFeature` 把 megatron 的 `train` 换成 `hcu_megatron.training.training.train`,它调用的是本模块的 `train_step`;只挂 `megatron.training.training.train_step` 时 `train`/`setup_model_and_optimizer` 钩子生效但 `train_step` 钩子永不触发 → dump/monitor **只有 initialized、没有任何数据**。`scripts/*_feature.py` 已用 `_train_step_target()` 自动选择。
3. import 为 `hcuprobe.*`(whl 文件名 `hcu_probe-*.whl`)。

---

## 接入概述 — 2 个适配点

1. **写 4 个 feature/wrapper 薄壳**(`scripts/` 已备,拷进你的 hcu_megatron fork `hcu_megatron/features_manager/probe/`)— 用 hcu_megatron 的 patch 机制把 hcu_probe lifecycle 织进训练 hook。
2. **L1 dump 模式**才需:本地 patch `support_wrap_ops.yaml`(避免 Megatron MLP swiglu fusion identity check 失败)。L0/monitor 不需要。

**机制(为什么 0 行代码改训练脚本)**:hcu_megatron 的 `features_manager` 维护一个 **`ADAPTOR_FEATURES` list**(`features_manager/__init__.py`),启动时遍历它、调每个 Feature 的 `register_patches()` → `patch_manager.register_patch(megatron函数, wrapper)` 用 monkey-patch 把 wrapper 织进 megatron 的 `train`/`train_step`/`setup_model_and_optimizer`。所以**把你的 Feature 加进这个 list** + 训练命令加 flag 就激活,megatron/训练脚本本体 0 改动。`AbstractFeature` / `patch_manager` 都是 hcu_megatron 自带的,你不用建。

---

## Step 1: 4 个薄壳文件(在 `scripts/`,拷进你的 hcu_megatron fork)

**两步,缺一不可**:① 拷 4 文件 ② 在 `ADAPTOR_FEATURES` 注册(光拷不注册 = 静默无效,无报错、行为同 baseline)。

**① 拷文件**:本 skill `scripts/` 下 4 个薄壳**拷进你的 hcu_megatron fork 的 `hcu_megatron/features_manager/probe/`**(它们 `from ..feature import AbstractFeature`,放对位置才能 import):

| `scripts/` 文件 | 职责 | 关键契约 |
|---|---|---|
| `hcu_probe_feature.py` | 注册 `--hcu-probe-config` + `--hcu-probe-seed` flag,`register_patches()` 把 dump wrapper 挂到 `train` / `train_step` | dump 入口 |
| `hcu_probe_wrapper.py` | 2 个 dump 薄壳:`train` 包 `dump.on_train_start/on_train_end`、`train_step` 包 `dump.on_step_start/on_step_end` | 薄壳调 `hcuprobe.integrations.megatron.dump.on_*` |
| `monitor_feature.py` | 注册 `--hcu-probe-monitor-config` flag,`register_patches()` 把 monitor wrapper 挂到 `setup_model_and_optimizer` / `train` / `train_step` | monitor 入口 |
| `monitor_wrapper.py` | 3 个 monitor 薄壳:`setup` 出口拿 model+optimizer 起 monitor、`train_step` 出口 `monitor.on_step_end`、`train` finally `on_train_end` | 薄壳调 `hcuprobe.integrations.megatron.monitor.on_*` |

**② 注册进 `ADAPTOR_FEATURES`**(关键,别漏)—— 编辑 `hcu_megatron/features_manager/__init__.py`,仿照里面已有的 Feature(core_v0.18.2 为 15 个),加 2 行 import + 2 个实例:

```python
# __init__.py 顶部 import 区加:
from .probe.hcu_probe_feature import HcuProbeFeature
from .probe.monitor_feature import MonitorFeature

# ADAPTOR_FEATURES = [ ... ] list 末尾加:
    HcuProbeFeature(),
    MonitorFeature(),
```

> hcu_megatron **没有自动扫描**,Feature 靠这个 list 显式登记;`probe/` 子目录可不建 `__init__.py`(同仓其他子目录一样走 namespace package)。

**为什么薄壳只有几十行**:业务逻辑(`PrecisionDebugger` init / monitor 启停 / rank 过滤 / cfg 解析)全在 hcu_probe whl 的 `hcuprobe.integrations.megatron.*` 翻译层,薄壳只做"注册 flag + 注册 patch + 透传"。所以 trl / deepspeed / lightning 等其他框架接入也只需类似的几行薄壳。

**要加新 hook 点**(如想在 `pretrain_gpt.forward_step` 也 dump):在对应 `register_patches()` 多注册一条 + 写对应 wrapper 薄壳调 `dump.on_xxx`(若翻译层没 `on_xxx`,先在 hcu_probe whl 那边加)。

### 5 个 hook 点 → 翻译层函数

| hook | 挂在 megatron 哪 | 触发条件 |
|---|---|---|
| `dump.on_train_start(args)` | `train()` 入口 | 传了 `--hcu-probe-config` |
| `dump.on_step_start(model, args)` | `train_step()` 入口 | 传了 `--hcu-probe-config` |
| `dump.on_step_end(result, args)` | `train_step()` 出口 | 同上 |
| `dump.on_train_end()` | `train()` finally | 同上 |
| `monitor.on_setup_done(models, opt, args)` | `setup_model_and_optimizer()` 出口 | 传了 `--hcu-probe-monitor-config` |
| `monitor.on_step_end` / `on_train_end` | `train_step()` 出口 / `train()` finally | 同上 |

---

## Step 2: cfg 模板(在 `config/`)

3 个模板已捆在本 skill `config/`,改 `dump_path` / `output_dir` 即用:

| 文件 | 用途 |
|---|---|
| `hcu_probe_config_statistics_L0.json` | L0 module 级 statistics(快,生产推荐) |
| `hcu_probe_config_statistics_L1.json` | L1 API 级(同 L0 改 `"level":"L1"`;⚠️ 需 Step 4 yaml patch) |
| `monitor_v2_config_full_probe.json` | monitor 三通道(weight_grad/optimizer/param) |

**cfg 编写 5 条 hcu_megatron 专属约束**(原理见 [`references/runbook.md`](references/runbook.md)):
- `"risk_level"`(若用)必须**大写** `ALL`/`FOCUS`/`CORE`,小写 init failed;且只在 **L1** 生效,L0 忽略。
- monitor cfg **必设** `"patch_optimizer_step": false`(Megatron nested optimizer 让 upstream 自动 patch 扑空,由 wrapper 手动驱动 step)。
- monitor cfg **不开** `monitors.module` 通道(与 Megatron pipeline `deallocate_pipeline_outputs` 冲突)。
- L1 cfg 跑时 launcher 自动 `export TORCHDYNAMO_DISABLE=1`(L1 monkey-patch 与 dynamo 冲突)。

---

## Step 3: 命令行怎么进到 megatron(关键:flag 要在启动脚本里解析 + 转发)

⚠️ **flag 不会自动从命令行进 megatron** —— megatron 的 `pretrain_gpt.py` 只认 `--hcu-probe-config`(**连字符**,Step 1 的 `register_args` 注册的);但典型 launcher 用 `--hcu_probe_config`(**下划线**)接收。你的启动脚本必须做「解析下划线 flag → 转成连字符 flag 拼进 pretrain_gpt 命令」这一步。真实链路三层:

```
① launcher  解析 --hcu_probe_config=xxx(下划线,=号)→ mpirun/torchrun 透传
② train 脚本 再解析 → 读 cfg 的 level,若 L1 自动 TORCHDYNAMO_DISABLE=1
            → 组装 HCU_PROBE_ARGS=(--hcu-probe-config xxx)   ← 下划线转连字符
③ pretrain_gpt.py  ${HCU_PROBE_ARGS[@]}  → megatron argparse 收 --hcu-probe-config
```

**启动脚本要加的 3 段**(从已接入的脚本提炼,直接抄):

```bash
# (a) 解析自定义 flag —— 加进脚本已有的入参循环(hcu_megatron train 脚本本就有
#     `for para in $*` 的 if/elif 链解析 --data_path 等,照同样写法加 3 个分支):
elif [[ $para == --hcu_probe_config* ]];        then hcu_probe_config=${para#*=}
elif [[ $para == --hcu_probe_seed* ]];          then hcu_probe_seed=${para#*=}
elif [[ $para == --hcu_probe_monitor_config* ]];then hcu_probe_monitor_config=${para#*=}

# (b) L1 自动关 dynamo(读 cfg 的 level 判断,不用用户手动)
if [[ -n "${hcu_probe_config}" && -f "${hcu_probe_config}" ]]; then
  level=$(python3 -c "import json;print(json.load(open('${hcu_probe_config}'))['level'])" 2>/dev/null)
  [[ "${level}" == "L1" ]] && export TORCHDYNAMO_DISABLE=1
fi

# (c) 下划线 flag → megatron 连字符 flag,作为一个参数数组拼进 pretrain_gpt 命令
HCU_PROBE_ARGS=()
[[ -n "${hcu_probe_config}" ]]         && HCU_PROBE_ARGS+=(--hcu-probe-config "${hcu_probe_config}")
[[ -n "${hcu_probe_seed}" ]]           && HCU_PROBE_ARGS+=(--hcu-probe-seed "${hcu_probe_seed}")
[[ -n "${hcu_probe_monitor_config}" ]] && HCU_PROBE_ARGS+=(--hcu-probe-monitor-config "${hcu_probe_monitor_config}")

# 把 ${HCU_PROBE_ARGS[@]} 加进你脚本组装 pretrain_gpt 命令的地方。
# hcu_megatron train 脚本是把命令拼成 APP 字符串再启动,所以加在 APP 的参数数组列表里:
#   APP="python -u ${MEGATRON_PATH}/pretrain_gpt.py \
#       ${GPT_MODEL_ARGS[@]} ${TRAINING_ARGS[@]} ... \
#       ${HCU_PROBE_ARGS[@]}"          # ← 加这一行(与其他 *_ARGS 数组并列)
#   ${launch_with_binding} ${LOCAL_RANK} ${APP}
```

**5 模式调用**(launcher 层用下划线 flag;不传 = baseline 原训练):

```bash
# baseline:不加 = 原训练
bash <你的launcher>.sh

# L0 dump
bash <你的launcher>.sh --hcu_probe_config=$(realpath config/hcu_probe_config_statistics_L0.json)

# L1 dump(launcher 会自动关 dynamo;另需 Step 4 yaml patch)
bash <你的launcher>.sh --hcu_probe_config=$(realpath config/hcu_probe_config_statistics_L1.json)

# monitor 三通道
bash <你的launcher>.sh --hcu_probe_monitor_config=$(realpath config/monitor_v2_config_full_probe.json)

# combined:两个 flag 都给
bash <你的launcher>.sh --hcu_probe_config=<L0.json> --hcu_probe_monitor_config=<monitor.json>
```

> 上面 (a)(b)(c) 三段嵌进你**现有的** megatron 启动脚本即可,不新写 launcher、不依赖外部仓。**(a)(b)(c) 加在真正组装 `pretrain_gpt.py` 命令的那个脚本**(hcu_megatron 是 `train_*.sh`:本就有 `for para in $*` 的 if/elif 解析 `--data_path`,APP 字符串拼 8 个 `*_ARGS` 数组)。
>
> ⚠️ **若你的启动是「外层 `run_*.sh`(硬编码变量 + mpirun 调内层) + 内层 `train_*.sh`(拼 pretrain_gpt)」两层结构**:flag 要**穿两层**——外层加变量 + 在 mpirun 调内层的命令串里把 `--hcu_probe_config=$xxx` 透传下去,内层才能用 (a) 收到。否则 `bash run_*.sh --hcu_probe_config=…` 会被外层静默忽略(它通常只解析自己认识的 flag)。单层脚本无此问题。

---

## Step 4: L1 dump 时 patch `support_wrap_ops.yaml`(仅 L1)

**现象**:L1 cfg 跑 hcu_megatron 报 `ValueError: Only support fusion of gelu and swiglu`(Megatron `mlp.py`)。
**根因**:Megatron mcore mlp 用 `self.activation_func is F.silu` 做 identity 检查;hcu_probe L1 wrap `F.silu` 后 identity 失败。

**解**:注释 `support_wrap_ops.yaml` 里 **`functional:` 区**(文件靠前,`functional:` 标题下、`tensor:` 区之前)的 `gelu`/`silu` 两行。⚠️ 这两个词在文件里出现多次(还有 `torch:` 区的、`npu_*` 的),**只注释 functional 区那两行**,别误注释别处。

```yaml
functional:
  ...
  # - gelu      ← 注释
  ...
  # - silu      ← 注释
```

**改哪个文件**(看你怎么装的 hcu_probe):
- **wheel 装的(常见)**:直接改**安装位置**的 yaml,改完即生效、**不用重打 whl**:
  `python -c "import hcuprobe,os;print(os.path.dirname(hcuprobe.__file__))"` 找到包目录 → 改其下 `pytorch/dump/api_dump/support_wrap_ops.yaml`。容器里多个 rank 共用一份,改一次即可。
- **源码 editable(`pip install -e`)装的**:改 `python/hcuprobe/.../support_wrap_ops.yaml` 即生效。

> hcu_probe 主仓 yaml 与上游字字一致(红线),这是 **hcu_megatron 专属本地改**,用完还原(`git checkout` 或反注释),别提交进主仓。

> ⚠️ **distributed 段:非 MoE 不要注释**。`support_wrap_ops.yaml` 还有 `distributed:` 段(send/recv/all_gather/all_to_all 等通信算子)。
> - **非 MoE(如 llama2-7B,TP/PP 无 EP)**:**只注释 gelu/silu 就够** —— 实测注释 distributed 与否,L1 逐 iter 数值字字一致,PP 的 `batch_isend_irecv` 被 wrap 也不破坏。**别多注释**。
> - **MoE/EP 模型(如 mixtral)**:才需连 `distributed:` 段一起注释 —— EP 的 `all_to_all` 时序敏感,L1 wrap 会破坏。
>
> 依据见 [`references/runbook.md`](references/runbook.md) 坑 3。

**副作用**:L1 dump.json 内不会有 `Functional.silu`/`Functional.gelu` entries(精度-兼容性 trade-off)。

---

## Step 5: 确认真启动(看 marker,不只看 wrapper applied)

跑起来后 grep log,**必须看到 `initialized`/`started` 才算真启动**(只看 `wrapper applied` 是假象 —— monitor 缺 tensorboard 会静默 init 失败但训练 loss 正常,见 runbook 坑 1):

```
[hcu_probe.dump] PrecisionDebugger initialized with <cfg>        ← dump 真起
[hcu_probe.monitor] TrainerMonitorV2 initialized and started     ← monitor 真起
```

产物(dump.json / monitor csv)默认 0o750,docker 内 `chmod -R 755 <dump目录>` 让 host 可读。产物目录结构 + 验证三件套见 [`references/runbook.md`](references/runbook.md)。

---

## Quick Reference

| 项 | 内容 |
|---|---|
| 适配文件 | `scripts/{hcu_probe,monitor}_{feature,wrapper}.py`(拷进 fork 的 `features_manager/probe/`) |
| CLI flag | `--hcu-probe-config=<dump.json>` / `--hcu-probe-monitor-config=<monitor.json>` / `--hcu-probe-seed=42` |
| 翻译层(whl 自带) | `hcuprobe.integrations.megatron.{dump,monitor}` |
| cfg 模板 | 本 skill `config/`(L0 / L1 / monitor 三通道) |
| L1 yaml patch | `support_wrap_ops.yaml` 注释 gelu/silu(**非 MoE 仅此;MoE 才加 distributed 段**) |
| 6 坑 + 验证 + 实测状态 | [`references/runbook.md`](references/runbook.md) |

姊妹 skill:verl 训推一致性(`hcu_probe-verl-fsdp-v07-integration` / `hcu_probe-verl-fsdp-v08-integration` / `hcu_probe-verl-megatron-v08-integration`)、`hcu_probe-pytorch-integration`(原生 PyTorch / 自定义 train loop)。
