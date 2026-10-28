# hcuprobe Megatron 场景快速入门

<br>

## 1. 概述

本文以 **Megatron-LM-das（hcu_megatron，core_v0.18.2）** 在 HCU 上预训练 **Qwen3-0.6B** 为例，演示如何不改训练脚本、只在启动命令中加 hcuprobe 的命令行参数，完成：

- **L0 精度数据采集**：每个 Module 前向/反向的输入输出统计量，用于定位"哪一层开始不一致"；
- **Monitor 训练状态监控**：权重梯度、优化器状态（Adam m/v）、参数三通道的 min/max/mean/norm/nan 统计，用于发现 NaN/梯度爆炸等异常。

接入原理：hcu_megatron 通过 `features_manager` 中的 `ADAPTOR_FEATURES` 列表以 monkey-patch 方式扩展 Megatron。hcuprobe 提供两个 Feature（`HcuProbeFeature`、`MonitorFeature`），注册后会新增 `--hcu-probe-config`、`--hcu-probe-monitor-config` 参数，并把采集逻辑挂到 Megatron 的 `train` / `train_step` / `setup_model_and_optimizer` 上。不传参数时行为与原始训练完全一致。

**体验地图**

| 步骤 | 环节 | 核心工具 | 说明 |
| :---: | :--- | :--- | :--- |
| 1 | 环境准备 | Megatron-LM-das 官方镜像 + hcuprobe whl | 拉镜像、起容器、拉代码、装 whl |
| 2 | 模型与数据 | modelscope + `preprocess_data.py` | Qwen3-0.6B 分词器、oscar-en-10k |
| 3 | 基线训练 | 官方 `run_qwen.sh` + `train_qwen3_0.6B.sh` | 确认训练能跑通 |
| 4 | 接入 hcuprobe | 4 个 Feature 薄壳文件 + `ADAPTOR_FEATURES` 注册 | 一次性改动 |
| 5 | L0 采集 | `--hcu-probe-config` | 产出 `dump.json` |
| 6 | Monitor 监控 | `--hcu-probe-monitor-config` | 产出三通道 CSV |
| 7 | L1 采集 / combined（可选） | `level: L1` + `support_wrap_ops.yaml` 修改 | API 级数据，精度预检的输入 |
| 8 | 离线分析 | `hcuprobe compare` / `overflow_check` / `graph_visualize` / `multi_acc_check` | 比对 → 定位首个异常节点 → 可视化 → API 精度预检 |

### 1.1 版本清单（复现必须一致）

| 组件 | 版本 / 地址 |
| --- | --- |
| 容器镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron-0.18.2-ubuntu22.04-dtk2604-py3.10-20260908-0350`（digest `sha256:7a0aafdc5367…`，即 Megatron-LM-das getting-started 推荐镜像） |
| Megatron-LM-das | <https://github.com/HYGON-AI/Megatron-LM-das> 分支 `core_v0.18.2`，commit `52b4450`（子模块 Megatron-LM `571370c`） |
| hcuprobe | <https://github.com/HYGON-AI/hcu_probe-das> 分支 `26.2.0-dev`（发行名 `hcu-probe`，import 名 `hcuprobe`） |
| 接入文件 | hcu_probe-das 仓 `.claude/skills/hcu_probe-megatron-integration/`（`scripts/` 4 个薄壳，`config/` 配置模板） |

## 2. 操作步骤

### 2.1 环境准备（必做）

#### 2.1.1 前置条件

| 项目 | 要求 | 验证方法 |
| --- | --- | --- |
| 硬件 | 1 台 8 卡 HCU（BW 系列，单卡 64 GB） | `hy-smi` 能列出 8 张卡 |
| 容器 | 已安装 Docker，可访问 `harbor.sourcefind.cn:5443` | `docker ps` 无报错 |
| 网络 | 容器内可访问 GitHub、PyPI 镜像、ModelScope | 无外网时参考第 3 章 |

#### 2.1.2 宿主机：准备代码、拉取镜像、启动容器

```bash
export WORK=/path/to/hcuprobe_verify          # 修改为你的工作目录
mkdir -p $WORK/{src,models,data,runs}
cd $WORK/src
git clone -b 26.2.0-dev https://github.com/HYGON-AI/hcu_probe-das.git
git clone -b core_v0.18.2 https://github.com/HYGON-AI/Megatron-LM-das.git
cd Megatron-LM-das && git checkout 52b4450 && git submodule update --init --recursive

export IMG=harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron-0.18.2-ubuntu22.04-dtk2604-py3.10-20260908-0350
docker pull $IMG
docker run -itd --name hcuprobe_mega --shm-size=64G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  --ipc=host --network=host --group-add video --privileged \
  -v /opt/hyhal:/opt/hyhal:ro -v $WORK:/workspace -w /workspace \
  $IMG /bin/bash
docker exec -it hcuprobe_mega bash
```

以下命令如无特别说明，均在**容器内**执行。

#### 2.1.3 容器内：安装依赖与 hcuprobe

```bash
git config --global --add safe.directory '*'
cd /workspace/src/Megatron-LM-das
pip install -r ./requirements/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# getting-started 要求:python3.10 下把 `from typing import override` 换成 typing_extensions
python3 - << 'EOF'
import re, pathlib
for f in pathlib.Path("/workspace/src/Megatron-LM-das").rglob("*.py"):
    s = f.read_text(errors="ignore")
    m = re.search(r"^from typing import ([^\n]*\boverride\b[^\n]*)$", s, re.M)
    if not m: continue
    names = [n.strip() for n in m.group(1).split(",") if n.strip() != "override"]
    new = ("from typing import " + ", ".join(names) + "\n" if names else "") + "from typing_extensions import override"
    f.write_text(s[:m.start()] + new + s[m.end():]); print("patched", f)
EOF

# 源码构建 hcuprobe whl(whl 文件名是 hcu_probe-*.whl)
cd /workspace/src/hcu_probe-das
pip wheel . --no-deps -w dist/ && pip install dist/hcu_probe-*.whl --no-deps
pip freeze | grep -iE "^(numpy|torch|transformers|transformer.engine|flash.attn)==" > /tmp/constraints.txt
pip install -c /tmp/constraints.txt einops pandas pyyaml tqdm openpyxl matplotlib tensorboard skl2onnx \
  packaging pytz psutil h5py modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> [!NOTE]说明
>
> `tensorboard` 必须安装：hcuprobe 的 Monitor 依赖它，缺失时 Monitor 会静默初始化失败（训练照常、但没有任何 CSV 输出）。

#### 2.1.4 容器内：检查环境

```bash
python3 -c "import torch, hcuprobe, tensorboard; from hcuprobe.integrations.megatron import dump, monitor; \
assert torch.cuda.device_count()==8; print('torch', torch.__version__)" \
&& echo -e "\e[32m[PASS] HCU / hcuprobe megatron 集成层检查通过\e[0m"
```

### 2.2 准备模型与数据（getting-started 步骤 4~6）

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir /workspace/models/Qwen3-0.6B
modelscope download --dataset kanliu/oscar-en-10k-megatron oscar-en-10k.jsonl --local_dir /workspace/data/oscar-en-10k
cd /workspace/src/Megatron-LM-das
python 3rdparty/Megatron-LM/tools/preprocess_data.py \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model /workspace/models/Qwen3-0.6B \
  --input /workspace/data/oscar-en-10k/oscar-en-10k.jsonl \
  --output-prefix /workspace/data/oscar-en-10k/oscar-en-10k-qwen3 --append-eod --workers 16
ls /workspace/data/oscar-en-10k/oscar-en-10k-qwen3_text_document.{bin,idx}
```

### 2.3 基线训练

按 getting-started 步骤 7 配置官方脚本：

```bash
cd /workspace/src/Megatron-LM-das/examples/qwen3
# 1) env.sh:GLOO/NCCL_SOCKET_IFNAME 改成本机实际网卡(默认 eth0;下面取默认路由所在网卡)
IFNAME=$(awk '$2=="00000000"{print $1; exit}' /proc/net/route)
sed -i "s/^export GLOO_SOCKET_IFNAME=.*/export GLOO_SOCKET_IFNAME=$IFNAME/; s/^export NCCL_SOCKET_IFNAME=.*/export NCCL_SOCKET_IFNAME=$IFNAME/" ../../requirements/env.sh
# 2) run_qwen.sh:填写路径,训练脚本选 Qwen3-0.6B
sed -i 's#^DTK_ENV=""#DTK_ENV="/opt/dtk/env.sh"#; s#^DATA_PATH=""#DATA_PATH="/workspace/data/oscar-en-10k/oscar-en-10k-qwen3_text_document"#; s#^TOKENIZER_MODEL_PATH=""#TOKENIZER_MODEL_PATH="/workspace/models/Qwen3-0.6B"#; s#^CHECKPOINT_PATH=""#CHECKPOINT_PATH="/workspace/runs/qwen3_06b/ckpt"#; s#^TRAIN_SCRIPT=.*#TRAIN_SCRIPT=train_qwen3_0.6B.sh#' run_qwen.sh
# 3) train_qwen3_0.6B.sh:缩短到 10 步;修正参数名(该脚本误写 --use-qk-norm,其余 qwen3 脚本均为 --qk-layernorm)
sed -i 's/^TRAIN_ITERS=.*/TRAIN_ITERS=10/; s/--use-qk-norm/--qk-layernorm/' train_qwen3_0.6B.sh
echo localhost > hostfile
bash run_qwen.sh hostfile 1
```

**预期结果（实测）**：

```text
iteration  1/10 | lm loss: 1.194700E+01 | grad norm: 16.801
iteration  2/10 | lm loss: 1.194701E+01 | grad norm: 16.502
iteration  3/10 | lm loss: 1.171801E+01 | grad norm: 39.350
iteration  5/10 | lm loss: 1.147737E+01 | grad norm: 2.967
...                                                         # 稳态约 27 s/iter(GBS=256,seq 4096,TP4)
```

> [!NOTE]说明
>
> 若不修正 `--use-qk-norm`，会报 `pretrain_gpt.py: error: unrecognized arguments: --use-qk-norm`；若不改网卡，会报 `Unable to find address for: eth0`。

### 2.4 接入 hcuprobe（一次性）

```bash
SK=/workspace/src/hcu_probe-das/.claude/skills/hcu_probe-megatron-integration
M=/workspace/src/Megatron-LM-das
mkdir -p $M/hcu_megatron/features_manager/probe
cp $SK/scripts/{hcu_probe_feature,hcu_probe_wrapper,monitor_feature,monitor_wrapper}.py $M/hcu_megatron/features_manager/probe/
```

编辑 `$M/hcu_megatron/features_manager/__init__.py`，在 import 区和 `ADAPTOR_FEATURES` 列表末尾各加两行（**只拷文件不注册会静默无效**）：

```python
from .probe.hcu_probe_feature import HcuProbeFeature
from .probe.monitor_feature import MonitorFeature

ADAPTOR_FEATURES = [
    ...,
    MegaMoeFeature(),
    HcuProbeFeature(),
    MonitorFeature(),
]
```

| 文件 | 作用 |
| --- | --- |
| `hcu_probe_feature.py` | 注册 `--hcu-probe-config` / `--hcu-probe-seed`，把 dump wrapper 挂到 `train`、`train_step` |
| `hcu_probe_wrapper.py` | 调用 `hcuprobe.integrations.megatron.dump.on_train_start / on_step_start / on_step_end / on_train_end` |
| `monitor_feature.py` | 注册 `--hcu-probe-monitor-config`，把 monitor wrapper 挂到 `setup_model_and_optimizer`、`train`、`train_step` |
| `monitor_wrapper.py` | 调用 `hcuprobe.integrations.megatron.monitor.on_setup_done / on_step_end / on_train_end` |

检查注册：

```bash
cd $M && PYTHONPATH=$M:$M/3rdparty/Megatron-LM python3 -c \
"from hcu_megatron.features_manager import ADAPTOR_FEATURES; print([type(f).__name__ for f in ADAPTOR_FEATURES][-2:])"
# 预期输出 ['HcuProbeFeature', 'MonitorFeature']
```

### 2.5 L0 精度数据采集

#### 2.5.1 准备采集配置

```bash
mkdir -p /workspace/runs/qwen3_06b_L0
cat > /workspace/runs/qwen3_06b_L0/hcuprobe_L0.json << 'EOF'
{
    "task": "statistics",
    "dump_path": "/workspace/runs/qwen3_06b_L0/hcuprobe_dump/statistics_L0",
    "rank": [],
    "step": [0],
    "level": "L0",
    "async_dump": false,
    "statistics": {"scope": [], "list": [], "data_mode": ["all"], "summary_mode": "statistics"}
}
EOF
```

> [!NOTE]说明
>
> 官方 0.6B 配置为 GBS=256、MBS=1、TP=4，即每个 step 有 128 个 micro batch，L0 会逐个 micro batch 记录全部 Module 的前向/反向/参数梯度。实测每个被采集的 step 单 rank 约 13.5 万条记录、dump.json 约 120 MB，8 卡合计约 1.1 GB，且序列化为 CPU 密集操作（约 15 分钟/step）。因此入门建议 `"step": [0]`；需要多步时可同时调小 `GBS`，或用 `rank` 只采部分卡。

#### 2.5.2 启动采集

在 `train_qwen3_0.6B.sh` 的 `pretrain_gpt.py` 参数中追加（mpirun 分支即 `${INITIALIZATION_ARGS[@]}` 之后）：

```bash
sed -i 's#^\(\s*\)\${INITIALIZATION_ARGS\[@\]} \\#\1${INITIALIZATION_ARGS[@]} --hcu-probe-config /workspace/runs/qwen3_06b_L0/hcuprobe_L0.json --hcu-probe-seed 1234 \\#' train_qwen3_0.6B.sh
sed -i 's#^CHECKPOINT_PATH=.*#CHECKPOINT_PATH="/workspace/runs/qwen3_06b_L0/ckpt"#' run_qwen.sh
bash run_qwen.sh hostfile 1
```

**预期结果**：日志中依次出现

```text
[hcu_probe-wrapper] train() wrapper applied via Feature mechanism.
[hcu_probe-wrapper] train_step() wrapper applied via Feature mechanism.
[hcu_probe.dump] PrecisionDebugger initialized ...
```

> ⚠️ 只看到 `wrapper applied` 不代表采集真正启动，必须看到 `initialized`。

**产物（实测）**：

```text
/workspace/runs/qwen3_06b_L0/hcuprobe_dump/statistics_L0/
├── step0/
│   ├── rank0/{dump.json, stack.json, construct.json}     # dump.json 约 120 MB
│   ├── ...
│   └── rank7/
└── loss_tracking/                                          # 每步 loss / grad_norm
```

`dump.json` 中的键形如：

```text
Module.module.module.embedding.word_embeddings.VocabParallelEmbedding.forward.0
Module.module.module.decoder.layers.0.self_attention.linear_qkv.TELayerNormColumnParallelLinear.forward.0
...
Module.module.module.embedding.word_embeddings.VocabParallelEmbedding.parameters_grad.127
```

末尾数字是 micro batch 序号（0~127）。采集不改变数值：开启 L0 后第 1 步 `lm loss 1.194700E+01 / grad norm 16.801`，与 2.3 节基线完全一致。

得到两份 L0 数据（例如同一脚本在两种环境或两个版本上各跑一次）后，可用 `hcuprobe compare` 做逐层精度比对，参见《[精度比对](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)》。

### 2.6 Monitor 训练状态监控

```bash
mkdir -p /workspace/runs/qwen3_06b_monitor
cat > /workspace/runs/qwen3_06b_monitor/hcuprobe_monitor.json << 'EOF'
{
  "framework": "pytorch",
  "output_dir": "/workspace/runs/qwen3_06b_monitor/hcuprobe_dump/monitor_v2",
  "format": "csv",
  "patch_optimizer_step": false,
  "monitors": {
    "weight_grad": {"enabled": true, "ops": ["min", "max", "mean", "norm", "nans"]},
    "optimizer":   {"enabled": true, "mv_distribution": true, "ops": ["min", "max", "mean", "norm", "nans"]},
    "param":       {"enabled": true, "ops": ["min", "max", "mean", "norm", "nans"]}
  }
}
EOF
# 把 train_qwen3_0.6B.sh 中追加的参数换成 monitor(两者也可同时传入)
sed -i 's#--hcu-probe-config [^ ]* --hcu-probe-seed 1234#--hcu-probe-monitor-config /workspace/runs/qwen3_06b_monitor/hcuprobe_monitor.json#' train_qwen3_0.6B.sh
sed -i 's#^CHECKPOINT_PATH=.*#CHECKPOINT_PATH="/workspace/runs/qwen3_06b_monitor/ckpt"#' run_qwen.sh
bash run_qwen.sh hostfile 1
```

| 配置项 | 说明 |
| --- | --- |
| `patch_optimizer_step: false` | **必须**。Megatron 的 DistributedOptimizer → ChainedOptimizer 嵌套会让自动 patch 落空，由 wrapper 在 `on_step_end` 手动驱动 |
| 不开 `module` 通道 | module 通道与 Megatron 流水线的 `deallocate_pipeline_outputs` 冲突，只开 weight_grad / optimizer / param |

**预期结果**：日志出现 `[hcu_probe.monitor] TrainerMonitorV2 initialized and started (grad_acc_steps=128).`，结束时出现 `TrainerMonitorV2 stopped.`。

**产物（实测，4 步）**：每个 rank 每步 3 个 CSV，共 8 × 3 × 4 = 96 个文件：

```text
/workspace/runs/qwen3_06b_monitor/hcuprobe_dump/monitor_v2/rank_<r>/
    weight_grad_step<N>-<N>.csv
    optimizer_step<N>-<N>.csv
    param_step<N>-<N>.csv
```

以 `weight_grad_step3-3.csv` 为例：

```text
vpp_stage,step,module_name,scope,micro_step,min,max,mean,norm,nans
-1,3,module.module.output_layer.weight,unreduced,128,-0.0872,0.0906,6.33e-07,2.5448,0.0
-1,3,module.module.decoder.final_layernorm.weight,unreduced,128,-0.00073,0.00014,-8.36e-05,0.00445,0.0
```

重点关注 `nans` 列（非 0 即出现 NaN）和 `norm` 列的突变。Monitor 同样不改变训练数值：4 步 loss 与基线逐位一致。

### 2.7 L1 精度数据采集（API 级，可选）

L1 在 L0 基础上进一步记录 `Tensor.*` / `Torch.*` / `Functional.*` / `Distributed.*` 等 API 调用，粒度更细、数据量大得多，入门时建议把 GBS 调小（下例改为 16，即每步 8 个 micro batch）。

**必做前置**：注释已安装 hcuprobe 的 `support_wrap_ops.yaml` 中 `functional:` 区的 `gelu` / `silu` 两行（只改 `functional:` 区，其他区同名条目不动）。否则 L1 会替换 `F.silu`，Megatron MLP 的 `activation_func is F.silu` 判断失败，实测报 `Only support fusion of gelu and swiglu` 后训练崩溃。

```bash
Y=$(python3 -c "import hcuprobe,os;print(os.path.join(os.path.dirname(hcuprobe.__file__),'pytorch/dump/api_dump/support_wrap_ops.yaml'))")
cp $Y $Y.orig
python3 - "$Y" << 'EOF'
import sys
p = sys.argv[1]; lines = open(p).read().split("\n"); sec = None
for i, l in enumerate(lines):
    if l and not l.startswith(" ") and l.endswith(":"): sec = l[:-1]
    if sec == "functional" and l.strip() in ("- gelu", "- silu"): lines[i] = l.replace("- ", "# - ", 1)
open(p, "w").write("\n".join(lines))
EOF
grep -nE "^  # - (gelu|silu)$" $Y      # 预期两行;用完 cp $Y.orig $Y 还原
```

配置与启动（`level` 改为 `L1`；L1 与 torch dynamo 冲突，需在训练脚本中 `export TORCHDYNAMO_DISABLE=1`）：

```bash
mkdir -p /workspace/runs/qwen3_06b_L1
sed 's#"level": "L0"#"level": "L1"#; s#qwen3_06b_L0#qwen3_06b_L1#; s#statistics_L0#statistics_L1#' \
  /workspace/runs/qwen3_06b_L0/hcuprobe_L0.json > /workspace/runs/qwen3_06b_L1/hcuprobe_L1.json
sed -i '1a export TORCHDYNAMO_DISABLE=1' train_qwen3_0.6B.sh
sed -i 's/^GBS=.*/GBS=16/' train_qwen3_0.6B.sh
sed -i 's#--hcu-probe-[a-z-]*config [^ ]*\( --hcu-probe-seed 1234\)\?#--hcu-probe-config /workspace/runs/qwen3_06b_L1/hcuprobe_L1.json --hcu-probe-seed 1234#' train_qwen3_0.6B.sh
sed -i 's#^CHECKPOINT_PATH=.*#CHECKPOINT_PATH="/workspace/runs/qwen3_06b_L1/ckpt"#' run_qwen.sh
bash run_qwen.sh hostfile 1
```

**产物（实测，GBS=16，step0）**：8 个 rank 合计约 56 MB；rank0 的 `dump.json` 约 7200 条，构成为 `Tensor.*` 3849、`Distributed.*` 2355、`Torch.*` 513、`Functional.*` 472（最多的是 `Tensor.__mul__`、`Distributed.all_gather_into_tensor`、`Distributed.reduce_scatter_tensor`）。按预期不含 `Functional.silu` / `Functional.gelu`（已从 wrap 列表中注释）。L1 数据也是第 2.9.4 节精度预检的输入。

### 2.8 同时开启 L0 与 Monitor（combined）

两个参数可以同时传入：

```bash
--hcu-probe-config /workspace/runs/qwen3_06b_L0/hcuprobe_L0.json --hcu-probe-seed 1234 \
--hcu-probe-monitor-config /workspace/runs/qwen3_06b_monitor/hcuprobe_monitor.json
```

实测（GBS=16）日志中 `PrecisionDebugger initialized` 与 `TrainerMonitorV2 initialized and started (grad_acc_steps=8)` 均出现，L0 数据与 Monitor CSV 同时产出，loss 与同 GBS 基线一致（1.195066E+01 / 1.194816E+01 / 1.1718E+01）。

### 2.9 离线分析：比对、溢出定位、可视化、精度预检

以下命令只读 dump 数据、不占用加速卡（精度预检除外），均在容器内执行。示例数据为同一配置（GBS=16、`--hcu-probe-seed 1234`）跑两次得到的 L0 数据 `runA`、`runB`（各采 step0、step1）。

```bash
A=/workspace/runs/runA/hcuprobe_dump/statistics_L0
B=/workspace/runs/runB/hcuprobe_dump/statistics_L0
O=/workspace/runs/offline_analysis
```

#### 2.9.1 两份数据精度比对：`hcuprobe compare`

```bash
hcuprobe compare -tp $A/step0 -gp $B/step0 -o $O/compare_A_vs_B
```

`-tp` 为待测数据、`-gp` 为标杆数据，传入 step 目录即按 rank 逐一比对，输出 `compare_result_rank<N>_<时间戳>.csv`（8 个 rank 共约 36 s）。实际使用中通常是"HCU vs GPU""新旧 DTK / 驱动""开关某个优化前后"等两份数据。

**实测分析（rank0，32744 行，全部 `pass`）**：

| 条目类型 | 行数 | 完全相同 | NormRelativeErr 最大 |
| --- | --- | --- | --- |
| `.forward.` | 23448 | 23448 | 0% |
| `.backward.` | 7480 | 7310 | 0.76% |
| `.parameters_grad.` | 1816 | 36 | 0.27% |

结论：两次运行的前向逐位一致，差异只出现在反向，且从最后一层（`layers.27`）的梯度开始，符合反向梯度规约中原子加的非确定性，不是精度问题。分析时可按此思路：**先看第一个出现差异的条目在前向还是反向、在哪一层，再结合 `NormRelativeErr` / `MaxRelativeErr` 判断量级。**

#### 2.9.2 首个异常节点定位：`hcuprobe overflow_check`

```bash
hcuprobe overflow_check -i $A/step0 -o $O/overflow_clean
```

正常数据会输出 `Cannot find any anomaly node, no need to generate analyze file.`，不生成文件（这是正常现象）。

为演示定位能力，复制 runA 的 rank0 数据，把 `layers.10.mlp.linear_fc1` 的前向输出及其后 5 个模块的统计量改为 NaN，模拟 NaN 从该层开始向后传播：

```bash
hcuprobe overflow_check -i $O/nan_dump/step0 -o $O/overflow_nan
```

**实测结果**：生成 `anomaly_analyze_<id>.json`，`rank_0` 的第一个条目即为 `Module.module.module.decoder.layers.10.mlp.linear_fc1.TELayerNormColumnParallelLinear.forward.0`：输入统计量正常（Max 0.055、Norm 9.5），输出为 NaN。工具只报告"输入正常、输出异常"的起点，不会把下游被污染的模块一并报出，便于直接定位到出问题的层。产物字段说明见《[溢出分析结果解读](../user_guide/offline_analysis/overflow_result_analysis.md)》。

#### 2.9.3 分级可视化：`hcuprobe graph_visualize`

```bash
hcuprobe graph_visualize -tp $A/step0/rank0 -o $O/graph_single                       # 单图
hcuprobe graph_visualize -tp $A/step0/rank0 -gp $B/step0/rank0 -o $O/graph_diff     # 双图对比
```

实测分别生成 `build_<时间戳>.vis.db`（约 19 MB）与 `compare_<时间戳>.vis.db`（约 62 MB），各约 20~30 s。浏览方式见《[graph_viz](../user_guide/offline_analysis/graph_viz.md)》。

#### 2.9.4 API 精度预检：`hcuprobe multi_acc_check`

精度预检以 L1 数据为输入，在加速卡上逐个重放 API，并与 CPU fp64 标杆比较。rank0 的 step0 数据通常已足够：

```bash
hcuprobe multi_acc_check -api_info /workspace/runs/qwen3_06b_L1/hcuprobe_dump/statistics_L1/step0/rank0/dump.json \
  -o $O/acc_check -d 0 1 2 3 4 5 6 7
```

单卡 `hcuprobe acc_check -d 0` 约 1.9 API/s，8 卡 `multi_acc_check` 约 10 API/s（本例 6709 个 API 用时约 12 min）。启动时打印的 `Please verify that the CANN environment is properly configured.` 来自上游昇腾逻辑，在 HCU 上可忽略。

产物为 `accuracy_checking_result_<时间戳>.csv`（每个 API 的前向/反向结论）和 `accuracy_checking_details_<时间戳>.csv`（余弦相似度、双千指标、最大绝对误差、相对误差、ULP 等逐项指标）。

**实测分析（rank0 step0，6709 个 API）**：

| 前向结论 | 数量 | 说明 |
| --- | --- | --- |
| pass | 3028 | |
| warning | 1326 | `Torch.sigmoid` 448、`Tensor.__mul__` 422、`Tensor.__rsub__` 224、`Tensor.__radd__` 224、`Torch.matmul` 8 |
| error | 0 | |
| SKIP | 2355 | `Distributed.*` 通信 API 不支持预检 |

warning 条目的典型指标：设备侧 `bfloat16`、标杆 `float32`，余弦相似度 0.999998，相对误差最大值约 0.0039（即 bf16 的精度 2⁻⁸），双千指标 0.34~0.47。这是 bf16 计算相对高精度标杆的正常舍入误差，不是 HCU 算子缺陷。`sigmoid` / `__mul__` / `__rsub__` 大量出现，是因为 2.7 节注释了 `silu` 的 wrap 后，SwiGLU 在 Python 层被拆成了这些基础运算。

分析时优先看 **error** 和余弦相似度明显低于 0.9999 的条目；要区分"HCU 特有"还是"bf16 共性"，可在 GPU 上对同一份 L1 数据再跑一次预检，然后用 `hcuprobe api_precision_compare` 对比两边结果（参见《[precheck](../user_guide/offline_analysis/precheck.md)》）。

## 3. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ImportError: cannot import name 'override' from 'typing'` | python3.10 的 typing 无 `override` | 按 2.1.3 替换为 typing_extensions（getting-started 已说明） |
| `unrecognized arguments: --use-qk-norm` | `train_qwen3_0.6B.sh` 参数名笔误 | 改为 `--qk-layernorm` |
| `Unable to find address for: eth0` | `requirements/env.sh` 默认网卡为 eth0 | 改成本机实际网卡（2.3 节） |
| 训练正常但没有任何 dump / CSV | ① Feature 只拷贝未注册；② 缺 tensorboard（Monitor）；③ 旧版 hcuprobe 内部仍 import `hcu_probe.*` | ① 按 2.4 检查注册；② 安装 tensorboard；③ 更新 hcu_probe-das 并重建 whl |
| L1 采集报 `Only support fusion of gelu and swiglu` 后训练崩溃 | L1 会 wrap `F.silu`，破坏 Megatron MLP 的融合判断 | 按 2.7 节注释 `support_wrap_ops.yaml` 中 `functional:` 区的 `- gelu` / `- silu` |
| L0 每个采集 step 耗时很长（GBS=256 时约 15 min） | 每个 micro batch 都记录全部 Module | 只采 `"step": [0]`，或调小 GBS / 用 `rank` 只采部分卡 |
| `overflow_check` 没有生成文件 | 数据中没有 NaN/Inf，属正常 | 日志显示 `Cannot find any anomaly node` 即为通过 |
| `risk_level` 报 invalid | 必须大写 `ALL` / `FOCUS` / `CORE`，且只对 L1 生效 | 修改配置 |
