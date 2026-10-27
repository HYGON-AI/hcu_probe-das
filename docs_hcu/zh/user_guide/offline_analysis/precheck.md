# precheck — API 精度预检三件套

`acc_check` / `multi_acc_check` / `api_precision_compare`:基于 L1 dump 的算子级精度体检。

工作原理:从 L1 dump 提取每个 PyTorch API 调用的 shape + dtype 签名 → 按签名生成随机输入 → 在 **当前 device** + **CPU fp64** 上分别执行同一算子 → 多维度(余弦 / 双百 / 双千 / ULP)对比 → 判 Pass / Warning / Error。

## 场景范围(只支持 PyTorch 训练)

hcu_probe 的 precheck 工具**只支持 PyTorch 训练场景**:

- ✅ dump.json `"framework": "pytorch"`,API 名形如 `Functional.linear.0` / `Tensor.add_.0` / `Torch.matmul.0`
- ❌ 不支持 **MindSpore 动态图**(`"framework": "mindspore"` + Mint API)— 那是华为生态的训练框架,跟 HCU 走 PyTorch 不是一条路
- ❌ 不支持 **MSAdapter**(PyTorch→MindSpore 兼容层,实质是为了让 PyTorch 代码跑在 NPU)
- ❌ 不支持 **推理场景**(vLLM / 离线模型),`offline_dump` 子命令已从 hcu-probe CLI 砍掉

上游 mindstudio-probe README 列的 5 大场景里 hcu_probe 只对应 1 个:**PyTorch 训练**。

## HCU 上的实证

### ✅ 2026-05-13 v26.0.0-alpha.1 + hcu_probe whl 实证(本仓库)

scnet docker `hcu_probe_lab`(f09r1n12)+ 主机 conda env `megatron_train`,3 段实证锁死 acc_check 在 HCU 上**真用 HCU**:

| 实证 | 数据 | 结论 |
|---|---|---|
| **sanity matmul** 30s loop | HCU **80-100%** / 55.37 TFLOPS | hy-smi 工具靠谱 |
| **`ApiTemplate.forward` 30s loop**(走 acc_check 真实算子路径)| HCU **持续 100%** / 54.25 TFLOPS | acc_check 框架真路径 100% 用 HCU |
| **`ApiTemplate.forward` monkey-patch trace**(跑 `hcu-probe acc_check`,前 30 次 forward)| 完美交替模式:奇数次 `cpu`(CPU fp64 标杆) / 偶数次 `cuda:0`(HCU 实跑) | 每个 API 真跑 2 次,HCU bf16 vs CPU fp64 真比对 |

docker partial 24986/61618 API(40%)csv:
- **21378 Pass(85.6%)** + 2082 Warning(8.3%)+ 1525 SKIP(6.1%)
- 比例跟 2026-04-13 8.3.3 历史实证(87.4% / 11.1% / 1.5%)一致

### ⚠️ multi_acc_check 时 hy-smi 显示 HCU 0% 是**正常的**

```
每 worker 跑单 API 的时间分布:
  ├── 数据生成 (CPU)         ~50 ms
  ├── tensor.to(cuda)        ~5 ms
  ├── HCU 算子执行           ~5-20 ms   ← GPU 真工作(占 <10%)
  ├── output.to(cpu)         ~10 ms
  ├── CPU fp64 标杆          ~100 ms    ← CPU bound,大头
  ├── 多维度比对算法 (CPU)    ~50 ms
  └── hook overhead          ~50 ms
  ─────────────────────────────────
  总 ~270 ms,GPU 工作占 < 10%
```

multi-process 32-64 worker × 8 卡 → 每卡每个采样窗口落在 GPU idle 间隙概率 > 95% → hy-smi 显示 0%。**这是采样 miss,不是 HCU 没用**。

### 怎么验证 HCU 真用了(3 个有效方法)

| 方法 | 命令 | 验证点 |
|---|---|---|
| **trace 法**(最严谨) | monkey-patch `ApiTemplate.forward` 加 `print(args[0].device)`,跑 `hcu-probe acc_check` 30 次 forward | 看到 `cpu` / `cuda:0` 交替模式 → HCU 真用 |
| **TFLOPS 法**(简单) | 写脚本 `ApiTemplate.forward` 大算子 loop 30s | HCU 100% / 50+ TFLOPS → cuda dispatch work |
| **csv 比例对照法**(参考) | 跑完 multi_acc_check 看 csv 第 2 列 SKIP/Pass/Warning 分布 | 跟 2026-04-13 历史比例对齐,印证 HCU bf16 vs CPU fp64 真比对 |

⚠️ **不要**只用 `hy-smi` 一次性快照判断 — 大概率 0%,但**不代表 broken**。

### 历史实证(2026-04-13 mindstudio-probe 8.3.3 独立工具)

scnet node119 BW200 × 8(单独 conda env + pip install mindstudio-probe==8.3.3),80 分钟跑 46258 API,0 Error / 5120 Warning(全部 bf16 共性 — sigmoid/mul/rsub 双千)/ 681 SKIP。本次 2026-05-13 在 hcu_probe whl(v26.0.0-alpha.1)上重新跑,比例一致,印证**一个 whl 同时支持训练 dump + 离线 precheck** 这一愿景达成。

## Python API

```python
from hcu_probe.tools import precheck

# 1. 单进程(慢,~0.6 items/s)
precheck.acc_check(
    api_info="./l1_dump/step0/rank0/dump.json",
    output_path="./acc_check_out",
)

# 2. 多进程多卡(快,~10 items/s,推荐)
precheck.multi_acc_check(
    api_info="./l1_dump/step0/rank0/dump.json",
    output_path="./acc_check_out",
    processes=32,
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
)

# 3. 跨端比对(HCU 输出 csv vs H20 输出 csv)
precheck.api_precision_compare(
    hcu_csv="./hcu_out/accuracy_checking_details.csv",
    gpu_csv="./h20_out/accuracy_checking_details.csv",
    output_path="./precision_diff",
)
```

## CLI

```bash
# 单进程
hcu-probe acc_check -api_info ./dump.json -o ./out

# 多卡并行
hcu-probe multi_acc_check -api_info ./dump.json -o ./out -n 32 -d 0 1 2 3 4 5 6 7

# 跨端比对
hcu-probe api_precision_compare -hcu ./hcu.csv -gpu ./h20.csv -o ./diff
```

## L1 dump 准备

precheck 输入是 **L1 级 statistics 模式**的 dump。配置示例:

```json
{
    "task": "statistics",
    "dump_path": "./l1_dump",
    "step": [0],
    "level": "L1",
    "statistics": { "data_mode": ["all"], "summary_mode": "statistics" }
}
```

L1 wrapping 与 `torch.compile` dynamo 不兼容,跑 L1 dump 时建议 `export TORCHDYNAMO_DISABLE=1`。

## 输出产物

- `accuracy_checking_result_{ts}.csv`:API 级 Pass/Warning/Error 汇总
- `accuracy_checking_details_{ts}.csv`:每个 API 调用的详细指标(余弦 / 双百 / 双千 / 最大绝对误差 / ULP)
- `run_ut.log`:执行日志

## 采样建议

| 维度 | 建议 |
|---|---|
| rank | rank 0 一份足够(DP 完全代表,TP 基本代表;PP 在双边对比中盲区对称) |
| step | step 0 一份足够(静态网络下 API 签名跨 step 不变) |

完整说明:rank 0 + step 0 = 最小充分样本。继续采更多 rank/step 只是重复数据。

## 与其他工具的协同

- `compare`(L0/L1 dump 直接 diff)+ `precheck`(各自对 CPU fp64):前者答"两硬件输出差多少",后者答"哪边相对绝对真值偏差更大"
- `api_precision_compare` 自动 diff 出 "HCU 独有劣化 API" / "NV 独有劣化 API" / "双边共性 bf16 Warning"

## 完整指标参考(上游 PyTorch 文档)

hcu_probe 的 precheck 实现完全沿用上游 mindstudio-probe v26.0.0-alpha.1 的 PyTorch 路径代码。以下细节本文档没展开,需要时查上游 PyTorch 场景文档:

- `accuracy_checking_details_{ts}.csv` 完整 23 列字段说明(Cosine / MaxAbsErr / 双百 / 双千 / 双万 / 误差均衡性 / 均方根误差 / 小值域错误占比 / inf-nan 错误率 / ULP 误差最大值/平均值/大于阈值占比 / ...)
- API 预检判定规则(Pass / Warning / Error 的判定流程图)
- 小值域阈值(torch.float32 / float16 / bfloat16 的小值判定 + 绝对误差阈值)
- 算子看护等级 L0 / L1 / L2(默认 L0;L1 含 embedding / npu_fusion_attention / scatter_;L2 含 matmul / mm / addmm / layer_norm / sum / GELU 等)
- 支持的融合算子列表(npu_apply_adam_w / npu_rms_norm / npu_rotary_mul / npu_fusion_attention 等 15 个)
- `api_precision_compare` 详细字段(标杆比对法 / 二进制一致法 / 绝对阈值法 / ULP 误差比对法各自的判定规则)
- 断点续检(`-csv_path` 参数从中断处续跑)
- `config.yaml` / `config.json` 配置(white_list / black_list / error_data_path / precision 等)

上游文档地址:

```
https://gitcode.com/Ascend/msprobe/blob/26.0.0-alpha.1/docs/zh/accuracy_checker/pytorch_accuracy_checker_instruct.md
```

> ⚠️ **不要参考同目录下 `mindspore_accuracy_checker_instruct.md`** — 那是 NPU + MindSpore 动态图专属。API 范围(Mint API 而不是 torch.* / Tensor.*)、dump.json `framework` 字段(`"mindspore"` 而不是 `"pytorch"`)、CSV 字段集(只 5 列,缺少双百/双千/双万/ULP 误差等)都跟 hcu_probe 走的 PyTorch 路径不一样,看了会误导。
