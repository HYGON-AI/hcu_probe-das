# config_check — 训练环境配置采集与比对

采集训练环境快照(环境变量 / 三方库版本 / 训练超参 / 权重 / 数据集元信息)到 zip,跨环境比对差异。**排查"代码相同但跑出不同结果"类问题最快的工具**。

## 典型场景

- 同代码 DCU vs H20 跑出不同 loss → 比对两端 zip 找环境差异
- DCU 不同节点间训练结果不一致 → 比对两节点环境
- DTK / driver 升级前后回归 → 比对升级前后环境

## Python API

```python
from dcu_probe.tools import config_check

# 1. 在 DCU 上采集环境快照
config_check.collect(
    shell_path="./run_llama2_7b.sh",
    output_path="./dcu_env.zip",
)

# 2. 在 H20 上采集同样脚本的环境快照(同代码不同硬件)
# config_check.collect(shell_path="./run_llama2_7b.sh", output_path="./h20_env.zip")

# 3. 比对
config_check.compare(
    bench_zip="./h20_env.zip",
    target_zip="./dcu_env.zip",
    output_path="./env_diff",
)
```

## CLI

```bash
# 采集
dcu-probe config_check -d ./run_train.sh -o ./dcu_env.zip

# 比对
dcu-probe config_check -c ./h20_env.zip ./dcu_env.zip -o ./env_diff
```

## DCU 特有环境变量(对精度有影响)

上游默认采集面向 NPU 的环境变量。在 DCU 上还要关注这些(对训练结果有影响):

| 类别 | 关键变量 |
|---|---|
| DCU 硬件控制 | `GPU_MAX_HW_QUEUES` / `CUDA_DEVICE_MAX_CONNECTIONS` / `HSA_FORCE_FINE_GRAIN_PCIE` |
| ROCm / HIP | `HIP_VISIBLE_DEVICES` / `ROCM_PATH` / `PYTORCH_ROCM_ARCH` |
| MIOpen | `MIOPEN_FIND_MODE`(kernel 搜索策略,间接影响选择的 kernel) / `MIOPEN_SYSTEM_DB_PATH` |
| RCCL 通信 | `NCCL_DEBUG`(RCCL 兼容 NCCL 变量名) / `NCCL_SOCKET_IFNAME` / `RCCL_ALLREDUCE_ALGO` |
| TE / FA | `NVTE_DISABLE_FC2_DGRAD_OVERLAP` / `NVTE_NO_PIPELINE_OVERLAP` / `FLASH_ATTENTION_DISABLE` |
| torch.compile | `TORCHDYNAMO_DISABLE` / `TORCH_COMPILE_DEBUG` |
| dcu_megatron 特有 | `GROUPED_GEMM_BatchLinear` |

实际比对时上游 ConfigChecker 会自动采集所有当前 shell 已 set 的环境变量,以上是**人工 review 比对结果时重点关注**的变量子集。

## DCU 上的状态

✅ **2026-05-12 scnet docker 采集 OK**:容器内跑 `dcu-probe config_check -d <path> -o env.zip` 落盘 zip,内含 env / pip / hyperparameters 三段。

⚠️ **DCU 特有环境变量清单尚未沉淀进上游 ConfigChecker**。短期 workaround:跑 `compare()` 后人工 grep 上述变量。长期方案:向上游 PR DCU 环境变量扩展。

## 输出产物

- `*.zip`:包含 `env.json` / `pkg_version.json` / `train_args.json` / `weights_meta.json` / `dataset_meta.json` 等
- `env_diff/diff_report.json`:差异条目列表
