# vLLM 推理精度调试

## 概述

在 vLLM 推理框架中接入 dcu_probe 的 `PrecisionDebugger`，采集模型运行时的精度数据（dump），然后与 baseline 数据做精度比对，最后生成 TensorBoard 可视化数据定位精度问题。

完整链路：**接入 → dump → `dcu-probe compare` 比对 → `dcu-probe graph_visualize` 构图 → TensorBoard 浏览**

## 适用场景

| vLLM 版本 | 模式 | 是否支持 |
|---|---|---|
| 社区原生 vLLM | PyTorch Eager 模式 | ✅ 本文档 |
| 社区原生 vLLM | PyTorch 图模式 | ❌ 不支持 |

> 社区版 vLLM 在 DCU 上使用时，确保已安装对应 DCU 的 PyTorch 构建版本。

## 接入流程

### 1. 创建配置文件

在启动目录下创建 config.json：

```json
{
    "task": "statistics",
    "dump_path": "/home/dcu_dump",
    "rank": [],
    "step": [],
    "level": "mix",
    "async_dump": false,
    "statistics": {
        "scope": [],
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

> 完整参数说明见 [配置与产物参考](./config.md)。

### 2. 修改 GPUModelRunner 源码

找到 vLLM 的 `vllm/v1/worker/gpu_model_runner.py`，修改 `GPUModelRunner` 类。

#### （A）__init__ 中添加初始化代码

```python
class GPUModelRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        ################################ dcu_probe ################################
        from dcu_probe.pytorch import PrecisionDebugger, seed_all

        seed_all(mode=True)
        self.debugger = PrecisionDebugger(config_path="./config.json")
        ################################ dcu_probe ################################
        self.vllm_config = vllm_config
        self.device = device
        ...
```

#### （B）execute_model 中添加 start / stop / step

vLLM 的 `execute_model` 通常较长且有多处 return，推荐使用 `try...finally` 统一处理结束逻辑：

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, torch.Tensor]:
    ################################ dcu_probe ################################
    if hasattr(self, "debugger"):
        self.debugger.start(model=self.model)
    ################################ dcu_probe ################################

    try:
        ...
        return output
    finally:
        ################################ dcu_probe ################################
        if hasattr(self, "debugger"):
            self.debugger.stop()
            self.debugger.step()
        ################################ dcu_probe ################################
```

> 如果不方便使用 `try...finally` 整体包裹，也可在各个 `return` 前分别添加 `stop()` 和 `step()` 调用。

**注意事项**：

| 要点 | 说明 |
|---|---|
| `start(model=self.model)` | 采集 L0 或 mix 级别时**必须**传入 model；仅采集 L1 级别可省略 |
| `try...finally` | 确保即使推理异常退出也会落盘完整数据，**强烈推荐** |
| `@torch.inference_mode()` | vLLM 已自带，无需额外处理 |

### 3. 启动 vLLM 服务

```bash
#!/bin/bash
export TORCHDYNAMO_DISABLE=1

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --dtype float16 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000
```

> **必须指定 `--enforce-eager`**：社区版 vLLM 不支持图模式下的 dump。

### 4. 发送推理请求触发 dump

服务启动后，发送请求即可自动触发 dump：

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Explain gravity in one sentence.",
        "max_tokens": 32,
        "temperature": 0
    }'
```

每次请求对应一个新的 step（step0、step1 …）。如果只发了 1 条请求，dump 数据位于 `step0/` 下。

### 5. dump 产物

推理完成后，数据位于 `dump_path/step0/rank{n}/` 下。详细的目录结构和各文件说明见 [配置与产物参考](./config.md#dump-产物目录结构)。

---

## 精度比对

dump 完成后在 DCU 和 baseline 环境各得到一份数据。比对命令和结果解读与 SGLang 完全一致，详见 [SGLang 精度比对](./sglang.md#精度比对)。

快速示例：

```bash
# 多卡比对
dcu-probe compare \
    -tp ./dcu_dump/step0 \
    -gp ./baseline_dump/step0 \
    -o ./compare_output \
    --xlsx
```

产物：`compare_output/compare_result_{timestamp}.xlsx`

---

## TensorBoard 可视化

可视化命令与 SGLang 完全一致，详见 [SGLang TensorBoard 可视化](./sglang.md#tensorboard-可视化)。

快速示例：

```bash
# 生成可视化数据
dcu-probe graph_visualize \
    -tp ./dcu_dump/step0/rank0 \
    -gp ./baseline_dump/step0/rank0 \
    -o ./vis_output

# 启动 TensorBoard
tensorboard --logdir ./vis_output --bind_all
```

浏览器打开 `http://<服务器地址>:6006`，点击左上方 **GRAPH_ASCEND** 进入分级可视化页面。

---

## 常见问题

### dump 后 loss / 输出发生变化

工具中的 hook 机制可能引入额外的 `item()` 调用触发同步，导致计算图产生微小变化。这是 PyTorch hook 机制的正常副作用。

### dump 数据中缺少部分算子的反向数据

PyTorch 框架的 BackwardHook 对原地操作（如 `nn.ReLU(inplace=True)`）有限制，可能导致该模块及其上一个模块的反向数据缺失。这是 PyTorch 框架本身的限制，排查时关注前向数据即可。

### 遇到 dynamo 相关报错

设置环境变量 `export TORCHDYNAMO_DISABLE=1` 全局关闭 dynamo。

### 需要在多卡上分别 dump

工具默认采集所有卡的数据（`"rank": []`）。如需仅采集特定卡，在 config.json 中配置 `"rank": [0, 1]` 或 `"rank": ["4-6"]`。
