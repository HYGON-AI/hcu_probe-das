# SGLang 推理精度调试

## 概述

在 SGLang 推理框架中接入 dcu_probe，采集模型运行时的精度数据（dump），然后与 baseline 数据做精度比对，最后生成 TensorBoard 可视化数据定位精度问题。

完整链路：**接入 → dump → `dcu-probe compare` 比对 → `dcu-probe graph_visualize` 构图 → TensorBoard 浏览**

## 两种接入方式

SGLang 在不同版本中，接入 dcu_probe 的方式不同：

| | 方式一：手动接入（侵入式） | 方式二：原生接入 |
|---|---|---|
| **适用版本** | SGLang < 0.5.11（也兼容所有版本） | SGLang >= 0.5.11 |
| **改动方式** | 修改 `ModelRunner` 源码 ~20 行 | 启动参数 `--msprobe-dump-config` |
| **源码修改** | 需要 | **不需要** |
| **PD 分离处理** | 手动判断 `disaggregation_mode` + bootstrap_room 固定 | 只需分别指定 prefill / decode 配置文件 |
| **dcu_probe 兼容** | 完全兼容（代码中直接 `import dcu_probe`） | 需确认 `dcu_probe` 能否替代 SGLang 内置的 `msprobe` 导入 |

> 根据你的 SGLang 版本选择对应章节阅读即可。

---

## 方式一：手动接入（SGLang < 0.5.11）

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

### 2. 修改 ModelRunner 源码

找到 `sglang/srt/model_executor/model_runner.py`，修改 `ModelRunner` 类。

#### （A）__init__ 中添加初始化代码

放在 `# Parse args` 注释之前：

```python
class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        ...
    ):
        ...
        ################################ dcu_probe ################################
        from dcu_probe.pytorch import PrecisionDebugger, seed_all

        seed_all(mode=True)
        self.debugger = PrecisionDebugger(config_path="./config.json")
        ################################ dcu_probe ################################
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        ...
```

#### （B）forward 方法中添加 start / stop / step

```python
def forward(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
) -> ModelRunnerOutput:
    self.forward_pass_id += 1
    ################################ dcu_probe ################################
    if hasattr(self, 'debugger'):
        self.debugger.start(model=self.model)
    ################################ dcu_probe ################################

    ...

    ################################ dcu_probe ################################
    if hasattr(self, 'debugger'):
        self.debugger.stop()
        self.debugger.step()
    ################################ dcu_probe ################################

    return output
```

> 仅采集 L1 级别统计数据时 `start()` 可以不传 `model` 参数；采集 L0 或 mix 级别时**必须**传 `model=self.model`。

#### （C）DP 场景

若 `--dp-size` > 1，需在 `start` 中传入 `rank_id` 以区分不同数据并行的 dump 数据：

```python
if hasattr(self, 'debugger'):
    self.debugger.start(model=self.model, rank_id=self.gpu_id)
```

### 3. 启动 SGLang 推理

**在线模式**：

```bash
#!/bin/bash
export TORCHDYNAMO_DISABLE=1

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 1027 \
    --disable-cuda-graph \
    --skip-server-warmup
```

发送请求后自动开始 dump：

```bash
curl -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Hello, my name is"}],
        "max_tokens": 10
    }' \
    http://127.0.0.1:1027/v1/chat/completions
```

**离线模式**：

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    disable_cuda_graph=True,
)

outputs = llm.generate(
    ["Hello, my name is"],
    sampling_params={"temperature": 0.8, "top_p": 0.95},
)
```

**启动参数注意事项**：

| 参数 | 要求 | 原因 |
|---|---|---|
| `--disable-cuda-graph` | 必须 | dump 依赖 PyTorch eager 模式 |
| `--skip-server-warmup` | 在线模式必须 | 避免采集 warmup 阶段的数据 |
| `TORCHDYNAMO_DISABLE=1` | 遇到报错时设置 | 全局关闭 dynamo |

### 4. PD 分离场景

SGLang 的 Prefill-Decode 分离部署场景需额外处理。

#### （A）配置文件

创建两份配置，`dump_path` **必须不同**以避免写入冲突：

- `config_prefill.json` → `"dump_path": "/home/dcu_dump_prefill"`
- `config_decode.json` → `"dump_path": "/home/dcu_dump_decode"`

> 若只需采集其中一侧，将另一侧的 `config_path` 设为空字符串 `""` 即可。

#### （B）ModelRunner 初始化

根据 `disaggregation_mode` 选择对应配置：

```python
################################ dcu_probe ################################
from dcu_probe.pytorch import PrecisionDebugger, seed_all

seed_all(mode=True)
config_path = ""
disagg_mode = server_args.disaggregation_mode
if disagg_mode == "prefill":
    config_path = "./config_prefill.json"   # 不采 prefill 则改为 ""
elif disagg_mode == "decode":
    config_path = "./config_decode.json"    # 不采 decode 则改为 ""
if config_path:
    self.debugger = PrecisionDebugger(config_path=config_path)
################################ dcu_probe ################################
```

#### （C）固定 bootstrap_room（DP > 1 时必须）

PD 分离 + DP > 1 场景下，SGLang Router 会随机分配请求到不同显卡，导致多次采集不可复现。需在 `sglang/srt/managers/io_struct.py` 中固定 `bootstrap_room` 值：

```python
@dataclass
class GenerateReqInput(BaseReq):
    ...

    def _normalize_bootstrap_params(self, num):
        ...

        # Normalize bootstrap_room
        ################################ dcu_probe ################################
        self.bootstrap_room = 6347036608774465186  # 固定值，保证可复现
        ################################ dcu_probe ################################
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num
        ...
```

#### （D）启动命令

```bash
# 避免 Router 的 /health 请求触发模型 forward
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0

# 启动 Prefill 服务
python3 -m sglang.launch_server \
    --model-path /home/models/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8995 \
    --base-gpu-id 0 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8000 \
    --disable-cuda-graph \
    --skip-server-warmup

# 启动 Decode 服务
python3 -m sglang.launch_server \
    --model-path /home/models/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --base-gpu-id 1 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8001 \
    --disable-cuda-graph \
    --skip-server-warmup

# 启动 Router
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:8000 8995 \
    --decode http://127.0.0.1:8001 \
    --host 127.0.0.1 \
    --port 6688
```

> `--disaggregation-transfer-backend` 根据 DCU 实际支持的 backend 选择。

### 5. dump 产物

推理完成后，数据位于 `dump_path/step0/rank{n}/` 下。详细的目录结构和各文件说明见 [配置与产物参考](./config.md#dump-产物目录结构)。

---

## 方式二：原生接入（SGLang >= 0.5.11）

SGLang 0.5.11+ 已内置 msprobe 支持，通过启动参数 `--msprobe-dump-config` 即可启用，**无需修改源码**。

> **dcu_probe 适配说明**：SGLang 原生 flag 内部调用 `msprobe.pytorch.PrecisionDebugger`。如果 DCU 环境中 `import dcu_probe` 的 API 与 `msprobe` 兼容（`PrecisionDebugger` 和 `seed_all` 接口一致），可直接使用此方式。否则请回退到 [方式一](#方式一手动接入sglang--0511)。

### 1. 创建配置文件

与方式一相同，创建 config.json。详细参数见 [配置与产物参考](./config.md)。

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

### 2. 启动命令

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 1027 \
    --msprobe-dump-config /home/config.json
```

SGLang 自动完成 `PrecisionDebugger` 的初始化和 `start` / `stop` / `step` 的生命周期管理。

> 遇到 dynamo 报错时同样设置 `export TORCHDYNAMO_DISABLE=1`。

### 3. PD 分离场景

方式二下 PD 分离更简单——只需为 prefill 和 decode 分别指定不同的配置文件。

#### （A）创建两份配置

`dump_path` 必须不同：

- `config_prefill.json` → `"dump_path": "/home/dcu_dump_prefill"`
- `config_decode.json` → `"dump_path": "/home/dcu_dump_decode"`

#### （B）启动命令

```bash
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0

# Prefill 服务
python3 -m sglang.launch_server \
    --model-path /home/models/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8995 \
    --base-gpu-id 0 --tp-size 1 \
    --host 127.0.0.1 --port 8000 \
    --msprobe-dump-config /home/config_prefill.json

# Decode 服务
python3 -m sglang.launch_server \
    --model-path /home/models/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --base-gpu-id 1 --tp-size 1 \
    --host 127.0.0.1 --port 8001 \
    --msprobe-dump-config /home/config_decode.json

# Router
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:8000 8995 \
    --decode http://127.0.0.1:8001 \
    --host 127.0.0.1 --port 6688
```

> 方式二下不需要手动修改 `bootstrap_room`——SGLang >= 0.5.11 已内置处理。

---

## 精度比对

dump 完成后，在 DCU 环境和 baseline 环境（另一台 DCU 或 GPU）各得到一份 dump 数据：

```
dcu_dump/step0/rank{0..N}/      ← DCU（目标侧, target）
baseline_dump/step0/rank{0..N}/ ← GPU / 另一 DCU（标杆侧, golden）
```

### 比对命令

```bash
# 单卡比对
dcu-probe compare \
    -tp ./dcu_dump/step0/rank0/dump.json \
    -gp ./baseline_dump/step0/rank0/dump.json \
    -o ./compare_output

# 多卡比对（自动遍历所有 rank 并合并结果）
dcu-probe compare \
    -tp ./dcu_dump/step0 \
    -gp ./baseline_dump/step0 \
    -o ./compare_output

# 输出 xlsx 格式（更便于查看）
dcu-probe compare \
    -tp ./dcu_dump/step0 \
    -gp ./baseline_dump/step0 \
    -o ./compare_output \
    --xlsx
```

产物：`compare_output/compare_result_{timestamp}.csv`（或 xlsx）。

### 比对结果关键字段

| 字段 | 含义 | 参考阈值 |
|---|---|---|
| `Target Name` | DCU 侧算子 / 模块名 | — |
| `Bench Name` | 匹配上的 baseline 算子 / 模块名 | — |
| `CosineSimilarity` | 余弦相似度 | > 0.99 |
| `MaxAbsoluteError` | 最大绝对误差 | — |
| `MaxRelativeError` | 最大相对误差 | — |
| `MeanRelativeError` | 平均相对误差 | < 1.0 |
| `RelativeEuclideanDistance` | 欧氏相对距离 | < 0.05 |
| `KullbackLeiblerDivergence` | KL 散度 | < 0.005 |
| `RootMeanSquareError` | 均方根误差 | < 1.0 |
| `Overflow` | 是否溢出（NO / 上溢 / 下溢） | NO |
| `IsPrecisionError` | 精度是否被标记为异常 | NO |
| `IsOutputNode` | 是否为模型整网输出节点 | — |
| `CompareFailReason` | 比对失败原因（如除零、不匹配等） | — |

### 分析优先级

1. **先看整网输出节点**（`IsOutputNode=YES`）的精度是否达标
2. 如果输出达标，中间节点的精度偏差通常**无需处理**（误差可能在传播中消解）
3. 如果输出不达标，从 `CosineSimilarity` 最低或 `MaxRelativeError` 最大的节点开始向上游追溯

---

## TensorBoard 可视化

### 分级可视化构图

```bash
# 双图比对（单 rank）
dcu-probe graph_visualize \
    -tp ./dcu_dump/step0/rank0 \
    -gp ./baseline_dump/step0/rank0 \
    -o ./vis_output

# 多 rank 批量比对
dcu-probe graph_visualize \
    -tp ./dcu_dump/step0 \
    -gp ./baseline_dump/step0 \
    -o ./vis_output

# 开启溢出检测
dcu-probe graph_visualize \
    -tp ./dcu_dump/step0/rank0 \
    -gp ./baseline_dump/step0/rank0 \
    -o ./vis_output \
    -oc
```

产物：`vis_output/compare_{timestamp}.vis.db`

### 启动 TensorBoard

```bash
tensorboard --logdir ./vis_output --bind_all
```

浏览器打开 `http://<服务器地址>:6006`，点击左上方 **GRAPH_ASCEND** 进入分级可视化页面。

### 趋势可视化（可选）

将多个 step 的 dump 数据导入 SQLite，查看精度数据在 step / rank / 模块三个维度的趋势分布：

```bash
dcu-probe data2db --data ./dcu_dump --db ./trend_output

tensorboard --logdir ./trend_output --bind_all
```

浏览器打开后点击左上方 **TREND ANALYZER** 进入趋势可视化页面。

> 趋势可视化需要安装 tensorboard 插件 `trend_analyzer`，参见 [工具运行环境](../offline_analysis/README.md#浏览侧-tensorboard-插件whl-不含需另装)。

### 浏览器操作速查

| 功能 | 操作 |
|---|---|
| 切换 Step / Rank / MicroStep | 左侧「数据选择」下拉栏 |
| 精度误差筛选 | 左侧「精度误差筛选」→ 勾选误差类型 → 自动展开节点列表 |
| 溢出筛选 | 需 `-oc` 参数开启 → 左侧「溢出筛选」 |
| 节点搜索 | 左侧「节点搜索」→ 输入名称模糊匹配 |
| 颜色含义 | 红色 = error，橙色 = warning，白色 = pass，灰色 = 未匹配 |
| 手动匹配节点 | 左侧「节点匹配」→ 依次点击两侧灰色节点 → 「建立匹配」 |
| 展开 / 折叠节点 | 鼠标左键双击节点 |
| 缩放 / 平移 | 键盘 W / S 缩放，A / D 左右移动，滚轮上下移动 |
