# overflow_check 产物 `anomaly_analyze.json` 分析指南

`hcu-probe overflow_check` 检测到异常时,产 `anomaly_analyze_{timestamp}.json`。本文教你怎么读这份产物。

---

## 是什么

| 状态 | 产物 |
|---|---|
| 健康数据(无 NaN/Inf) | **无产物**(by design,上游 `analyzer.py:49` 直接 return,不创建文件) |
| 检测到异常 | `anomaly_analyze_{ns}.json`,按 rank 分组列出**起点节点** |

**关键 design**:只标"input 正常 + output 异常"的**起点节点**,不报下游受影响的节点。代码层判定见 `graph.py:66-69`:

```python
return (not is_input_anomaly) and is_output_anomaly
```

千卡 NaN 传播场景下,grep dump.json 会得几千行噪音,overflow_check 直接命中 1 行起点。

---

## 顶层结构

```json
{
  "rank_0": [ {...anomaly_1...}, {...anomaly_2...}, ... ],
  "rank_3": [ {...anomaly_a...}, ... ],
  ...
}
```

- 按 rank 分组
- 每个 rank 可能有多个 anomaly 节点(三阶段 `_pre_analyze / _analyze / _post_analyze` 都可能检出)
- 多 rank 间通过通信算子关联(`_connect_comm_nodes`)

---

## 4 个核心字段逐项

每个 anomaly 节点是一个 JSON 对象,4 字段:

### 1. `op_name`(谁错了)

格式:`Module.<模块路径>.<类名>.<forward|backward>.<调用次数>`

```
Module.module.module.embedding.word_embeddings.VocabParallelEmbedding.forward.0
                    ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                    模型嵌套路径 + 类名
```

| 段 | 含义 |
|---|---|
| `Module.` | L0(module 级)hook 前缀;L1 dump 会是 `Functional.` / `Tensor.` / `Torch.` |
| `module.module.` | 多重 wrapping(Float16Module → DDP → 本体)|
| `.forward.0` / `.backward.31` | forward 第 0 次调用 / backward 第 31 次 |

### 2. `data_info`(异常的数值证据)

```json
"data_info": {
    "input_args": [{...Max=正常数, Min=正常数, ...}],
    "input_kwargs": {},
    "output": [{...Max=NaN, ...}]    ← 异常字段
}
```

每个 tensor 含统计量:`type` / `dtype` / `shape` / `Max` / `Min` / `Mean` / `Norm` / `requires_grad`。

**判定规则**(`utils.py:136`):
- `check_item_anomaly()` 扫 input/input_args/input_kwargs/output 的每个 tensor
- 看 `str(item['Max']).lower() in OVERFLOW_LIST`(`nan` / `inf` / `-inf` 等)
- input 全有限 + output 含 NaN/Inf → anomaly 起点

> ⚠️ 只查 **Max / Min** 字段,不查 Mean / Norm。所以如果 Max 是正常的,但 Mean 是 NaN(数学上 Max 也得是 NaN,理论不会发生),不会触发。

### 3. `construct_info`(模块嵌套层次,谁包谁)

从最外层模型到具体异常节点的完整链:

```
[最外]  Module.module.Float16Module.forward.0              ← BF16 训练 wrapper
        Module.module.module.GPTModel.forward.0              ← GPT 模型本体
        Module.module.module.embedding.LanguageModelEmbedding.forward.0   ← 嵌入层(父)
[最内]  Module.module.module.embedding.word_embeddings.VocabParallelEmbedding.forward.0    ← ★ 异常起点
```

**实战价值**:千卡训练 debug 时直接定位"是模型的哪一层",不用猜模型架构。

### 4. `stack_info`(Python 调用栈,~30 帧)

从训练入口到 hook 触发点的完整调用链。逆向重整(从顶到底):

```
pretrain_gpt.py:264               ← 训练入口
    pretrain()
Megatron-LM/.../training.py:737   ← Megatron 主循环
    train(...)
hcu_megatron/.../monitor_wrapper.py:36   ← hcu_probe wrapper
    return original_train(...)
hcu_megatron/.../msprobe_wrapper.py:23   ← hcu_probe wrapper
hcu_megatron/training/training.py:1969   ← 真 train()
... train_step + forward_backward 调度 ...
pretrain_gpt.py:178               ← forward_step()
    output_tensor = model(...)
... GPTModel.forward → _preprocess → embedding ...
language_model_embedding.py:122   ← ★ 异常代码位置(self.word_embeddings(input_ids))
... PyTorch nn.Module call chain ...
torch/nn/modules/module.py:1801   ← PyTorch forward_hook 触发点(hcu_probe 注册的)
    hook_result = hook(self, args, kwargs, result)   ← 这一步检测到 NaN
```

**实战价值**:
1. **找异常代码文件 + 行号**(`language_model_embedding.py:122`)
2. **看训练框架调用路径**(Megatron pipeline → forward_step → model)
3. **看 hcu_probe wrapper 插入点**(`msprobe_wrapper.py:23 / 35`)

---

## 实战阅读流程(从产物到根因)

1. **打开 JSON,逐 rank 看** — 通常多 rank 一起炸是上游 / 数据问题,单 rank 炸是该 rank 硬件 / 通信问题
2. **看第一个 anomaly 节点的 op_name** — 这是异常起点(同 rank 多个时,看 `_pre_analyze` 报的最优先)
3. **读 data_info.input vs output** — 验证 "input 正常,output 异常" 是否真成立(确认 anomaly 不是上游传过来)
4. **读 construct_info** — 知道在模型架构哪一层(GPT 的哪个子模块)
5. **读 stack_info 找代码行号** — 跳到对应代码看实现
6. **根据异常类型推根因** — 见下方表格

---

## 翻译人话模板

```
rank <X> 的训练里,通过 <op_name 末段> 后 output 出现 <NaN/Inf>。
input 是 <input.dtype / shape / Max>(正常),output 是 <output.dtype / shape / Max=NaN>(异常)。
→ 异常起点:不是上游传的,从这个 module 自己开始。

模型架构层次:<construct_info 从最外到最内,4-6 层>
代码位置:<stack_info 里 hcu_megatron / project 代码的某帧 file:line>

可能根因(按异常 module 类型):
  - <根据下方表格选>
```

---

## 按 module 类型的常见根因

| 异常 module | 可能根因 | 排查方向 |
|---|---|---|
| `Embedding` / `VocabParallelEmbedding` | 权重含 NaN / token id 越界 / pad_id 配置错 | 看 input_ids Max 是否超 vocab_size、weight L2 norm 是否爆炸 |
| `Linear` / `RowParallelLinear` / `ColumnParallelLinear` / GEMM | weight 累积 NaN / grad 爆炸 / 混合精度 cast 出问题 | 看 grad norm 历史曲线、optimizer 状态(`m`/`v`)、检查上一步 backward 是否已警告 |
| `FlashAttention` / `SelfAttention` | attention scores overflow(softmax 前 logits 太大)/ mask 不对 | 看 attention mask shape、temperature 设置、检查序列长度突变 |
| `LayerNorm` / `RMSNorm` | input 全 0 或 inf 导致除零 / weight 含 NaN | 看上游 module 的 output Max/Min 范围 |
| `Dropout` | 罕见,可能 mask 生成器问题 | 看 dropout 概率配置、torch.compile 状态 |
| 通信算子(`all_reduce` / `reduce_scatter` 等) | 跨卡数据传输错误 / 上一步某 rank 已 NaN 传染 | 对比 dst rank 跟 src rank 的 input,看 NaN 来自哪个 src |

---

## 跟其他 hcu_probe 工具协同

| 工具 | 协同方式 |
|---|---|
| `graph_visualize -oc` | 把 anomaly 标到模型图上,浏览器直观看在哪层 |
| `compare` | 拿同样 step 的"正常 dump"(比如前一个 ckpt 跑的)跟当前 dump 比,找哪个 step 开始崩 |
| `precheck` (`acc_check`) | 针对 anomaly module 类型(如 `Linear` / `RMSNorm`)单独跑 API 精度预检,验算子本身在 HCU 上是否精度可靠 |
| `data2db` | 把多 step monitor csv 导入 trend.db,在 TENSORBOARD 上看 weight / grad norm 长期趋势,验证 NaN 起源 step |

---

## 已知局限

1. **只检测 Max / Min 字段**(不查 Mean / Norm,理论上不会漏但极端 case 可能)
2. **只标第一个起点**(`for ... break`),同 rank 同 step 多起点只报第一个;rerun 时换 step 数据可能换报另一个
3. **跨 rank 关联依赖通信算子识别**(`is_communication_op`),如果有自定义 comm op 没识别可能漏报传播路径
4. **L0 / L1 dump 兼容**,**L2 kernel dump 上游 NPU 专属**(HCU 没必要管)
5. **真实场景需 task=statistics dump**(JSON 元数据),task=tensor dump 也能用但产物含真实 tensor 数据时无意义(NaN 在 .pt 里,工具检 dump.json 字段)

---

## 实测样例(快速复现)

容器内跑:

```bash
cd /workspace/hcu_megatron_clean/examples/hcu_probe/whl/tools_validation
bash 10_overflow_nan_demo.sh
```

预期产物:`/workspace/data/tools_practice/overflow_with_nan_in_output/anomaly_analyze_{ts}.json`(~6 KB)。本文档的所有字段示例都来自该实测。

---

## 上游文档参考

```
https://gitcode.com/Ascend/msprobe/blob/26.0.0-alpha.1/docs/zh/overflow_check/overflow_check_instruct.md
```

> ⚠️ 不要参考同目录下 `mindspore_overflow_check_instruct.md`(MindSpore 专属,通信算子分类规则与 PyTorch 路径不同)。
