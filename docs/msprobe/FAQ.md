

# 1 数据采集

1. dump.json中API或Module统计信息里出现null或None值的原因是什么？

   dump.json里出现null或None值的可能性较多，常见的场景有：

   - 输入或者输出参数本身是一个None值。
   - 输入参数或输出参数类型当前工具不支持，会有日志打印提醒。
   - 输入或者输出tensor的dtype为bool时，Mean和Norm等字段为null。

2. 如果存在namedtuple类型的数据作为nn.Module的输出，工具会将各字段数据dump下来，但是输出数据类型会被转成tuple，原因是什么？
   - 这是由于pytorch框架自身，在注册module的backward hook时，会将namedtuple类型转成tuple类型。

3. 如果某个api在dump支持列表support_wrap_ops.yaml中，但没有dump该api的数据，原因是什么？
   - 首先确认api调用是否在采集范围内，即需要在 **start** 和 **stop** 接口涵盖的范围内。
   - 其次，由于工具只在被调用时才对api进行patch，从而使得数据可以被dump下来。因此当api是被直接import进行调用时，由于该api的地址已经确定，
   工具无法再对其进行patch，故而该api数据无法被dump下来。如下示例，relu将无法被dump：
   ```python
   import torch
   from torch import relu  # 此时relu地址已经确定，无法修改

   from msprobe.pytorch import PrecisionDebugger

   debugger = PrecisionDebugger(dump_path="./dump_data")
   x = torch.randn(10)
   debugger.start()  # 此时会对torch下面的api进行patch，但已无法对import进来的api进行patch了
   x = relu(x)          
   debugger.stop()
   ```
   在上述场景中，若希望采集relu数据，只需要将`relu(x)`修改为`torch.relu(x)`即可。

4. 在使用L0 dump时，发现有些 module 的数据没有采集下来，原因是什么？
   - 确认日志打印中是否存在`The {module_name} has registered deprecated register_backward_hook`信息，
     该信息说明 module 挂载了被 PyTorch 框架废弃的 register_backward_hook，这与工具使用的 register_full_backward_hook 接口会产生冲突，故工具会跳过该 module 的反向数据采集。
   - 如果您希望所有 module 数据都能采集下来，可以将模型中使用的 register_backward_hook 接口改为 PyTorch 框架推荐的 register_full_backward_pre_hook 或 register_full_backward_hook 接口。

5. 在vllm场景下进行数据dump时，发现报错：`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and npu:0!`
   - 这是因为工具的debugger实例化早于LLM实例化导致的，解决方法就需要将debugger的实例化移至LLM实例化之后进行，可参考下方示例：
   ```python
   from vllm import LLM, SamplingParams
   from msprobe.pytorch import PrecisionDebugger
   prompts = [
      "Hello, my name is",
      "The president of the United States is",
      "The capital of France is",
      "The future of AI is",
   ]

   sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
   llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
   
   debugger = PrecisionDebugger("./config.json")  # debugger实例化晚于LLM实例化
   
   debugger.start()
   outputs = llm.generate(prompts, sampling_params)
   debugger.stop()
   ```

6. 在使用 msprobe 进行 PyTorch 框架的数据采集功能时，请注意确认环境变量 NPU_ASD_ENABLE=0 ，即关闭特征值检测功能。 由于工具冲突， 在该功能开启的情况下可能导致某些 api 数据采集的缺失。

# 2 精度预检(PyTorch)

1. 预检工具在 dump 和 run_ut 的过程中，是否需要同时开启或关闭 jit 编译（jit_compile）？

   答：是。

2. 预检工具对于 type_as 这类涉及数据类型转换操作的 API，是否具有参考性？

   由于这类 API 在 CPU 侧存在精度先提升后下降的操作，因此这类 API 的有效性的参考价值有限。

3. run ut 过程中出现报错：ERROR: Got unsupported ScalarType BFloat16。

   答：请使用最新版本的工具。

4. Dropout 算子，CPU 和 NPU 的随机应该不一样，为什么结果比对是一致的？

   答：这个结果是正常的，工具对该算子有特殊处理，只判定位置为 0 的位置比例大约和设定 p 值相当。

5. 为什么浮点型数据 bench 和 CPU 的 dtype 不一致？

   答：对于 fp16 的数据，CPU 会上升一个精度 fp32 去计算，这是和算子那边对齐的精度结论，CPU 用更高精度去计算会更接近真实值。

6. Tensor 魔法函数具体对应什么操作？

   答：

   | Tensor魔法函数  | 具体操作         |
   | --------------- | ---------------- |
   | `__add__`       | +                |
   | `__and__`       | &                |
   | `__bool__`      | 返回 Tensor 布尔值 |
   | `__div__`       | /                |
   | `__eq__`        | ==               |
   | `__ge__`        | >=               |
   | `__gt__`        | >                |
   | `__iadd__`      | +=               |
   | `__iand__`      | &=               |
   | `__idiv__`      | /=               |
   | `__ifloordiv__` | //=              |
   | `__ilshift__`   | <<=              |
   | `__imod__`      | %=               |
   | `__imul__`      | *=               |
   | `__ior__`       | \|=              |
   | `__irshift__`   | >>=              |
   | `__isub__`      | -=               |
   | `__ixor__`      | ^=               |
   | `__lshift__`    | <<               |
   | `__matmul__`    | 矩阵乘法         |
   | `__mod__`       | %                |
   | `__mul__`       | *                |
   | `__nonzero__`   | 同 `__bool__`     |
   | `__or__`        | \|               |
   | `__radd__`      | +（反向）        |
   | `__rmul__`      | *（反向）        |
   | `__rshift__`    | >>               |
   | `__sub__`       | -                |
   | `__truediv__`   | 同 `__div__`      |
   | `__xor__`       | ^                |

# 3 精度比对(PyTorch)

## 3.1 工具使用

### 3.1.1 dump 指定融合算子

数据采集当前支持融合算子的输入输出，需要在 `mstt/debug/accuracy_tools/msprobe/pytorch/hook_module/support_wrap_ops.yaml` 中添加，比如以下代码段调用的 softmax 融合算子。

```python
def npu_forward_fused_softmax(self, input_, mask):
    resl = torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)
    return resl
```

如果需要 dump 其中调用的 npu_scaled_masked_softmax 算子的输入输出信息，需要在 `support_wrap_ops.yaml` 中的 `torch_npu: ` 中自行添加该融合算子：

```yaml
- npu_scaled_masked_softmax
```

（npu_scaled_masked_softmax 融合算子工具已支持 dump，本例仅供参考）。

## 3.2 常见问题

1. 在同一个目录多次执行 dump 会冲突吗？

    答：会，同一个目录多次 dump，会覆盖上一次结果，可以使用 dump_path 参数修改 dump 目录。

2. 如何 dump 算子级的数据？
   
   答：需要配置 level 为 L2 模式。

3. 工具比对发现 NPU 和标杆数据的 API 无法完全对齐？

    答：torch 版本和硬件差异属于正常情况。

## 3.3 异常情况

1. HCCL 报错： error code: EI0006。

    **故障现象**：使用 msprobe 工具时，报错：error code: EI0006。

    **故障原因**：CANN 软件版本较低导致不兼容。

    **故障处理**：升级新版 CANN 软件版本。

2. torch_npu._C._clear_overflow_npu() RuntimeError NPU error，error code is 107002。

    如果运行溢出检测功能遇到这个报错，采取以下解决方法：

    如果是单卡运行，添加如下代码，0 是卡号，选择自己空闲的卡号。

    ```python
    torch.npu.set_device('npu:0')
    ```

    如果多卡运行，请在代码中修改对应卡号，比如进程使用卡号为 {rank} 时可以添加如下代码：

    ```python
    torch.npu.set_device(f'npu:{rank}')
    ```

    如果运行精度比对功能遇到这个报错，尝试安装最新版本的 msprobe。

3. dump 得到的 `VF_lstm_99_forward_input.1.0.npy`、`VF_lstm_99_forward_input.1.1.npy` 类似的数据是否正常？

    带 1.0/1.1/1.2 后缀的 npy 是正常现象，例如，当输入数据为 [[tensor1, tensor2, tensor3]] 会生成这样的后缀。

4. 进行 compare 报错：The current file contains stack information, please turn on the stack_mode。

    在比对脚本中，设置 `stack_mode=True`，例如：

    ```python
    from msprobe.pytorch import compare
    dump_result_param={
    "npu_json_path": "./npu_dump/dump.json",
    "bench_json_path": "./gpu_dump/dump.json",
    "stack_json_path": "./npu_dump/stack.json",
    "is_print_compare_log": True
    }
    compare(dump_result_param, output_path="./output", stack_mode=True)
    ```

5. dump 指定反向 API 的 kernel 级别的数据报错：NameError：name 'torch_npu' is not defined。

   答：如果是 npu 环境，请安装 torch_npu；如果是 gpu 环境，暂不支持 dump 指定 API 的 kernel 级别的数据。

6. 配置 dump_path 后，使用工具报错：[ERROR] The file path /home/xxx/dump contains special characters。

   答：请检查你设置的 dump 绝对路径是否包含特殊字符，确保路径名只包含大小写字母、数字、下划线、斜杠、点和短横线；注意，如果执行脚本的路径为 /home/abc++/，设置的 dump_path="./dump"，工具实际校验的路径为绝对路径 /home/abc++/dump，++ 为特殊字符，会引发本条报错。

7. 无法 dump matmul 权重的反向梯度数据。

   答：matmul 期望的输入是二维，当输入不是二维时，会将输入通过 view 操作展成二维，再进行 matmul 运算，因此在反向求导时，backward_hook 能拿到的是 UnsafeViewBackward 这步操作里面数据的梯度信息，取不到 MmBackward 这步操作里面数据的梯度信息，即权重的反向梯度数据。典型的例子有，当 linear 的输入不是二维，且无 bias 时，会调用 output = input.matmul(weight.t())，因此拿不到 linear 层的 weight 的反向梯度数据。

8. dump.json 文件中的某些 api 的 dtype 类型为 float16，但是读取此 api 的 npy 文件显示的 dtype 类型为 float32。

    答：msprobe 工具在 dump 数据时需要将原始数据从 npu to cpu 上再转换为 numpy 类型，npu to cpu 的逻辑和 gpu to cpu 是保持一致的，都存在 dtype 可能从 float16 变为 float32 类型的情况，如果出现 dtype 不一致的问题，最终采集数据的 dtype 以 pkl 文件为准。

9. 使用 dataloader 后 raise 异常 Exception("msprobe: exit after iteration {}". format(max(self.config.step)))。

   答：正常现象，dataloader 通过 raise 结束程序，堆栈信息可忽略。

10. 使用 msprobe 工具数据采集功能后，模型出现报错，报错信息为：`activation_func must be F.gelu` 或 `ValueError(Only support fusion of gelu and swiglu)`。

    答：这一类报错常见于 Megatron/MindSpeed/ModelLink 等加速库或模型仓中，原因是工具本身会封装 torch 的 API（API类型和地址会发生改变），而有些 API 在工具使能前类型和地址就已经确定，此时工具无法对这类 API 再进行封装，而加速库中会对某些 API 进行类型检查，即会把工具无法封装的原始的 API和工具封装之后的 API 进行判断，所以会报错。
    规避方式有3种：①将PrecisionDebugger的实例化放在文件的开始位置，即导包后的位置，确保所有API都被封装；②注释 `mstt/debug/accuracy_tools/msprobe/pytorch/hook_module/support_wrap_ops.yaml` 文件中的 `-gelu` 或者 `-silu`，工具会跳过采集该 API。③ 可以考虑根据报错堆栈信息注释引发报错的类型检查。

11. 添加 msprobe 工具后触发与 AsStrided 算子相关、或者编译相关的报错，如：`Failed to compile Op [AsStrided]`。

    答：注释工具目录 `mstt/debug/accuracy_tools/msprobe/pytorch/hook_module/support_wrap_ops.yaml` 文件中 `Tensor: `下的 `-t` 和 `- transpose`。
