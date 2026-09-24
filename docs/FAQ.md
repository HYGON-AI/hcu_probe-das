# HCUProbe FAQ

## 目录

1. [HCUProbe 是否可以 Dump 出来训练网络的结构？](#1-HCUProbe-是否可以-dump-出来训练网络的结构)
2. [nn.Module 在进行计算时，注册的钩子函数调用逻辑和流程是怎样的？](#2-nnmodule-在进行计算时注册的钩子函数调用逻辑和流程是怎样的)
3. [Pytorch 里面的算子运算的数据如何进行 dump？](#3-pytorch-里面的算子运算的数据如何进行-dump)
4. [如何从 dump 出来的 construct.json 和 dump.json 文件中，构造 tensorboard 能够识别的数据？](#4-如何从-dump-出来的-constructjson-和-dumpjson-文件中构造-tensorboard-能够识别的数据)
5. [按模型或者API方式dump的每一层数据的唯一标识如何生成？](#5-按模型或者api方式dump的每一层数据的唯一标识如何生成)
6. [为什么模型权重的梯度是 NaN?](#6-为什么模型权重的梯度是-nan)

---

## 1. HCUProbe 是否可以 Dump 出来训练网络的结构？

**答案是肯定的**，但是前提是我们在 `config.json` 里面是按照 `Module` 方式进行数据的 Dump，这样在 `dump_path/step/rank/construct.json` 文件中会保存整个模型的结构。

### 可选阅读部分：以 ResNet50 为例说明工具如何保存网络模型结构

我们使用以下代码可以输出 PyTorch 框架里面 ResNet50 的整个模型结构：

```python
import torch
import torchvision.models as models

# ResNet-50
model = models.resnet50(pretrained=True)

# 修改最后一层
num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 打印 ResNet50 模型结构
print(model)
```

上面的代码会输出以下模型结构信息：

```txt
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(...)
    (2): Bottleneck(...)
  )
  (layer2): Sequential(...)
  (layer3): Sequential(...)
  (layer4): Sequential(...)
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=10, bias=True)
)
```

PyTorch 里面的网络模型结构类似于一种**树状结构**，ResNet50 模型本身里面还有多个其他子模型，子模型里面还有属于该子模型的子模型，以此类推。也就是模型结构是一种多叉树，一个父节点可能有多个子节点，子节点又可能又有多个子节点，以此类推。

模型的计算过程却是一个**线性过程**，从 ResNet50 的第一个子节点的子节点的子节点的...，也就是递归最深处的那个子节点开始模型的计算；然后是該子节点的兄弟节点；当该子节点的所有兄弟节点都运行完了，就会运行该子节点的父节点的所有其他兄弟节点，以此类推，直至 ResNet50 里面的所有子节点都运行完成。整个过程就是深度优先遍历整个树结构的过程，每当遍历到一个树节点，就做计算。要注意的是，所有的计算均在叶子节点完成，这里的叶子节点也就是 PyTorch 里面我们常用的 Linear、Dropout、ReLU、BatchNorm、LayerNorm、Conv2d 等。而构成父节点的一般就是自定义 nn.Module、nn.Sequential、nn.Bottleneck、nn.ModuleList、nn.ModuleDict、nn.ParameterList、nn.ParameterDict 等。

有了前面的介绍，想要理清 ResNet50 的模型结构就很简单了，下面就用树状的列表来罗列出所有的节点结构：

```yaml
# ResNet50 树状结构
ResNet:
  - conv1: Conv2d
  - bn1: BatchNorm2d
  - relu: ReLU
  - maxpool: MaxPool2d
  # Sequential
  - layer1:
    # Bottleneck
    - 0:
      - conv1: Conv2d
      - bn1: BatchNorm2d
      - conv2: Conv2d
      - bn2: BatchNorm2d
      - conv3: Conv2d
      - bn3: BatchNorm2d
      - relu: ReLU
      # Sequential
      - downsample:
        - 0: Conv2d
        - 1: BatchNorm2d
    # Bottleneck
    - 1:
      - conv1: Conv2d
      - bn1: BatchNorm2d
      - conv2: Conv2d
      - bn2: BatchNorm2d
      - conv3: Conv2d
      - bn3: BatchNorm2d
      - relu: ReLU
    # Bottleneck
    - 2:
      - conv1: Conv2d
      - bn1: BatchNorm2d
      - conv2: Conv2d
      - bn2: BatchNorm2d
      - conv3: Conv2d
      - bn3: BatchNorm2d
      - relu: ReLU
  # Sequential
  - layer2:
    # Bottleneck
    - 0:
      - conv1: Conv2d
      - bn1: BatchNorm2d
      - conv2: Conv2d
      - bn2: BatchNorm2d
      - conv3: Conv2d
      - bn3: BatchNorm2d
      - relu: ReLU
      # Sequential
      - downsample:
        - 0: Conv2d
        - 1: BatchNorm2d
    - 1: ...
    - 2: ...
    - 3: ...
  # Sequential
  - layer3:
    - 0: ... (with downsample)
    - 1: ...
    - 2: ...
    - 3: ...
    - 4: ...
    - 5: ...
  # Sequential
  - layer4:
    - 0: ... (with downsample)
    - 1: ...
    - 2: ...
  - avgpool: AdaptiveAvgPool2d
  - fc: Linear
```

### 工具 Dump 出来的模型结构信息 (construct.json)

接下来，分析一下工具在模型运行过程中 dump 出来的模型结构信息，该结构信息位于 `construct.json` 文件中。以下信息是通过工具 Dump 出来的 ResNet50 的模型结构信息，使用父节点和子节点的形式表述：

```json
{
 "Module.conv1.Conv2d.forward.0": null,
 "Module.bn1.BatchNorm2d.forward.0": null,
 "Module.relu.ReLU.forward.0": null,
 "Module.maxpool.MaxPool2d.forward.0": null,
 "Module.layer1.Sequential.forward.0": null,
 "Module.layer1.0.Bottleneck.forward.0": "Module.layer1.Sequential.forward.0",
 "Module.layer1.0.conv1.Conv2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.bn1.BatchNorm2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.relu.ReLU.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.conv2.Conv2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.bn2.BatchNorm2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.relu.ReLU.forward.1": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.conv3.Conv2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.bn3.BatchNorm2d.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.downsample.Sequential.forward.0": "Module.layer1.0.Bottleneck.forward.0",
 "Module.layer1.0.downsample.0.Conv2d.forward.0": "Module.layer1.0.downsample.Sequential.forward.0",
 "Module.layer1.0.downsample.1.BatchNorm2d.forward.0": "Module.layer1.0.downsample.Sequential.forward.0",
 "Module.layer1.0.relu.ReLU.forward.2": "Module.layer1.0.Bottleneck.forward.0",
 ...
}
```

### construct.json 字段说明

| 字段 | 含义 |
|------|------|
| `Module` | 代表这是一个模型 |
| `null` | 代表该节点没有任何父节点（根节点） |
| `forward` | 前向计算 |
| `forward.0` | 代表第一次前向计算 |
| `backward` | 反向计算 |
| `backward.0` | 代表第一次反向计算 |
| `Module.conv1.Conv2d` | 代表该模型一个名为 `conv1` 的子模型，且该模型为 `Conv2d` |
| `Module.layer1` | 代表该模型的一个名为 `layer1` 的子模型 |
| `Module.layer1.0` | 代表该模型的一个名为 `layer1` 的子模型的，一个名为 `0` 的子模型 |
| `Module.layer1.0.conv1` | 代表该模型的一个名为 `layer1` 的子模型里面名为 `0` 的子模型里面的名为 `conv1` 的子模型 |

### 从 construct.json 还原模型结构

使用上面工具 Dump 出来的模型结构，尝试还原整个 ResNet50 的模型结构，对比看看和 PyTorch 输出的模型结构是否一致：

```yaml
Resnet50:
  # 前向
  - Module.conv1.Conv2d.forward.0
  - Module.bn1.BatchNorm2d.forward.0
  - Module.relu.ReLU.forward.0
  - Module.maxpool.MaxPool2d.forward.0
  - Module.layer1.Sequential.forward.0:
    - Module.layer1.0.Bottleneck.forward.0:
      - Module.layer1.0.conv1.Conv2d.forward.0
      - Module.layer1.0.bn1.BatchNorm2d.forward.0
      - Module.layer1.0.relu.ReLU.forward.0
      - Module.layer1.0.conv2.Conv2d.forward.0
      - Module.layer1.0.bn2.BatchNorm2d.forward.0
      - Module.layer1.0.relu.ReLU.forward.1
      - Module.layer1.0.conv3.Conv2d.forward.0
      - Module.layer1.0.bn3.BatchNorm2d.forward.0
      - Module.layer1.0.downsample.Sequential.forward.0:
        - Module.layer1.0.downsample.0.Conv2d.forward.0
        - Module.layer1.0.downsample.1.BatchNorm2d.forward.0
      - Module.layer1.0.relu.ReLU.forward.2
    - Module.layer1.1.Bottleneck.forward.0: ...
    - Module.layer1.2.Bottleneck.forward.0: ...
  - Module.layer2.Sequential.forward.0: ...
  - Module.layer3.Sequential.forward.0: ...
  - Module.layer4.Sequential.forward.0: ...
  - Module.avgpool.AdaptiveAvgPool2d.forward.0
  - Module.fc.Linear.forward.0

  # 反向
  - Module.fc.Linear.backward.0
  - Module.avgpool.AdaptiveAvgPool2d.backward.0
  - Module.layer4.Sequential.backward.0:
    - Module.layer4.2.Bottleneck.backward.0:
      - Module.layer4.2.relu.ReLU.backward.2:
        - Module.layer4.2.conv3.Conv2d.backward.0
        - Module.layer4.2.relu.ReLU.backward.1:
          - Module.layer4.2.conv2.Conv2d.backward.0
          - Module.layer4.2.relu.ReLU.backward.0:
            - Module.layer4.2.conv1.Conv2d.backward.0
```

### 结论

上面对比了完整的前向过程和部分的反向过程。发现：

- 在前向计算过程中，在每一个 Bottleneck 里面都会多了两个 ReLU 的计算，这是和 PyTorch 输出的 ResNet50 结构不一致的地方。
- 在反向传播计算过程中，其传播过程则完全与前向传播的计算相对应。

通过阅读 PyTorch 里面的源码发现，在一个 Bottleneck 里面的模型结构，和上面 `print` 命令输出的 ResNet50 的结构不太一样，工具 Dump 出来的结构是和框架里面的结构是一样的。所以，**结论肯定是可以重构出模型的完整结构**。我们可以通过 Dump 出来的模型父子节点间的关系，重构出完整的模型结构，进而在 TensorBoard 里面进行可视化展示。同时，将每一个 Layer dump 出来的数据统计信息，也可以在 TensorBoard 里面进行可视化展示出来。

---

## 2. nn.Module 在进行计算时，注册的钩子函数调用逻辑和流程是怎样的？

总的来讲，一个 `nn.Module` 在进行计算时，其调用的钩子函数的逻辑如下：

1. **前向 pre-hooks 调用**：该钩子函数有两类：全局和局部的，先调用全局、再调用局部钩子函数。该钩子函数会在模型计算前调用；
2. **模型的 `forward()` 计算**；
3. **前向 hooks 调用**：该钩子函数有两类：全局和局部的，先调用全局、再调用局部钩子函数。该钩子函数会在模型计算后调用；
4. **反向 pre-hooks 调用**：该钩子函数有两类：全局和局部的，先调用全局、再调用局部钩子函数；
5. **反向 hooks 调用**：该钩子函数有两类：全局和局部的，先调用全局、再调用局部钩子函数；

### nn.Module 详细调用逻辑

我们在进行一个 `nn.Module` 的训练或者推理时，一般采用的调用方式如下：

```python
net = MyNet()
output = net(input)
```

这里的 `net(input)` 会去调用 `__call__` 接口，如果我们自己定义的模型有重写 `__call__` 方法，则会调用该方法。但最终会调用到 `nn.Module.__call__` 里面去，`__call__` 会调用 `_wrapped_call_impl` 接口，根据是否走 `torch.compile` 和正常的调用逻辑，分别调用 `_compiled_call_impl` 或 `_call_impl`。

### `_call_impl` 源码分析

```python
def _call_impl(self, *args, **kwargs):
    forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
    # If we don't have any hooks, we want to skip the rest of the logic in
    # this function, and just call forward.
    if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or
            self._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)
    try:
        result = None
        called_always_called_hooks = set()
        full_backward_hooks, non_full_backward_hooks = [], []
        backward_pre_hooks = []
        if self._backward_pre_hooks or _global_backward_pre_hooks:
            backward_pre_hooks = self._get_backward_pre_hooks()
        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook_id, hook in (
                *_global_forward_pre_hooks.items(),
                *self._forward_pre_hooks.items(),
            ):
                if hook_id in self._forward_pre_hooks_with_kwargs:
                    args_kwargs_result = hook(self, args, kwargs)
                    if args_kwargs_result is not None:
                        if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                            args, kwargs = args_kwargs_result
                        else:
                            raise RuntimeError(
                                "forward pre-hook must return None or a tuple "
                                f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                            )
                else:
                    args_result = hook(self, args)
                    if args_result is not None:
                        if not isinstance(args_result, tuple):
                            args_result = (args_result,)
                        args = args_result
        bw_hook = None
        if full_backward_hooks or backward_pre_hooks:
            bw_hook = hooks.BackwardHook(self, full_backward_hooks, backward_pre_hooks)
            args = bw_hook.setup_input_hook(args)
        result = forward_call(*args, **kwargs)
        if _global_forward_hooks or self._forward_hooks:
            for hook_id, hook in (
                *_global_forward_hooks.items(),
                *self._forward_hooks.items(),
            ):
                if hook_id in self._forward_hooks_always_called or \
                   hook_id in _global_forward_hooks_always_called:
                    called_always_called_hooks.add(hook_id)
                if hook_id in self._forward_hooks_with_kwargs:
                    hook_result = hook(self, args, kwargs, result)
                else:
                    hook_result = hook(self, args, result)
                if hook_result is not None:
                    result = hook_result
        if bw_hook:
            if not isinstance(result, (torch.Tensor, tuple)):
                warnings.warn("For backward hooks to be called,"
                              " module output should be a Tensor or a tuple of Tensors"
                              f" but received {type(result)}")
            result = bw_hook.setup_output_hook(result)
        # Handle the non-full backward hooks
        if non_full_backward_hooks:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next(v for v in var.values() if isinstance(v, torch.Tensor))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    grad_fn.register_hook(_WrappedHook(hook, self))
                self._maybe_warn_non_full_backward_hook(args, result, grad_fn)
        return result
    except Exception:
        # run always called hooks if they have not already been run
        for hook_id, hook in _global_forward_hooks.items():
            if hook_id in _global_forward_hooks_always_called and \
               hook_id not in called_always_called_hooks:
                try:
                    hook_result = hook(self, args, result)
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn("global module forward hook with ``always_call=True`` "
                                  "raised an exception that was silenced as another error "
                                  f"was raised in forward: {str(e)}")
                    continue
        for hook_id, hook in self._forward_hooks.items():
            if hook_id in self._forward_hooks_always_called and \
               hook_id not in called_always_called_hooks:
                try:
                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn("module forward hook with ``always_call=True`` "
                                  "raised an exception that was silenced as another error "
                                  f"was raised in forward: {str(e)}")
                    continue
        raise
```

### 源码逐行解读

- **第 2 行** — `torch.jit.trace` 的调用过程
- **第 5-8 行** — 如果没有任何钩子函数被注册的话，就直接调用 `forward()` 接口
- **第 10-88 行** — 调用钩子函数和 `forward()`
  - **第 11 行** — `result` 用于保存经过钩子函数和 `forward()` 接口的输出值
  - **第 12 行** — `called_always_called_hooks` 用于记录哪些钩子函数需要始终被调用
  - **第 14 行** — `full_backward_hooks` 和 `non_full_backward_hooks` 反向钩子函数列表，这两个列表要么全为空，要么一个不为空、另一个为空，不可能存在两个都不为空的情况
  - **第 16-17 行** — 获取反向的 pre-hooks 列表，其中全局反向 pre-hooks 在前，Module 的反向 pre-hooks 在后
  - **第 19-20 行** — 获取反向的 hook 列表，其中全局反向 hook 在前，Module 的反向 hook 在后
  - **第 22-42 行** — 如果存在全局前向 pre-hooks 和 Module 前向 pre-hooks 的话，需要对模型的输入调用钩子函数：
    - **第 23-42 行** — 依次遍历全局前向 pre-hooks 和 Module 前向 pre-hooks。在 nn.Module 中，一个钩子函数会保存在一个字典里面，字典的关键字类型为 `Int`，而键值为钩子函数
    - **第 27-36 行** — 如果当前钩子函数可以接受关键字参数：
      - **第 28 行** — 则按照关键字参数调用该前向 pre-hook
      - **第 29-36 行** — 如果该钩子函数有返回值的话，那么返回值类型必须是 `tuple` 类型，且元组长度只能是 2（一个为位置参数列表 `args`，一个为关键字参数 `kwargs`）；否则报错
    - **第 38-42 行** — 不按照关键字参数传递：
      - **第 38 行** — 调用钩子函数，并获取返回值
      - **第 39-42 行** — 如果存在返回值的话，将返回值以 tuple 的形式保存在 `__call__` 的位置参数列表 args 里面
  - **第 44 行** — `bw_hook` 用于保存 `BackwardHook` 对象
  - **第 45-47 行** — 如果存在反向钩子（这里只有反向的 full-hooks 和反向的 pre-hooks），那么需要根据输入设置输入并保存到 `args` 中
  - **第 49 行** — `forward()` 接口的调用
  - **第 50-65 行** — 如果有前向钩子函数，则按照先调用全局前向钩子，再调用 Module 前向钩子
  - **第 67-72 行** — 如果存在反向钩子函数，需要设置反向输出的钩子
  - **第 75-86 行** — 这是处理反向的 non-full 钩子函数，则是老版本的实现，后续会弃用掉
  - **第 88 行** — 返回前向计算的结果
- **第 90-119 行** — 这是在遇到任何异常时，那些被标记为始终要运行的钩子函数，需要在异常处理过程中被调用，以保证 always call 的逻辑

---

## 3. Pytorch 里面的算子运算的数据如何进行 dump？

例如我们常用的 `torch.matmul` ？

前面介绍如何对一个 `nn.Module` 模块的前向和反向的输入、输出以及梯度数据进行 Dump，对于一个 PyTorch 里面的算子数据的话，也可以建立在模块数据 Dump 的基础之上。利用 `nn.Module` 和 `Tensor` 上面的钩子函数，实现对算子接口的前向和反向过程的输入、输出数据进行 Dump。有了这个前提，要实现对算子的前向和反向数据的 Dump，就相对容易。

### PyTorch 里面的算子分类

在 PyTorch 里面的算子，主要涉及以下几类：

| 模块 | 说明 | 常用算子 |
|------|------|---------|
| `torch.nn.functional` | PyTorch 最为基础的算子模块 | `conv1d`, `conv2d`, `relu`, `scaled_dot_product_attention` |
| `torch.Tensor` | PyTorch 张量类里面的算子接口 | `Tensor.abs`, `Tensor.add`, `Tensor.bmm`, `Tensor.matmul` |
| `torch` | 位于 `torch` 模块里面的一些算子 | `torch.linalg.vecdot`, `torch.linalg.qr`, `torch.addbmm`, `torch.addmm` |
| `torch._VF` | 工具里面只 Dump `torch._VF.lstm` 算子 | `torch._VF.lstm` |
| `torch.distributed` | 分布式相关的算子 | `torch.distributed.send`, `torch.distributed.reduce`, `torch.distributed.all_gather` |

### 实现方式

我们只需要对上面这五类模块里面的算子，使用一个 `nn.Module` 对其调用做一层封装。在 `nn.Module` 进行前向计算时，走该算子的调用即可（这里，我们只需要对 `nn.Module` 的 `__call__` 接口进行重载即可）；在进行反向传播过程中，则会根据计算图的逆向过程计算该算子的反向。

要实现该过程，可以使用下面的逻辑：

```python
def wrap_api_func(api):
    def api_function(*args, **kwargs):
        HookModule(api)(*args, **kwargs)
    api_function.__name__ = api.__name__
    return api_function
```

在上面的代码中，`HookModule` 就是用来对算子 `api` 进行封装的 `nn.Module` 模块，该模块的 `__call__` 方法会去调用 `api` 算子。`wrap_api_func` 会对算子进行一个 `HookModule` 的封装，使其成为一个新的算子（新的函数对象）`api_function`。我们要把五个模块中的每一个算子（Python 层面来讲就是函数对象的地址）修改为我们封装的函数对象 `api_function`。那么在我们训练模型的时候，只要调用了相应的算子，我们就可以调用到我们新设置的函数对象 `api_function` 这里，进而使用 `HookModule` 进行调用。而在 `HookModule` 里面，我们注册了前向和反向的 pre-hooks、hooks，而在这些钩子函数里面，我们就可以拿到相应的输入、输出数据了。

---

## 4. 如何从 dump 出来的 construct.json 和 dump.json 文件中，构造 tensorboard 能够识别的数据？

### 可视化步骤

首先，说明一下用户想要可视化 dump 出来的数据需要以下两个步骤：

1. 需要从 dump 出来的 `stack.json`、`construct.json` 和 `dump.json` 三个文件中生成一个中间文件，该中间文件的数据格式也是 JSON 格式；
2. 使用 `tensorboard --logdir` 命令指定生成的中间文件目录，这样用户就可以可视化整个 dump 出来的数据信息；

### Dump 文件说明

| 文件 | 作用 |
|------|------|
| `stack.json` | 模型调用栈、或者算子调用栈，这里的调用栈是完整的 Python 调用栈 |
| `construct.json` | 用于模型数据 dump（算子数据 dump 内容为空）。里面记录了模型的父子节点之间的关系，用于重构整个网络的模型（前向和反向） |
| `dump.json` | 记录模型和算子的输入、输出、梯度张量的数据类型、shape、最大值、最小值、均值、L2 Norm；还可以 dump 模型里面的权重张量的上述信息 |

### TensorBoard 可视化内容

TensorBoard 能够把用户 dump 出来的每一个 step 里面的每一个 rank 里面数据在界面里可视化出来，可视化的内容有以下：

- 不同 step 里面的不同 rank 数据的选择
- **单个模型**的完整结构：模型父子节点之间的对应关系，包括前反向（`construct.json` 里面的结构信息）
- **对比模型**的完整结构：模型父子节点之间的对应关系，包括前反向（`construct.json` 里面的结构信息）
- 模型节点的选择，展示该模型节点 dump 出来的数据（`dump.json` 里面的各项数据）；展示该模型的调用栈（`stack.json` 里面的数据）

### 中间 JSON 文件格式

每一个 step 里面的每一个 rank 里面的各个 dump 出来的数据格式均是一样的，我们只需要对一套 `stack.json`、`construct.json` 和 `dump.json` 进行解析即可，最终保存在一个 JSON 格式的字典格式文件里面。

#### 全局信息

| 字段 | 说明 |
|------|------|
| `root` | 根节点名称，用户自定义名称，用来作为整个模型的根 |
| `dump_data_dir` | dump 数据的目录 |
| `node` | 模型的所有节点数据，root 则为子节点的祖宗节点，即根节点 |
| `MicroSteps` | 微步信息 |
| `OverflowCheck` | 是否要进行浮点值溢出检测 |

#### 节点信息 (node)

每一个节点有以下完全相同的信息：

| 字段 | 说明 |
|------|------|
| `id` | 节点标识符 |
| `node_type` | 节点的类型，一般都是 Model，其值为 0 |
| `output_data` | 该模型节点的输出数据的信息 |
| `input_data` | 该模型节点的输入数据的信息；包括模型参数张量的数据信息 |
| `upnode` | 父节点的 ID |
| `subnodes` | 所有子节点的 ID 列表 |
| `matched_node_link` | 只在模型比对时使用，标杆模型 Bench 所对应的节点列表 |
| `suggestions` | 建议信息 |
| `stack_info` | 当前节点的调用栈信息 |
| `data` | 数据信息 |

#### 节点输入/输出数据信息

| 字段 | 说明 |
|------|------|
| `type` | 张量数据类型，一般为 `torch.Tensor` |
| `dtype` | 张量元素类型，例如：`torch.float16` |
| `shape` | 张量的形状 |
| `Max` | 最大值 |
| `Min` | 最小值 |
| `Mean` | 平均值 |
| `Norm` | L2 Norm |
| `data_name` | 可能是张量数据的文件保存目录 |

通过以上的这些信息，TensorBoard 便可以可视化所有 dump 出来的数据。

### 双模型比对可视化

上面介绍了单个模型的数据可视化，在可视化界面里面，还可以支持**两个模型的比对可视化**。要进行双模型的比对，需要使用 `msprobe -f pytorch graph` 命令生成 TensorBoard 能够解析的中间 JSON 文件。该中间 JSON 文件的数据内容大致如下：

#### 全局信息（根字典数据信息）

| 字段 | 说明 |
|------|------|
| `HCU` | 记录目标模型的 dump 数据，例如：我们要把 BMZ 和 A800 比较，这里的 HCU 数据就是从 BMZ 上面 dump 出来的数据 |
| `Bench` | 标杆模型的 dump 数据，例如：我们要把 BMZ 和 A800 比较，这里的 Bench 数据就是从 A800 上面 dump 出来的数据 |
| `ToolTip` | 字符串的提示 |
| `Colors` | 不同精度误差用不同的颜色加以区分 |
| `MicroSteps` | 微步信息 |
| `task` | 精度数据的任务 |
| `summary` | 统计模式 |
| `md5` | MD5 数据摘要模式 |
| `all` | 真实 tensor 数据模式 |
| `structure` | 存结构比较模式 |
| `OverflowCheck` | 是否浮点溢出检测 |

#### HCU 和 Bench 比对信息

两个比对模型里面保存的数据信息一致：

| 字段 | 说明 |
|------|------|
| `root` | 根节点名称，用户自定义，一般为 `DefaultModel` |
| `dump_data_dir` | HCU 和 Bench dump 数据的目录 |
| `node` | 保存所有节点的信息 |

#### node 节点信息

与单模型类似，每个节点包含：`id`、`node_type`、`output_data`、`input_data`、`upnode`、`subnodes`、`matched_node_link`、`suggestions`、`stack_info`、`micro_step_id`、`data`。

#### input_data 和 output_data 差异字段

以下信息只出现在 HCU 节点里面，Bench 里面没有（因为只需要 HCU 与 Bench 做比对获得精度比对结果即可）：

| 字段 | 说明 |
|------|------|
| `Max diff` | 最大值与标杆数据的差值 |
| `Min diff` | 最小值与标杆数据的差值 |
| `Mean diff` | 平均值与标杆数据的差值 |
| `Norm diff` | L2Norm 值与标杆数据的差值 |
| `MaxRelativeErr` | 最大值与标杆数据的相对误差 |
| `MinRelativeErr` | 最小值与标杆数据的相对误差 |
| `MeanRelativeErr` | 平均值与标杆数据的相对误差 |
| `NormRelativeErr` | L2Norm 值与标杆数据的相对误差 |

> **注意**：目前该工具还把模型参数梯度作为一个节点进行展示，里面信息和节点信息一样，有点儿与模型结构相悖，后续可能会修复。

---

## 5. 按模型或者API方式dump的每一层数据的唯一标识如何生成？

前面介绍过，如果是使用 Module 方式 Dump 数据时，会抓取出整个模型的结构信息，进而可以还原出整个模型的结构。那么该模型结构每一层的唯一标识符是如何生成的呢？这有助于帮助用户清晰地知道，我们当前处于模型的那一层结构，或者哪一个接口上面，进而来观察该层的输入、输出、梯度等信息是否正常。

在这里，我分别针对 Module 和 API 两类标识符，进行分别的介绍。

### 5.1 模型层标识符

对于一个继承自 `nn.Module` 的模型，在工具的 `ModuleProcessor::get_modules_and_names` 方法里面，会获取我们要 dump 模型及其子模型的所有信息（这里调用 `nn.Module::named_modules` 获取），针对当前模型及其子模型，注册能够标识该模型的标识符。

**标识符构成规则：**

- 所有模型 dump 的标识符，均以 `Module` 字符串起始，用 `.` 进行切分
- 每一个模型都会有一个名称（通过 `named_modules()` 获取），紧跟在上面字符串后面；如果我们要 dump 一组模型的话（例如：`list[nn.Module]` / `tuple(nn.Module)`），那么在该模型名称前面加一个索引值，以标识当前在 dump 哪一个模型
- 再加上该模型的类名，例如：`nn.Linear` 的类名则为 `Linear`
- 接下来是计算的标识：
  - `forward` — 标识为前向
  - `backward` — 标识为反向
  - `parameters_grad` — 标识为参数梯度
- 最后，使用一个索引值来标识当前模型层计算的次数

### 5.2 API 标识符

API 类的数据 dump，主要有以下几类：

| 类别 | 标识 |
|------|------|
| `functional` | `Functional` |
| `torch` | `Torch` |
| `tensor` | `Tensor` |
| `aten` | `Aten` |
| `_VF` | `VF` |
| `distributed` | `Distributed` |

**API 标识构成规则：**

- API 所属类别的标识，和 Module 一样，使用 `.` 进行切分
- 该 API 的名称
- 该 API 被调用的次数索引值

---

## 6. 为什么模型权重的梯度是 NaN?

这个问题的出处来自 Megatron-LM 训练 Llama2 过程中出现的。

**场景**：例如 PP=2, TP=1, GPUS=8, DP=4，如果 GBS=12，MBS=1，则 GA=3。那么，在一个设备上面就会运行三次前向和反向。

### 原因

如果在训练模型时打开了**梯度累加融合优化**的操作（这个一般是训练框架自己实现，不同框架有不同的实现），那么在每一次前反向过程中，模型权重的梯度就会 Dump 不到。工具会去权重的 `.grad` 属性里面拿梯度数据，但因为梯度累加融合操作改变了该逻辑（例如：Megatron-LM 里面会把梯度放到一个 `.main_grad` 属性里面），进而产生 NaN。

### 解决方案

在使用工具 dump 数据的时候，需要把训练框架的梯度累加融合优化**关闭**。例如 Megatron-LM 里面就使用：

```bash
--no-gradient-accumulation-fusion
```

进行关闭。
