# MindSpore 动态图场景的离线精度预检

## 1 简介

**MindSpore 动态图精度预检**<sup>a</sup>通过扫描昇腾 NPU 上用户训练 MindSpore 模型中的所有 Mint API 以及 Msadapter场景下迁移的 Mindspore API，输出精度情况的诊断和分析。工具以模型中所有 API 前反向的 dump 结果为输入，构造相应的 API 单元测试，将 NPU 输出与标杆（CPU 高精度）比对，计算对应的精度指标，从而找出 NPU 中存在精度问题的 API。本工具支持**随机生成模式和真实数据模式**<sup>b</sup>。

a. 支持 Mindspore 版本：2.4/2.5；

b. （可选）当使用Msadapter时，由于需要环境中同时存在 Torch 与 Msadapter，所以只支持在**安装原生Torch**的场景下通过export PYTHONPATH="xx/msadapter/build/lib"等通过**环境变量使能Msadapter的方式**的环境中进行预检,预检工具能够自动索引得到所需的 Torch 与 Msadapter环境,环境安装详细参考:[msadapter官网](https://gitee.com/mindspore/msadapter)（该网站需要申请权限方可访问）。

c. 在预检时可以由工具构造随机数据或者获取真实dump数据进行预检操作。随机生成模式执行效率高，可以快速获得结果，但结果准确度低，只能大致判断精度问题；真实数据模式执行效率略低于随机生成模式，并且需要较大磁盘空间存放待预检数据，但是结果准确度高，可以准确判断精度问题。

## 2 离线预检流程

操作流程如下：

1. 在 NPU 环境下安装 msprobe，详见[msprobe 安装](../msprobe_install_guide.md)。
2. 在 NPU 训练脚本内添加 msprobe 工具 dump 接口 PrecisionDebugger，采集待预检数据。详见《[MindSpore场景精度数据采集](../dump/mindspore_data_dump_instruct.md)》，注意需要配置 level="L1"。
3. 执行预检操作，查看预检结果文件，分析预检不达标的 API。

## 3 离线预检操作指导

### 3.1 使用 acc_check 执行预检

将 API 信息输入到 acc_check 模块进行精度检测并比对，运行如下命令：


```bash
msprobe acc_check -api_info ./dump.json -o ./checker_result
```

| 参数名称     | 说明                                                                                                                                                                             | 参数类型 | 是否必选     |
| ---------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------| ---------------------------------- |
| -api_info 或 --api_info_file   | 指定 API 信息文件 dump.json。对其中的mint api以及部分Tensor api进行预检，预检支持的Tensor api列表详见[预检支持列表](../../../python/msprobe/mindspore/api_accuracy_checker/checker_support_api.yaml)。 | str  | 是      |
| -o 或 --out_path    | 指定预检结果存盘路径，默认“./”。                                                                                                                                                             | str  | 否      |
| -csv_path 或 --result_csv_path | 指定本次运行中断时生成的 `accuracy_checking_result_{timestamp}.csv` 文件路径，执行 acc_check 中断时，若想从中断处继续执行，配置此参数即可。需要指定为上次中断的 `accuracy_checking_result_{timestamp}.csv` 文件。详见[3.3 断点续检](#33-断点续检)。 | str  | 否      |
| -save_error_data                    | 保存(随机数据模式)精度未达标的 API 输入输出数据。                                                                                                                                                   | 空    | 否      |

预检执行结果包括 `accuracy_checking_result_{timestamp}.csv` 和 `accuracy_checking_details_{timestamp}.csv` 两个文件。`accuracy_checking_result_{timestamp}.csv` 属于 API 级，标明每个 API 是否通过测试。建议用户先查看 `accuracy_checking_result_{timestamp}.csv` 文件，对于其中没有通过测试的或者特定感兴趣的 API，根据其 API Name 字段在 `accuracy_checking_details_{timestamp}.csv` 中查询其各个输出的达标情况以及比较指标。详细介绍请参见[4 预检结果](#4-预检结果)。

随机数据模式下，如果需要保存比对不达标的输入和输出数据，可以在 acc_check 执行命令结尾添加 `-save_error_data`，例如：

```bash
msprobe acc_check -api_info ./dump.json -o ./checker_result -save_error_data
```

数据默认会存盘到 '{out_path}/error_data' 路径下。

### 3.2 使用 multi_acc_check 执行多线程预检

multi_acc_check 脚本，可以并行在多个Device执行 acc_check 操作，从而减少预检耗时。示例如下：

```bash
msprobe multi_acc_check -api_info ./dump.json -d 0 1 2 3
```

| 参数名称                          | 说明                                                                                                                                                                              | 参数类型      | 是否必选     |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------| ---------------------------------- |
| -api_info 或 --api_info_file   | 指定 API 信息文件 dump.json。对其中的mint api以及部分Tensor api进行预检，预检支持的Tensor api列表详见[预检支持列表](../mindspore/api_accuracy_checker/checker_support_api.yaml)。 | str       | 是      |
| -o 或 --out_path               | 指定预检结果存盘路径，默认“./”。                                                                                                                                                              | str       | 否      |
| -csv_path 或 --result_csv_path | 指定本次运行中断时生成的 `accuracy_checking_result_{timestamp}.csv` 文件路径，执行 acc_check 中断时，若想从中断处继续执行，配置此参数即可。需要指定为上次中断的 `accuracy_checking_result_{timestamp}.csv` 文件。详见[3.3 断点续检](#33-断点续检)。 | str       | 否      |
| -d 或 --device                 | 指定 Device ID，选择 UT 代码运行所在的卡，默认值为 0，支持同时指定 0 ~ Device数量 - 1 ，例如 0 1 2 3 4。                                                                                                       | List[int] | 否      |
| -save_error_data              | 保存(随机数据模式)精度未达标的 API 输入输出数据。                                                                                                                                                    | 空         | 否      |

在不同卡数下，使用38B语言大模型的预检耗时基线参考[multi_acc_check耗时基线](../baseline/mindspore_accuracy_checker_perf_baseline.md)。


数据默认会存盘到 './ut_error_data{timestamp}' 路径下。

### 3.3 断点续检

断点续检操作通过如下命令执行：

```bash
msprobe acc_check -api_info ./dump.json -csv_path xxx/accuracy_checking_result_{timestamp}.csv
```

精度预检 acc_check 过程中，若因环境、数据量过大等原因导致预检进程中断，那么当用户解决这些问题后，重新执行 acc_check 操作，可以通过断点续检操作继续前面未完成的预检，会在 -csv_path 指定的 `accuracy_checking_result_{timestamp}.csv` 文件以及对应的 `accuracy_checking_details_{timestamp}.csv` 文件中继续写入后续的结果，不会重新创建结果文件。

须指定为上次预检中断的 `accuracy_checking_result_{timestamp}.csv` 文件。请勿修改 `accuracy_checking_result_{timestamp}.csv` 和 `accuracy_checking_details_{timestamp}.csv` 文件以及文件名，否则不对断点续检的结果负责。


## 4 预检结果

精度预检生成的 `accuracy_checking_result_{timestamp}.csv` 和 `accuracy_checking_details_{timestamp}.csv` 文件内容详情如下：

`accuracy_checking_details_{timestamp}.csv`

| 字段      | 含义       |
| ------------------- | ------------------------------------------------------------ |
| API Name            | API 名称。                                        |
| Bench Dtype         | 标杆数据的 API 数据类型。      |
| Tested Dtype        | 被检验数据的 API 数据类型。                                  |
| Shape               | API 的 Shape 信息。       |
| Cosine              | 被检验数据与标杆数据的余弦相似度。                         |
| MaxAbsErr           | 被检验数据与标杆数据的最大绝对误差。                       |
| MaxRelativeErr      | 被检验数据与标杆数据的最大相对误差。                     |
| Status              | API 预检通过状态，pass 表示通过测试，error 表示未通过。 |
| Message             | 提示信息。            |

注意：PyTorch 无法对 dtype 为整数类型的 tensor 进行反向求导，而 MindSpore 支持。反向过程的预检仅比较 dtype 为浮点型的输出。

`accuracy_checking_result_{timestamp}.csv`

| 字段                  | 含义      |
| --------------------- | ----------------- |
| API Name              | API 名称。         |
| Forward Test Success  | 前向 API 是否通过测试，pass 为通过，error 为错误。 |
| Backward Test Success | 反向 API 是否通过测试，pass 为通过，error 为错误，如果是空白的话代表该 API 没有反向输出。 |
| Message               | 提示信息。         |

Forward Test Success 和 Backward Test Success 是否通过测试是由 `accuracy_checking_details_{timestamp}.csv` 中的余弦相似度、最大绝对误差判定结果决定的。具体规则详见 [4.1 API 预检指标](#41-api-预检指标)。
需要注意的是 `accuracy_checking_details_{timestamp}.csv` 中可能存在一个 API 的前向（反向）有多个输出，那么每个输出记录一行，而在 `accuracy_checking_result_{timestamp}.csv` 中的结果需要该 API 的所有结果均为 pass 才能标记为 pass，只要存在一个 error 则标记 error。

### 4.1 API 预检指标

   - API 预检指标是通过对 `accuracy_checking_details_{timestamp}.csv` 中的余弦相似度、最大绝对误差的数值进行判断，得出该 API 是否符合精度标准的参考指标。
   - 余弦相似度大于 0.99，并且最大绝对误差小于 0.0001，标记“pass”，否则标记为“error”。
