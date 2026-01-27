# 趋势可视化

## 简介

趋势可视化功能将msProbe工具采集的精度数据进行解析，识别其中模型层的张量目标，以及其在迭代步数、节点rank和网络模型中的位置。将张量目标的统计量数据从迭代步数step、节点rank和张量目标三个维度进行趋势可视化，方便用户从数据整体的趋势分布观测精度数据，分析精度问题。

**基本概念**

- msProbe：全称MindStudio Probe，是精度调试工具包，可以定位模型训练或推理中的精度问题。
- dump：MindStudio Probe下数据采集功能，采集的数据称为dump数据。
- monitor：MindStudio Probe下训练状态监测功能，采集的数据称为monitor数据。

**使用流程**

1. 进行工具安装以及数据的采集，详见[使用前准备](#使用前准备)。
2. 使用命令行工具解析精度数据，生成db格式的SQLite数据库文件，详见[精度数据解析](#精度数据解析)。
3. 启动TensorBoard服务，将`--logdir`参数设置为精度数据解析功能的输出路径。
4. 用谷歌浏览器打开TensorBoard服务页面，在`MONVIS`插件窗口下查看数据。


## 使用前准备

**环境准备**

- 安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。
- 安装TensorBoard。
  ```shell
  pip install tensorboard
  ```

- 安装tb_trend_plugin插件。下载离线whl包（[下载链接](../resources/tb_trend_plugin-0.1.0-py3-none-any.whl)），然后通过pip安装（此处{version}为 whl 包实际版本）。
  ```shell
  pip install tb_trend_plugin-{version}-py3-none-any.whl
  ```

**数据准备**

- dump数据场景（采集模型数据，选择`level`为`L0`或者`mix`）
  - PyTorch框架详细采集方式请参见《[PyTorch场景精度数据采集](../dump/pytorch_data_dump_instruct.md)》。
  - MindSpore框架详细采集方式请参见《[MindSpore场景精度数据采集](../dump/mindspore_data_dump_instruct.md)》。
- monitor数据场景（输出格式`format`指定为`csv`）
  - 详细采集方式请参见《[Monitor训练状态轻量化监测工具](../monitor_instruct.md)》。
  

**约束**

- 支持PyTorch框架和MindSpore框架。

## 精度数据解析

### dump数据解析

**功能说明**

解析dump数据，识别其中各模型层下的张量目标，根据dump数据落盘顺序确定张量目标在迭代步数、节点rank和网络模型中的位置，并将解析结果转存为db格式的SQLite数据库文件。

**注意事项**

仅支持dump配置的`level`为`L0`或者`mix`级别。

**命令格式**

```shell
msprobe data2db --data <data_path> --db <db_path> [--micro_step <micro_step>] [--mapping <mapping_json>]
```

**参数说明**

| 参数             | 可选/必选 | 说明                                                         |
| ---------------- | --------- | ------------------------------------------------------------ |
| --data           | 必选      | dump数据路径，str类型。                                      |
| --db             | 必选      | 解析结果文件存盘目录，str类型。落盘文件名为：`monitor_metrics.db`。 |
| --mapping        | 可选      | 指定json映射文件路径（须配置到json文件名，例如./mapping.json），str类型。程序会在解析dump数据时，将其中模型层的名称或算子名称根据映射文件做转换，以达到精简名称或step间模型或算子名称对齐的效果。映射文件详细配置和介绍请参见[mapping配置文件说明](#mapping配置文件说明)。 |
| --micro_step     | 可选      | 配置微迭代（micro_step）步数，表示将一个迭代（step）数据拆分为多个微迭代。int类型，范围为[1, 10000]，默认未配置，表示不拆分迭代。微迭代是根据dump数据中模型层名或算子名的执行索引进行数据拆分的。该拆分方式用于在可视化时通过微迭代的维度进行趋势分析。 |

**使用示例**

解析`/data/dump_path`下的dump数据文件，将解析所得db格式的SQLite数据库文件放在`/data/db_path`路径下。

```shell
msprobe data2db --data /data/dump_path --db /data/db_path
```


**输出说明**

dump数据解析命令执行成功后，在`/data/db_path`下生成`monitor_metrics.db`文件


### monitor数据解析


**功能说明**

解析monitor数据，识别csv格式文件中各张量目标，根据数据落盘顺序确定张量目标在迭代步数、节点rank和网络模型中的位置，并将解析结果转存为db格式的SQLite数据库文件。

**注意事项**

仅支持输出格式`format`指定为`csv`的monitor数据。

**使用示例**

1. 创建Python脚本，以`csv2db.py`命名为例，将以下配置拷贝到文件中，并按实际情况修改。
   ```python
    from msprobe.core.monitor.csv2db import CSV2DBConfig, csv2db
    config = CSV2DBConfig(
        monitor_path="~/monitor_output",
        time_start="Dec03_21-34-40",
        time_end="Dec03_21-34-42",
        process_num=8,
        data_type_list=["grad_unreduced", "grad_reduced"],
        output_dirpath="~/monitor_output"
    )
    csv2db(config)
    ```
    参数详细介绍请参见[csv2db](#csv2db)接口。
    
2. 执行如下命令开启转换。

    ```shell
    python csv2db.py
    ```

**输出说明**

`csv2db`接口调用成功后，在配置的输出路径`output_dirpath`中，生成一个`monitor_metrics.db`文件，为db格式的SQLite数据库文件。


## Megatron模型并行可视化

**功能说明**

Megatron框架中的模型并行会将模型切分在不同节点rank上。分析精度数据时，各个节点下采集的模型层数据可能仅包含整体模型的部分层，无法直观看出这些层处于整体模型中的位置。Megatron模型并行可视化功能提供多节点模型并行切分的可视化能力，帮助用户快速识别当前模型并行配置下，模型层在各个设备上的分布情况。

**注意事项**

仅支持Megatron场景下，张量并行、流水线并行、虚拟流水线并行和数据并行模式。
仅支持节点rank数小于等于1024且模型层小于等于256层场景，即需`world_size ≤ 1024` 且 `num_layers ≤ 256`。

**使用示例**

1. 创建Python脚本，以创建的命名为`plot_model.py`的脚本为例，将以下代码拷贝到`plot_model.py`脚本中，并按实际情况修改`ParallelConfig`中配置。
    ```python
    from msprobe.core.common.megatron_utils import ParallelConfig, plot_model_parallelism

    config = ParallelConfig(
        world_size=32,
        num_layers=48,
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        num_layers_per_virtual_pipeline_stage=3,
        order="tp-cp-ep-dp-pp",
        standalone_embedding_stage=False,
        output_path='./'
    )
    plot_model_parallelism(config)
    ```
    参数详细介绍请参见[plot_model_parallelism](#plot_model_parallelism)接口。
    
2. 执行如下命令开启转换。

    ```shell
    python plot_model.py
    ```
  
**输出说明**

`plot_model`接口调用成功后，在配置的输出路径`output_path`中，生成一个`png`文件, 格式为`ws{world_size}_ln{num_layers}_tp{tensor_parallel_size}_pp{pipeline_parallel_size}_vpp{virtual_pipeline_parallel_size}.png`。其中`virtual_pipeline_parallel_size`为根据`num_layers_per_virtual_pipeline_stage`等传入参数计算出来的虚拟流水线并行分组大小。

浏览png文件，如下图所示：

![ws32_ln48_tp4_pp4_vpp4.png](../figures/trend_analyzer/ws32_ln48_tp4_pp4_vpp4.png)

图片内容介绍：
| 字段         | 说明 | 
| -------------- | --------- |
| Model Parallelism Configuration | 用户设置或计算得来的并行配置信息，包括：</br> Total Layers：模型的总层数，即脚本中的`num_layers`； </br> DP：数据并行分组大小，通过传入并行参数计算得来；</br> TP：张量并行分组大小，即脚本中的`tensor_parallel_size`； </br> PP：流水线并行分组大小，即脚本中的`pipeline_parallel_size`；</br> VPP：虚拟流水线并行分组大小，即文件名中`virtual_pipeline_parallel_size`，通过传入并行参数计算得来。    |                 
| TP Group | 纵坐标，张量并行分组，形如`Group{num}: Rank{start}-{end}`，其中`num`为分组编号，`start`和`end`分别表示分组内第一个rank的编号和最后一个rank的编号。例如，`Group0: Rank0-3`表示第0个分组，其中包含rank0到rank3共4个rank。     |                 
| Virtual Pipeline Stage | 横坐标，流水线并行阶段或虚拟流水线并行阶段，形如`Stage {num}`，其中`num`表示阶段编号。      |                 
| Model Copies | 模型副本图例。数据并行中，以不同颜色标记输入数据不同的模型副本。      |                 
| `Embed`/ `L{start}-{end}`/ `Out`  | 图中颜色矩阵上的文字，标识一个张量并行分组下的一个阶段包含哪些模型层，其中：</br>  `Embed`：表示当前阶段为模型第一个阶段，通常包含嵌入层；</br> `L{start}-{end}`：表示当前阶段包含模型从start至end的模型层，例如`L1-3`说明当前阶段包含整个模型的第1、第2和第3模型层；</br> `Out`: 表示当前阶段为模型最后一个阶段，通常包含输出层。</br> 如果同时满足多个阶段定义，以"+"连接。    |                 

## 附录

### mapping配置文件说明

mapping配置文件是为[dump数据解析](#dump数据解析)功能的--mapping参数提供数据输入。
配置--mapping参数后，dump数据解析在处理每个dump数据中的模型层的名称或算子名称时，会依次按mapping.json中配置的键、值进行替换。该功能适用于需要精简名称或step间目标名称不一致需对齐的场景。

json文件格式和示例如下，键值均为字符串。

```json
{
  ".TE": ".",
  ".MindSeed": "."
}
```

以上格式中，左侧字段为“键”（如".TE"），右侧字段为“值”（如"."），以上配置代表将名称".TE"替换为"."，将名称".MindSeed"替换为"."。

### 公开接口

#### csv2db

**功能说明**

csv2db接口是为[monitor数据解析](#monitor数据解析)功能提供的接口。用以解析monitor数据，识别csv格式文件中各张量目标，根据数据落盘顺序确定张量目标在迭代步数、节点rank和网络模型中的位置，并将解析结果转存为db格式的SQLite数据库文件。

**函数原型**

```python
csv2db(config: CSV2DBConfig) -> None
```

**参数说明**

配置参数实例（CSV2DBConfig类实例），在实例初始化时传入参数。

| 参数名         | 输入/输出 | 说明                                                         |
| -------------- | --------- | ------------------------------------------------------------ |
| monitor_path   | 输入      | 必选参数，待转换的csv存盘目录，str类型。                      |
| time_start     | 输入      | 可选参数，起始时间，str类型，例如"Dec03_21-34-40"。搭配time_end一起使用，从而指定一个时间范围（闭区间），会对这个范围内的文件进行转换。默认为None不限制。 |
| time_end       | 输入      | 可选参数，结束时间，str类型，例如"Dec03_21-34-41"。搭配time_start一起使用，从而指定一个时间范围（闭区间），会对这个范围内的文件进行转换。默认为None不限制。 |
| process_num    | 输入      | 可选参数，配置启动的进程个数，int类型，默认为1，更多的进程个数可以加速转换。 |
| data_type_list | 输入      | 可选参数，指定需要转换的数据类型，数据类型应来自输出文件前缀，数据类型包括：<br/> ["actv", "actv_grad", "exp_avg", "exp_avg_sq", "grad_unreduced", "grad_reduced", "param_origin", "param_updated", "other"]。<br/>默认未配置本参数，表示转换全部数据类型。list\[str\]类型。 |
| output_dirpath | 输入      | 可选参数，指定转换后的输出路径，str类型，默认输出到"{curtime}_csv2db"文件夹，其中curtime为自动获取的当前时间戳。 |

**返回值说明**

无


#### plot_model_parallelism

**函数原型**

```python
plot_model_parallelism(config: ParallelConfig) -> None
```

**参数说明**

配置参数实例（ParallelConfig类实例），在实例初始化时传入参数。

| 参数名                                | 输入/输出 | 说明                                                                                                                                                                    |
| ------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| world_size                            | 输入      | 必选参数，模型部署的总rank数，int类型，支持范围为[1, 1024]。                                                                                                                                 |
| num_layers                            | 输入      | 必选参数，模型的总层数，int类型，支持范围为[1, 256]。                                                                                                                                       |
| tensor_parallel_size                  | 输入      | 可选参数，张量并行分组大小，int类型，默认值为1。实际训练脚本中指定`--tensor-model-parallel-size T`，其中`T`为指定的张量并行分组大小。                                                                                                                            |
| pipeline_parallel_size                | 输入      | 可选参数，流水线并行分组大小，int类型，默认值为1。实际训练脚本中指定`--pipeline-model-parallel-size P`，其中`P`为指定的流水线并行分组大小。                                                                                                                          |
| num_layers_per_virtual_pipeline_stage | 输入      | 可选参数，每个虚拟流水线阶段包含的层数，int类型，默认值为None，表示未开启虚拟流水线并行。实际训练脚本中指定`--num-layers-per-virtual-pipeline-stage V`，其中`V`为指定的每个虚拟流水线阶段的层数。 |
| order                                 | 输入      | 可选参数，模型并行维度的排序顺序，str类型。默认为Megatron默认设置，即`tp-cp-ep-dp-pp`。                                                                                                             |
| standalone_embedding_stage            | 输入      | 可选参数，是否开启将嵌入层作为独立的流水线阶段的配置，bool类型，配置True表示开启，False表示关闭，默认值为False。                                                                                                   |
| output_path                           | 输入      | 可选参数，可视化结果输出路径，str类型，默认值为'./'。                                                                                                                   |

**返回值说明**

无