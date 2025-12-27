# 趋势可视化

## 简介

趋势可视化功能将msProbe工具采集的精度数据进行解析，识别其中模型层的张量目标，以及其在迭代步数、节点rank和网络模型中的位置。将张量目标的统计量数据从迭代步数step、节点rank和张量目标三个维度进行趋势可视化，方便用户从数据整体的趋势分布观测精度数据，分析精度问题。

**注意**：

* 前端可视化目前仅支持monitor数据场景。

**基本概念**

- msProbe：全称MindStudio Probe，是精度调试工具包，可以定位模型训练或推理中的精度问题。
- dump：MindStudio Probe下数据采集功能，采集的数据称为dump数据。
- monitor：MindStudio Probe下训练状态监测功能，采集的数据称为monitor数据。

**使用流程**

1. 进行工具安装以及数据的采集，详见[使用前准备](#使用前准备)。
2. 使用命令行工具解析精度数据，生成db格式的SQLite数据库文件，详见[精度数据解析](#精度数据解析)。
3. 启动TensorBoard服务，将`--logdir`参数设置为精度数据解析功能的输出路径（目前仅支持monitor数据场景）。
4. 用谷歌浏览器打开TensorBoard服务页面，在`MON_VIS`插件窗口下查看数据。


## 使用前准备

**环境准备**

- 安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。
- 安装TensorBoard。
  ```shell
  pip install tensorboard
  ```

- 安装tb_graph_ascend插件。下载离线whl包（[下载链接](../resources/tb_graph_ascend-3.0.0-py3-none-any.whl)），然后通过pip安装（此处{version}为 whl 包实际版本）。
  ```shell
  pip install tb_graph_ascend-{version}-py3-none-any.whl
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
msprobe data2db --data <data_path> --db <db_path> [--micro_step <micro_step>] [--mapping <mapping_json>] [--step_partition <step_partition>]
```

**参数说明**

| 参数             | 可选/必选 | 说明                                                         |
| ---------------- | --------- | ------------------------------------------------------------ |
| --data           | 必选      | dump数据路径，str类型。                                      |
| --db             | 必选      | 解析结果文件存盘目录，str类型。落盘文件名为：`monitor_metrics.db`。 |
| --mapping        | 可选      | 指定json映射文件路径（须配置到json文件名，例如./mapping.json），str类型。程序会在解析dump数据时，将其中模型层的名称或算子名称根据映射文件做转换，以达到精简名称或step间模型或算子名称对齐的效果。映射文件详细配置和介绍请参见[mapping配置文件说明](#mapping配置文件说明)。 |
| --micro_step     | 可选      | 配置微迭代（micro_step）步数，表示将一个迭代（step）数据拆分为多个微迭代。int类型，范围为[1, 10000]，默认未配置，表示不拆分迭代。微迭代是根据dump数据中模型层名或算子名的执行索引进行数据拆分的。该拆分方式用于在可视化时通过微迭代的维度进行趋势分析。 |
| --step_partition | 可选      | 指定的数据库分区大小，int类型，范围为[10, 10000000]，落盘数据库会按分区大小将数据表拆分为多个step表。默认为50， 每50个step一个表。 |


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
        step_partition=500,
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
| monitor_path   | 输入      | 必选参数，待转换的csv存盘目录，str类                         |
| time_start     | 输入      | 可选参数，起始时间，str类型，例如"Dec03_21-34-40"。搭配time_end一起使用，从而指定一个时间范围（闭区间），会对这个范围内的文件进行转换。默认为None不限制。 |
| time_end       | 输入      | 可选参数，结束时间，str类型，例如"Dec03_21-34-41"。搭配time_start一起使用，从而指定一个时间范围（闭区间），会对这个范围内的文件进行转换。默认为None不限制。 |
| process_num    | 输入      | 可选参数，配置启动的进程个数，int类型，默认为1，更多的进程个数可以加速转换。 |
| data_type_list | 输入      | 可选参数，指定需要转换的数据类型，数据类型应来自输出文件前缀，数据类型包括：<br/> ["actv", "actv_grad", "exp_avg", "exp_avg_sq", "grad_unreduced", "grad_reduced", "param_origin", "param_updated", "other"]。<br/>默认未配置本参数，表示转换全部数据类型。list\[str\]类型。 |
| step_partition | 输入      | 可选参数，控制数据库中按step分区的间隔，int类型，默认每500步一个表。 |
| output_dirpath | 输入      | 可选参数，指定转换后的输出路径，str类型，默认输出到"{curtime}_csv2db"文件夹，其中curtime为自动获取的当前时间戳。 |

**返回值说明**

无
