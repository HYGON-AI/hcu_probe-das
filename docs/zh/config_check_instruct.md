# 训练前配置检查

## 简介

该工具主要适用于对比两个环境下可能影响训练精度的配置差异，包括：

- 环境变量
- 三方库版本
- 训练超参
- 权重
- 数据集
- 随机操作

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](./msprobe_install_guide.md)》。

**约束**

支持PyTorch和MindSpore框架。

## 使用说明

用户需要在两个待比对的训练的环境上分别进行数据采集，工具会采集两个环境下影响精度的配置，采集结果上传到同一机器进行比对。

### 数据采集

#### 静态数据采集

静态数据采集仅支持环境变量，三方库版本及训练超参采集，其中环境变量，三方库版本默认采集，训练超参采集需要用户传入启动训练的 shell 脚本路径或 yaml 配置文件，
支持多个输入，不传入表示不采集。

启动命令如下
```
msprobe config_check [-d <**.sh> <**.yaml>] [-o <output_path>]
```

| 参数            | 可选/必选 | 说明|
|---------------|-------|----------------------------------------------|
| -d 或 --dump   | 必选    | 代表数据采集模式，可传入启动训练的 shell 脚本路径或 yaml 配置文件路径，不传入代表不采集训练超参，只采集环境变量和三方库信息。 |
| -o 或 --output | 可选    | 代表输出路径，可选，默认为 config_check_pack.zip，必须以 `.zip` 后缀结尾。         |


#### 动态数据采集


在训练流程执行到的第一个python脚本开始处插入如下代码：
```
from msprobe.core.config_check import ConfigChecker
ConfigChecker.apply_patches(fmk)
```

**参数说明**

apply_patches：启动数据采集所需的各项patch，参数如下：

- **fmk**：训练框架。可选 pytorch 和 mindspore ，不传默认为 pytorch。

在模型初始化好之后插入如下代码：

```
from msprobe.core.config_check import ConfigChecker
ConfigChecker(model=model, shell_path="", output_zip_path="./config_check_pack.zip", fmk="pytorch")
```
ConfigChecker对模型挂上数据采集所需的hook，会在每次模型前向将要被执行的一刻进行数据采集。

**参数说明**

- **model**：可选参数，初始化好的模型。不传或缺省就不会采集权重和数据集。
- **shell_path**：可选参数，动态采集模式下支持 **megatron** 训练超参自动捕获，使用 **megatron** 时推荐不传入，其他情况下可传入训练脚本路径，类型为列表，传入一个或多个训练配置/启动脚本。不传或缺省就不会采集超参。
- **output_zip_path**：可选参数，输出zip包的路径，不传默认为"./config_check_pack.zip"。
- **fmk**：可选参数，当前训练框架。可选 pytorch 和 mindspore ，默认为 pytorch。

采集完成后会得到一个zip包，里面包括各项[影响精度的配置](#简介)。会分rank和step存储，其中step为micro_step。

在另一个环境上执行上述操作，得到另一个zip包

### 数据比对

将两个zip包传到同一个环境下，使用如下命令进行比对：

```
msprobe config_check -c bench_zip_path cmp_zip_path [-o <output_path>]
```

| 参数            | 可选/必选 | 说明                                                                         |
|---------------|-------|----------------------------------------------------------------------------|
| -c 或--compare | 必选    | 数据对比，有两个参数。其中 **bench_zip_path** 为标杆侧采集到的数据， **cmp_zip_path** 为待对比侧采集到的数据。 |
| -o 或 --output | 可选    | 代表输出路径，**output_path 里原有的比对结果会被覆盖**，默认为 config_check_result。                                         |


## 输出结果文件说明

 在比对输出的 **output_path** 里会生成2个目录和1个文件：
- bench：bench_zip_path里打包的数据。
- cmp：cmp_zip_path里打包的数据。
- result.xlsx：比对结果。里面会有多个sheet页，其中**summary**总览通过情况，其余页是具体检查项的详情。其中step为micro_step。

| file_name       |pass_check|
|-----------------|----------|
| env             |pass|
| pip             |pass|
| dataset         |pass|
| weights         |pass|
| hyperparameters |pass|
| random          |pass|

以上六项分别对应环境变量， 三方库版本， 数据集， 权重， 训练超参， 随机函数检查。

前五项检查在**精度比对**前必须保证达成。

## FAQ

1. 在使用 MindSpeed-LLM 进行数据采集时，需要注意动态数据采集中的 [apply_patches](#动态数据采集) 函数需要在 MindSpeed-LLM 
框架 pretrain_gpt.py 的 megatron_adaptor 函数导入之后执行。

2. 静态数据采集功能只能获取到系统中的环境变量，shell 脚本中解析的超参不支持复杂运算的数据还原，有类似问题时建议使用[动态采集方式](#动态数据采集)。
