# 推理离线模型数据采集
## 简介
提供了传统小模型场景下tensor数据dump功能，获得精度数据，用于模型精度定位。适用于ONNX、OM模型，用户只需要通过参数指定原始模型对应的离线模型。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参加《[msProbe安装指南](../msprobe_install_guide.md)》。<br>
采集OM模型数据依赖aisbench包和aclruntime包，用户使用前可通过以下命令安装这两个依赖包。<br>
 ```sh
  msprobe install_deps -m offline [--no_check]
  ```
需要注意的是，--no_check参数，会跳过检查目标网站的证书信息，有一定的安全风险，用户需要谨慎使用并自行承担后果。

**约束**

仅支持ONNX、OM模型数据采集。

**安全风险提示**

在模型文件传给工具加载前，用户需要确保传入文件是安全可信的，若模型文件来源官方有提供SHA256等校验值，用户必须要进行校验，以确保模型文件没有被篡改。

## 推理离线模型数据采集功能介绍

### 功能说明
离线模型精度数据采集。

### 注意事项

仅支持ONNX、OM模型数据采集。

### 命令格式

 ```sh
  msprobe offline_dump --model_path /model_path/model.onnx(.om)/ -o /dump_output_path
  ```

### 参数说明

| 参数名                  | 描述                                                                                                                                                                                                     | 必选 |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|
| --model_path         | 模型文件[.onnx, .om]路径，路径需使用绝对路径，分别对应ONNX、OM。                                                                                                                                                              | 是  |
| --input_data         | 模型的输入数据路径，路径需使用绝对路径，默认根据模型的input随机生成，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy。注意：使用aipp模型时该输入为OM模型的输入，且支持自动将npy文件转为bin文件。                                                 | 否  |
| -o或--output_path     | 输出文件路径，路径需使用绝对路径，默认为当前路径的output文件夹。                                                                                                                                                                    | 否  |
| --input_shape        | 模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300"，节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称。                                                                                              | 否  |
| --rank               | 指定运行设备[0,255]，可选参数，默认0。                                                                                                                                                                                | 否  |
| --dym_shape_range    | 动态shape的阈值范围。如果设置该参数，那么将根据参数中所有的shape列表进行依次推理和精度比对。<br/>配置格式为："input_name1:1,3,200\~224,224-230;input_name2:1,300"。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 <br/> | 否  |
| --onnx_fusion_switch | onnxruntime算子融合开关，默认开启算子融合，如存在onnx dump数据中因算子融合导致缺失的，建议关闭此开关。使用方式：--onnx_fusion_switch False。                                                                                                          | 否  |

## 输出结果文件说明

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape}用来区分动态shape时不同的模型实际输入，静态shape时没有该层
├-- dump_data
│   ├-- npu                          # npu dump数据目录
│   │   ├-- {timestamp}              # 模型所有npu dump的算子输出
│   │   │   └-- 0                    # Rank设备ID号
│   │   │       └-- {om_model_name}  # 模型名称
│   │   │           └-- 1            # 模型ID号
│   │   │               ├-- 0        # 针对每个Task ID执行的次数维护一个序号，从0开始计数，该Task每dump一次数据，序号递增1
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {time_stamp}
│   │   │   ├-- output_0.bin
│   │   │   └-- output_0.npy
│   │   └-- {time_stamp}_summary.json
│   └-- {onnx} # 原模型dump数据存放路径，onnx分别对应ONNX模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
└-- model
    ├-- {om_model_name}.json                    # 离线模型OM模型(.om)通过atc工具转换后的json文件
    └-- new_{onnx_model_name}.onnx              # 把每个算子作为输出节点后新生成的ONNX模型

```
