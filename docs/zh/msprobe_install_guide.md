# msProbe工具安装指南

## 安装说明

本文主要介绍msProbe工具的安装。当前仅支持**编译安装**方式。

推荐使用[miniconda](https://docs.anaconda.com/miniconda/)管理环境依赖。

```bash
conda create -n msprobe python
conda activate msprobe
```

## 编译安装

**功能说明**

通过setup.py脚本编译msProbe工具的whl软件包。

**命令格式**

```
python3 setup.py bdist_wheel [--include-mod=<include_mode>] [--no-check]
```

**参数说明**

| 参数          | 可选/必选 | 说明                                                         |
| ------------- | :-------: | ------------------------------------------------------------ |
| --include-mod |   可选    | 指定可选模块，可取值：<br/>&#8226; adump：表示在编译whl包时加入adump模块。adump模块用于MindSpore静态图场景L2级别的dump。仅MindSpore 2.5.0及以上版本支持adump模块。<br/>&#8226; tb_graph_ascend：表示在编译whl包时加入模型分级可视化插件。模型分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、Npm v10.8.2。模型分级可视化插件的详细依赖及功能使用说明请参见[tb_graph_ascend](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/tb_graph_ascend.md)。<br/>&#8226; atb_probe：表示在编译whl包时加入atb_probe模块。atb_probe模块用于ATB推理场景下的数据采集。<br/>&#8226; aclgraph_dump：表示在编译whl包时加入aclgraph_dump模块，用于在aclgraph场景通过acl_save保存.pt文件。编译环境需要额外依赖`torch`和`torch_npu`。<br/>默认未配置该参数，表示编译基础工具包。<br/>指定多个模块时，模块间以","连接，例如adump,atb_probe。<br/>指定adump或atb_probe模块时，编译环境需具备git、curl、GCC 7.5或以上版本、CMake 3.19.3或以上版本等第三方依赖软件。且指定adump模块时，使能的CANN环境下需包含`libadump_server.a`文件。<br/>配置该参数生成的whl包，仅限编译时使用的Python版本和处理器架构可用。 |
| --no-check    |   可选    | 跳过证书校验。--include-mod指定adump或 tb_graph_ascend后，会下载所依赖的第三方库包，下载过程会进行证书校验，配置本参数可以跳过证书校验。 |

**使用示例**

- 编译安装基础工具包

  ```
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 编译安装基础工具包和adump模块

  ```
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=adump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 编译安装基础工具包和aclgraph_dump模块

  ```
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 编译安装基础工具包和分级可视化插件

  ```
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 编译安装基础工具包和atb_probe模块

  ```
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install setuptools wheel

  python3 setup.py bdist_wheel --include-mod=atb_probe --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

**输出说明**

打印如下信息时，表示msProbe安装成功。

```
Successfully installed mindstudio-probe-{version}
```

## 卸载

执行如下命令卸载msProbe工具。

```
pip uninstall mindstudio-probe 
```

打印如下信息时，表示msProbe卸载成功。

```
Successfully uninstalled mindstudio-probe-{version}
```

## 升级

msProbe工具不支持直接升级，需要先完成[卸载](#卸载)后再重新[安装](#msProbe工具安装指南)。

## 查看msprobe工具信息

```bash
pip show mindstudio-probe
```

示例如下：

```bash
Name: mindstudio-probe
Version: 8.3.x
Summary: Ascend MindStudio Probe Utils
Home-page: https://gitcode.com/Ascend/MindStudio-Probe
Author: Ascend Team
Author-email: pmail_mindstudio@xx.com
License: Mulan PSL v2
Location: /home/xxx/miniconda3/envs/xxx/lib/python3.x/site-packages/
Requires: einops, matplotlib, numpy, onnx, onnxruntime, openpyxl, pandas, protobuf, pyyaml, rich, skl2onnx, tensorboard, tqdm, wheel
Required-by:
```

## Ascend生态链接

### 安装CANN包

1. 根据CPU架构和NPU型号选择Toolkit或Kernel，可参见[昇腾社区](https://www.hiascend.cn/developer/download/community/result?module=cann)下的《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit)》。

   运行示例：
   ```bash
   Ascend-cann-toolkit_{version}_linux-{arch}.run --full --install-path={cann_path}
   Ascend-cann-kernels_{version}_linux-{arch}.run --install --install-path={cann_path}
   ```

2. 配置环境变量
   ```bash
   source {cann_path}/Ascend/ascend-toolkit/set_env.sh
   ```

### 安装PyTorch_NPU

请参见[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)。

### 安装MindSpeed LLM

请参见[MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)。
