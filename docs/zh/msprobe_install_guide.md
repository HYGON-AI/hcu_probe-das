# msProbe工具安装指南

## 安装说明

本文主要介绍msProbe工具的安装方式。包括**从PyPI安装**、**下载whl包安装**和**从源码安装**三种方式。

推荐使用[miniconda](https://docs.anaconda.com/miniconda/)管理环境依赖。

```bash
conda create -n msprobe python
conda activate msprobe
```

## 从PyPI安装
```shell
pip install mindstudio-probe
```

## 下载whl包安装

下载msProbe的whl软件包，软件包下载请参见[版本说明](./release_notes.md)中的“版本配套说明”章节。

获取到whl软件包后执行如下命令进行安装。

1. 验证whl包，若校验码一致，则whl包在下载中没有受损。


   ```bash
   sha256sum {name}.whl
   ```

2. 安装whl包

   ```bash
   pip install ./mindstudio_probe-{version}-py3-none-any.whl
   ```
   若覆盖安装，请在命令行末尾添加`--force-reinstall`参数。

   上面提供的whl包链接不包含adump功能，如果需要使用adump功能，请参考[从源码安装](#3-从源码安装)下载源码编译whl包。

## 从源码安装

执行如下命令安装：

```shell
git clone https://gitcode.com/Ascend/MindStudio-Probe.git
cd MindStudio-Probe

pip install setuptools wheel

python setup.py build_tb_graph_ascend # 可选，安装模型分级可视化插件
python setup.py bdist_wheel --include-mod=adump --no-check
cd ./dist
pip install ./mindstudio_probe*.whl
```

通过执行`build_tb_graph_ascend`构建指令来安装模型分级可视化功能。关于模型分级可视化插件的详细功能及使用说明请参见[tb_graph_ascend](./accuracy_compare/tb_graph_ascend.md)。


|参数|说明|可选/必选|
|--|--|:--:|
|--include-mod|指定可选模块，可取值`adump`，表示在编whl包时加入adump模块。默认未配置该参数，表示编译基础包。<br>&#8226; adump模块用于MindSpore静态图场景L2级别的dump。<br>&#8226; 仅MindSpore 2.5.0及以上版本支持adump模块。<br>&#8226; 若使用源码安装，编译环境需支持GCC 7.5或以上版本，和CMake 3.14或以上版本。<br>&#8226; 生成的whl包仅限编译时使用的python版本和处理器架构可用。|可选|
|--no-check|指定可选模块`adump`后，会下载所依赖的第三方库包，下载过程会进行证书校验。--no-check可以跳过证书校验。|可选|

# 查看msprobe工具信息

```bash
pip show mindstudio-probe
```

示例如下：

```bash
Name: mindstudio-probe
Version: 1.0.x
Summary: Pytorch Ascend Probe Utils
Home-page: https://gitcode.com/Ascend/mstt/tree/master/debug/accuracy_tools/msprobe
Author: Ascend Team
Author-email: pmail_mindstudio@xx.com
License: Apache License 2.0
Location: /home/xxx/miniconda3/envs/xxx/lib/python3.x/site-packages/mindstudio_probe-1.0.x-py3.x.egg
Requires: matplotlib, numpy, openpyxl, pandas, pyyaml, tqdm, wheel
Required-by: 
```

# Ascend生态链接

## 安装CANN包

1. 根据CPU架构和NPU型号选择Toolkit或Kernel，可参见[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0001.html)和[昇腾社区](https://www.hiascend.cn/developer/download/community/result?module=cann)。

   运行示例：
   ```bash
   Ascend-cann-toolkit_x.x.x_linux-xxxx.run --full --install-path={cann_path}
   Ascend-cann-kernels_x.x.x_linux.run --install --install-path={cann_path}
   ```

2. 配置环境变量
   ```bash
   source {cann_path}/ascend-toolkit/set_env.sh
   ```

## 安装PyTorch_NPU

请参见[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)。

## 安装MindSpeed LLM

请参见[MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)。
