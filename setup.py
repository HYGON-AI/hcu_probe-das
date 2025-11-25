# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__version__ = '8.3.1'

import os
import platform
import shutil
import subprocess
import sys

import setuptools
from wheel.bdist_wheel import bdist_wheel

# 检查操作系统，如果不是Linux则报错
if platform.system() != "Linux":
    raise SystemError("This package only supports Linux platform. {}".format(platform.system()))


def build_frontend():
    """构建前端资源"""
    fe_path = os.path.join("plugins", "tb_graph_ascend", "fe")

    if not os.path.exists(fe_path):
        return False

    original_cwd = os.getcwd()

    try:
        # 切换到fe目录
        os.chdir(fe_path)

        # 检查package.json是否存在
        if not os.path.exists("package.json"):
            return True

        # 安装依赖
        install_result = subprocess.run(
            ["npm", "install", "--force"],
            capture_output=True,
            text=True
        )
        if install_result.returncode != 0:
            return False

        # 执行构建
        build_result = subprocess.run(
            ["npm", "run", "buildLinux"],
            capture_output=True,
            text=True
        )
        if build_result.returncode != 0:
            return False
        else:
            return True
    except Exception:
        return False
    finally:
        # 切换回原始目录
        os.chdir(original_cwd)


def is_frontend_built():
    """检查前端是否已经构建完成"""
    fe_dist_path = os.path.join("plugins", "tb_graph_ascend", "fe", "dist")
    fe_build_path = os.path.join("plugins", "tb_graph_ascend", "fe", "build")

    # 检查是否存在构建产物
    index_html_exists = (
        os.path.exists(os.path.join(fe_dist_path, "index.html")) or
        os.path.exists(os.path.join(fe_build_path, "index.html"))
    )

    return index_html_exists


class BuildTbGraphAscendCommand(setuptools.Command):
    """自定义命令：只构建 tb_graph_ascend 前端部分"""

    description = "Build tb_graph_ascend frontend"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        success = build_frontend()
        if not success:
            raise Exception("tb_graph_ascend 前端构建失败")


class CustomBdistWheelCommand(bdist_wheel):
    """自定义wheel构建命令"""

    def run(self):
        # 检查前端是否已构建
        if is_frontend_built():
            # 包含所有包
            self.distribution.packages = packages
        else:
            # 只包含 msprobe 相关的包，排除 tb_graph_ascend
            self.distribution.packages = [pkg for pkg in packages if not pkg.startswith('tb_graph_ascend')]

        super().run()


INSTALL_REQUIRED = [
    "wheel",
    "einops",
    "numpy >=1.23.0, < 2.0",
    "pandas >= 1.3.5, < 2.1",
    "pyyaml",
    "tqdm",
    "openpyxl >= 3.0.6",
    "matplotlib",
    "tensorboard >= 2.11.2",
    "protobuf <= 3.20.2",
    "rich",
    "onnx >= 1.14.0",
    "onnxruntime >= 1.14.1,< 1.16.0",
    "skl2onnx >= 1.14.1"
]

if "--plat-name" in sys.argv or "--python-tag" in sys.argv:
    raise SystemError("Specifying platforms or python version is not supported.")

if platform.system() != "Linux":
    raise SystemError("MindStudio-Probe is only supported on Linux platforms.")

mod_list_range = {"adump", }
mod_list = []
for i, arg in enumerate(sys.argv):
    if arg.startswith("--include-mod"):
        if "--no-check" in sys.argv:
            os.environ["INSTALL_WITHOUT_CHECK"] = "1"
            sys.argv.remove("--no-check")
        if arg.startswith("--include-mod="):
            mod_list = arg[len("--include-mod="):].split(',')
            sys.argv.remove(arg)
        elif i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            mod_list = sys.argv[i + 1].split(',')
            sys.argv.remove(sys.argv[i + 1])
            sys.argv.remove(arg)
        mod_list = list(set(mod_list) & mod_list_range)
        break

# 当前只有adump一个mod
if mod_list:
    arch = platform.machine()
    sys.argv.append("--plat-name")
    sys.argv.append(f"linux_{arch}")
    sys.argv.append("--python-tag")
    sys.argv.append(f"cp{sys.version_info.major}{sys.version_info.minor}")
    build_cmd = f"bash ./build.sh -j16 -a {arch} -v {sys.version_info.major}.{sys.version_info.minor}"
    p = subprocess.run(build_cmd.split(), shell=False)
    if p.returncode != 0:
        raise RuntimeError(f"Failed to build source({p.returncode})")

# 添加scripts脚本
current_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(current_dir, 'scripts')
dst_path = os.path.join(current_dir, 'python', 'msprobe', 'scripts')
if not os.path.isdir(dst_path):
    shutil.copytree(src_path, dst_path)
else:
    for root, dirs, files in os.walk(src_path):
        target_root = os.path.join(dst_path, root[len(src_path) + 1:])
        for dir_name in dirs:
            os.makedirs(os.path.join(target_root, dir_name), mode=0o750, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(root, file), os.path.join(target_root, file))

# 只查找python目录下的包（msprobe相关）
packages = setuptools.find_packages(where="python")

# 手动添加tb_graph_ascend相关的包
tb_packages = [
    "tb_graph_ascend",
    "tb_graph_ascend.server",
    "tb_graph_ascend.fe",
]

packages.extend(tb_packages)

setuptools.setup(
    name="mindstudio-probe",
    version=__version__,
    description="Ascend MindStudio Probe Utils",
    long_description="MindStudio-Probe is a set of tools for diagnosing and improving model accuracy on Ascend NPU.",
    url="https://gitcode.com/Ascend/MindStudio-Probe",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    packages=packages,
    package_dir={
        "": "python",
        "tb_graph_ascend": "plugins/tb_graph_ascend",
        "tb_graph_ascend.server": "plugins/tb_graph_ascend/server",
        "tb_graph_ascend.fe": "plugins/tb_graph_ascend/fe",
    },
    package_data={
        "tb_graph_ascend.server": [
            "static/**",
            "static/**/*",
            "app/**",
            "app/**/*",
            "*.py",
            "**/*.py",
            "**/*.js",
            "**/*.css",
            "**/*.html"
        ],
    },
    platforms=["Linux"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache License 2.0',
    keywords='pytorch msprobe ascend',
    ext_modules=[],
    zip_safe=False,
    cmdclass={
        'bdist_wheel': CustomBdistWheelCommand,
        'build_tb_graph_ascend': BuildTbGraphAscendCommand,  # 新增自定义命令
    },
    entry_points={
        'console_scripts': ['msprobe=msprobe.msprobe:main'],
        'tensorboard_plugins': [
            'graph_ascend = tb_graph_ascend.server.plugin:GraphsPlugin',
        ],
    },
)
