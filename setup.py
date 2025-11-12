# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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


__version__ = '8.2.1'

import os
import subprocess
import platform
import sys
import setuptools


INSTALL_REQUIRED = [
    "wheel",
    "einops",
    "numpy >=1.23.0, < 2.0",
    "pandas >= 1.3.5, < 2.1",
    "pyyaml",
    "tqdm",
    "openpyxl >= 3.0.6",
    "matplotlib",
    "tensorboard",
    "protobuf <= 3.20.3",
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

setuptools.setup(
    name="mindstudio-probe",
    version=__version__,
    description="Ascend MindStudio Probe Utils",
    long_description="MindStudio-Probe is a set of tools for diagnosing and improving model accuracy on Ascend NPU.",
    url="https://gitcode.com/Ascend/MindStudio-Probe",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    packages=setuptools.find_packages(where="python"),  # 在python目录下查找包
    package_dir={"": "python"},
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
    entry_points={
        'console_scripts': ['msprobe=msprobe.msprobe:main'],
    }, )
