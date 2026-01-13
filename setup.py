# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


__version__ = '8.3.1'

import os
import platform
import shutil
import subprocess
import sys
import fnmatch

import setuptools
from wheel.bdist_wheel import bdist_wheel

# 检查操作系统，如果不是Linux则报错
if platform.system() != "Linux":
    raise SystemError("This package only supports Linux platform. {}".format(platform.system()))

# ========== 仅清理tb_graph_ascend相关缓存 ==========
def clean_build_cache():
    """仅清理tb_graph_ascend相关缓存，避免其文件被意外打包"""
    # 1. 清理plugins下tb_graph_ascend的前端构建产物
    tb_fe_cache_dirs = [
        os.path.join("plugins", "tb_graph_ascend", "fe", "dist"),
        os.path.join("plugins", "tb_graph_ascend", "fe", "build"),
    ]
    
    # 2. 清理build目录中残留的tb_graph_ascend相关文件/目录
    build_tb_dirs = [
        os.path.join("build", "lib", "tb_graph_ascend"),
        os.path.join("build", "bdist.linux-x86_64", "wheel", "tb_graph_ascend"),
    ]
    
    # 合并所有需要清理的tb_graph_ascend相关路径
    tb_cache_paths = tb_fe_cache_dirs + build_tb_dirs
    
    # 执行清理
    cleaned_paths = []
    for dir_path in tb_cache_paths:
        if os.path.exists(dir_path):
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                else:
                    os.remove(dir_path)
                cleaned_paths.append(dir_path)
            except Exception as e:
                print(f"清理 {dir_path} 失败: {e}")
    
    # 输出清理结果
    if cleaned_paths:
        print(f"已清理tb_graph_ascend相关缓存")
    else:
        print("未发现tb_graph_ascend相关缓存需要清理")


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

        print("Installing npm dependencies...")
        install_result = subprocess.run(
            ["npm", "install", "--force"],
            capture_output=True,
            text=True
        )
        if install_result.returncode != 0:
            print(f"npm install failed: {install_result.stderr}")
            return False

        print("Building frontend with npm run buildLinux...")
        build_result = subprocess.run(
            ["npm", "run", "buildLinux"],
            capture_output=True,
            text=True
        )
        if build_result.returncode != 0:
            print(f"npm run buildLinux failed: {build_result.stderr}")
            return False
        else:
            print("Frontend build successful!")
            return True
    except Exception as e:
        print(f"Exception during frontend build: {e}")
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
    
    user_options = bdist_wheel.user_options + [
        ('include-mod=', None, 'Include specific modules (comma-separated)'),
        ('no-check', None, 'Skip checking'),
    ]
    
    def initialize_options(self):
        self.include_mod = None
        self.no_check = None
        super().initialize_options()
    
    def finalize_options(self):
        super().finalize_options()
        # 将include-mod参数传递给distribution
        if self.include_mod:
            self.distribution.include_mod = self.include_mod.split(',')
        else:
            self.distribution.include_mod = []
        
        # 处理no-check参数
        if self.no_check:
            os.environ["INSTALL_WITHOUT_CHECK"] = "1"
        else:
            # 清除环境变量，避免影响后续构建
            if "INSTALL_WITHOUT_CHECK" in os.environ:
                del os.environ["INSTALL_WITHOUT_CHECK"]

    def run(self):
        # 执行前先清理tb_graph_ascend缓存
        clean_build_cache()
        super().run()

    # ========== 过滤文件收集 ==========
    def make_distribution(self):
        """重写构建逻辑，过滤不需要的文件"""
        dist = super().make_distribution()
        # 过滤掉tb_graph_ascend相关文件（未指定时）
        if "tb_graph_ascend" not in self.distribution.include_mod:
            # 过滤文件列表
            new_files = []
            for path, ftype, metadata in dist.files:
                if not fnmatch.fnmatch(path, "tb_graph_ascend/*"):
                    new_files.append((path, ftype, metadata))
            dist.files = new_files
        return dist


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
    "onnxruntime >= 1.14.1, != 1.16.0",
    "skl2onnx >= 1.14.1"
]

if "--plat-name" in sys.argv or "--python-tag" in sys.argv:
    raise SystemError("Specifying platforms or python version is not supported.")

if platform.system() != "Linux":
    raise SystemError("MindStudio-Probe is only supported on Linux platforms.")

# 重置所有状态
include_mod_list = []
should_build_frontend = False
should_build_adump = False
has_include_mod = False

# 解析命令行参数（使用list避免迭代时修改列表）
sys_argv_copy = sys.argv.copy()
for i, arg in enumerate(sys_argv_copy):
    if arg.startswith("--include-mod"):
        has_include_mod = True
        # 处理--no-check参数
        if "--no-check" in sys.argv:
            os.environ["INSTALL_WITHOUT_CHECK"] = "1"
            sys.argv.remove("--no-check")
        
        # 解析include-mod参数
        if arg.startswith("--include-mod="):
            include_mod_list = arg[len("--include-mod="):].split(',')
            if arg in sys.argv:
                sys.argv.remove(arg)
        elif i + 1 < len(sys_argv_copy) and not sys_argv_copy[i + 1].startswith("--"):
            val = sys_argv_copy[i + 1]
            include_mod_list = val.split(',')
            if arg in sys.argv:
                sys.argv.remove(arg)
            if val in sys.argv:
                sys.argv.remove(val)
        
        # 只保留合法的模块名
        include_mod_list = [mod.strip() for mod in include_mod_list if mod.strip() in {"adump", "tb_graph_ascend"}]
        break

# 强制确保未指定参数时列表为空
if not has_include_mod:
    include_mod_list = []
    # 清理所有可能的残留环境变量
    if "INSTALL_WITHOUT_CHECK" in os.environ:
        del os.environ["INSTALL_WITHOUT_CHECK"]

# 检查构建需求
should_build_frontend = "tb_graph_ascend" in include_mod_list
should_build_adump = "adump" in include_mod_list

# 构建前端（如果需要）
if should_build_frontend:
    success = build_frontend()
    if not success:
        raise RuntimeError("Failed to build tb_graph_ascend frontend")

# 构建adump（如果需要）
if should_build_adump:
    arch = platform.machine()
    original_sys_argv = sys.argv.copy()
    sys.argv.append("--plat-name")
    sys.argv.append(f"linux_{arch}")
    sys.argv.append("--python-tag")
    sys.argv.append(f"cp{sys.version_info.major}{sys.version_info.minor}")
    build_cmd = f"bash ./build.sh -j16 -a {arch} -v {sys.version_info.major}.{sys.version_info.minor}"
    p = subprocess.run(build_cmd.split(), shell=False)
    sys.argv = original_sys_argv
    if p.returncode != 0:
        raise RuntimeError(f"Failed to build adump source({p.returncode})")

# 添加scripts脚本
current_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(current_dir, 'scripts')
dst_path = os.path.join(current_dir, 'python', 'msprobe', 'scripts')
if os.path.exists(src_path):
    if not os.path.isdir(dst_path):
        shutil.copytree(src_path, dst_path)
    else:
        for root, dirs, files in os.walk(src_path):
            target_root = os.path.join(dst_path, root[len(src_path) + 1:])
            for dir_name in dirs:
                os.makedirs(os.path.join(target_root, dir_name), mode=0o750, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(root, file), os.path.join(target_root, file))

# exclude确保不扫描任何tb_graph_ascend相关内容
packages = setuptools.find_packages(
    where="python",
    include=["msprobe*"],  # 只扫描msprobe开头的包
)

# 只有明确指定时才添加tb_graph_ascend相关包
if "tb_graph_ascend" in include_mod_list:
    tb_packages = [
        "tb_graph_ascend",
        "tb_graph_ascend.server",
        "tb_graph_ascend.fe",
    ]
    # 先检查目录是否存在，避免添加不存在的包导致报错
    valid_tb_packages = []
    for pkg in tb_packages:
        # 拼接包对应的目录路径
        pkg_path = package_dir.get(pkg, "").replace(".", os.sep) if "package_dir" in locals() else ""
        if not pkg_path:
            if pkg == "tb_graph_ascend":
                pkg_path = os.path.join("plugins", "tb_graph_ascend")
            elif pkg.startswith("tb_graph_ascend."):
                sub_pkg = pkg.split(".", 1)[1]
                pkg_path = os.path.join("plugins", "tb_graph_ascend", sub_pkg.replace(".", os.sep))
        # 检查目录是否存在
        if os.path.exists(pkg_path):
            valid_tb_packages.append(pkg)
        else:
            print(f"警告：包 {pkg} 对应的目录 {pkg_path} 不存在，已跳过")
    packages.extend(valid_tb_packages)

if "tb_graph_ascend" not in include_mod_list:
    packages = [pkg for pkg in packages if not pkg.startswith("tb_graph_ascend")]

# 设置entry_points
entry_points_dict = {
    'console_scripts': ['msprobe=msprobe.msprobe:main'],
}

# 只有在指定了tb_graph_ascend且前端已构建时才注册tensorboard插件
if "tb_graph_ascend" in include_mod_list and is_frontend_built():
    entry_points_dict['tensorboard_plugins'] = [
        'graph_ascend = tb_graph_ascend.server.plugin:GraphsPlugin',
    ]

# 设置package_dir
package_dir = {
    "": "python",
}

# 只有明确指定时才添加tb_graph_ascend路径映射
if "tb_graph_ascend" in include_mod_list:
    package_dir.update({
        "tb_graph_ascend": "plugins/tb_graph_ascend",
        "tb_graph_ascend.server": "plugins/tb_graph_ascend/server",
        "tb_graph_ascend.fe": "plugins/tb_graph_ascend/fe",
    })

# 设置package_data（仅在指定时添加）
package_data = {}
if "tb_graph_ascend" in include_mod_list:
    package_data = {
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
    }

setuptools.setup(
    name="mindstudio-probe",
    version=__version__,
    description="Ascend MindStudio Probe Utils",
    long_description="MindStudio-Probe is a set of tools for diagnosing and improving model accuracy on Ascend NPU.",
    url="https://gitcode.com/Ascend/MindStudio-Probe",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
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
    license='Mulan PSL v2',
    keywords='pytorch msprobe ascend',
    ext_modules=[],
    zip_safe=False,
    cmdclass={
        'bdist_wheel': CustomBdistWheelCommand,
    },
    entry_points=entry_points_dict,
)