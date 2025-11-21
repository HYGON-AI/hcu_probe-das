# -------------------------------------------------------------------------
# Copyright (c) 2025, Huawei Technologies.
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
# --------------------------------------------------------------------------------------------#
import os
import shutil
import subprocess
import sys
import tempfile
import glob
import zipfile
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecureBuildManager:
    """安全构建管理器"""
    
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.original_cwd = os.getcwd()
    
    @staticmethod
    def _fix_whl_permissions(whl_file_path):
        """修复whl文件内权限"""
        temp_file_path = whl_file_path + ".new"
        
        try:
            with zipfile.ZipFile(whl_file_path, 'r') as source_zip:
                with zipfile.ZipFile(temp_file_path, 'w') as target_zip:
                    for zip_item in source_zip.infolist():
                        file_data = source_zip.read(zip_item.filename)
                        
                        new_zip_item = zipfile.ZipInfo(zip_item.filename)
                        new_zip_item.date_time = zip_item.date_time
                        new_zip_item.compress_type = zip_item.compress_type
                        
                        # 设置文件权限
                        permission = 0o550 << 16 if zip_item.filename.endswith('/') else 0o440 << 16
                        new_zip_item.external_attr = permission
                        
                        target_zip.writestr(new_zip_item, file_data)
            
            os.replace(temp_file_path, whl_file_path)
            
        except Exception as error:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise error
    
    @staticmethod
    def _transfer_artifacts(temp_dist_dir, original_dist_dir):
        """将构建产物传输回原目录"""
        os.makedirs(original_dist_dir, exist_ok=True)
        
        for whl_file in glob.glob(os.path.join(temp_dist_dir, '*.whl')):
            target_file = os.path.join(original_dist_dir, os.path.basename(whl_file))
            shutil.copy2(whl_file, target_file)

    def execute_secure_build(self):
        """执行安全构建流程"""
        with tempfile.TemporaryDirectory(prefix="secure_build_") as temp_dir:
            logger.info("创建安全构建环境")
            
            self._copy_required_files(temp_dir)
            os.chdir(temp_dir)
            
            try:
                logger.info("开始构建wheel包")
                build_result = subprocess.run(
                    [sys.executable, 'setup.py', 'bdist_wheel'],
                    capture_output=True, text=True, check=True
                )
                
                if build_result.stderr:
                    logger.warning("构建过程中出现警告")
                
                # 在安全环境中修复权限
                temp_dist_dir = os.path.join(temp_dir, 'dist')
                if os.path.exists(temp_dist_dir):
                    logger.info("执行权限修复")
                    for whl_file in glob.glob(os.path.join(temp_dist_dir, '*.whl')):
                        SecureBuildManager._fix_whl_permissions(whl_file)
                
                # 传输构建产物
                original_dist_dir = os.path.join(self.source_dir, 'dist')
                SecureBuildManager._transfer_artifacts(temp_dist_dir, original_dist_dir)
                
                logger.info("安全构建完成")
                
            except subprocess.CalledProcessError as error:
                logger.error("构建过程失败")
                raise error
            finally:
                os.chdir(self.original_cwd)

    def _copy_required_files(self, target_dir):
        """复制必要的文件到目标目录"""
        required_items = ['setup.py', 'server']
        
        for item in required_items:
            source_path = os.path.join(self.source_dir, item)
            target_path = os.path.join(target_dir, item)
            
            if not os.path.exists(source_path):
                continue
                
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
            else:
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)


class BuildCleaner:
    """构建清理器"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def clean_all(self):
        """清理所有构建产物"""
        cleanup_patterns = [
            'build',
            'dist',
            '*.egg-info',
            '__pycache__',
            '*.pyc',
            '*.pyo'
        ]
        
        for pattern in cleanup_patterns:
            for path in glob.glob(os.path.join(self.base_dir, pattern)):
                if not os.path.exists(path):
                    continue
                    
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                except Exception as error:
                    logger.warning(f"清理失败: {error}")


def main():
    """主执行函数"""
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        cleaner = BuildCleaner(os.path.dirname(os.path.abspath(__file__)))
        cleaner.clean_all()
    else:
        source_directory = os.path.dirname(os.path.abspath(__file__))
        build_manager = SecureBuildManager(source_directory)
        build_manager.execute_secure_build()


if __name__ == "__main__":
    main()