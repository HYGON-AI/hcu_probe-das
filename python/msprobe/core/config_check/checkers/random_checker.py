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


import random
from functools import wraps
from typing import Callable, List, Dict, Tuple, Optional
import inspect
import os
import json
from collections import defaultdict
import difflib

import numpy as np
import pandas as pd
from msprobe.core.config_check.config_checker import register_checker_item, register_pre_forward_fun_list
from msprobe.core.common.file_utils import create_file_in_zip, load_json
from msprobe.core.config_check.checkers.base_checker import BaseChecker
from msprobe.core.config_check.utils.utils import config_checking_print
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.const import Const
from msprobe.core.common.log import logger


# 数据结构：{随机操作名字: [{count: 调用次数, stack: 调用栈列表}]}
random_op_stats = defaultdict(list)


def get_call_stack(frame) -> List[str]:
    """获取详细的调用栈信息，每个元素包含完整路径、行号、函数名和代码行"""
    stack = []
    current_frame = frame.f_back  # 跳过当前函数
    
    while current_frame:
        frame_info = inspect.getframeinfo(current_frame)
        filename = os.path.abspath(frame_info.filename)
        code_line = frame_info.code_context[0].strip() if frame_info.code_context else ""
        
        # 格式化为详细的栈帧信息
        stack_entry = f"File {filename}, line {frame_info.lineno}, in {frame_info.function}, {code_line}"
        stack.append(stack_entry)
        
        current_frame = current_frame.f_back
    
    # 反转堆栈以显示正确的调用顺序（栈底到栈顶）
    return stack[::-1]


def track_random_call(func: Callable, name: str):
    """记录随机函数的调用信息"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe()
        stack = get_call_stack(frame)
        
        # 更新调用统计：操作名 -> [{count: 次数, stack: 调用栈列表}]
        # 检查是否已有相同调用栈的记录
        for entry in random_op_stats[name]:
            if entry['stack'] == stack:
                entry['count'] += 1
                break
        else:
            # 新增调用栈记录
            random_op_stats[name].append({'count': 1, 'stack': stack})
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            raise e
        finally:
            del frame
            
    return wrapper


def load_stats_files(directory: str) -> Dict[str, Dict[str, List[Dict]]]:
    """加载目录下所有统计文件并按rank组织数据"""
    rank_data = {}
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.startswith('rank') and file.endswith('.json'):
            rank = os.path.basename(file.split('.')[0])[4:]
            if not rank or not rank.isdigit():
                logger.error(f"extract rank id from {file} failed")
                raise ValueError
            
            # 加载并存储数据
            data = load_json(file_path)
            rank_data[int(rank)] = data

    return rank_data


def stack_match(stack1: List[str], stack2: List[str], threshold: float = 0.8) -> bool:
    """
    比较两个调用栈是否相似，同时考虑路径、函数名和代码行（各占1/3），每一层的相似度阈值需要达到0.8
    
    参数:
    - stack1: 第一个调用栈列表
    - stack2: 第二个调用栈列表
    - threshold: 相似度阈值，默认0.8
    
    返回:
    - 两个调用栈是否相似的布尔值
    """
    if len(stack1) != len(stack2):
        return False
    
    for frame1, frame2 in zip(stack1, stack2):
        # 提取路径、函数名和代码行
        path1, func1, code1 = _parse_frame(frame1)
        path2, func2, code2 = _parse_frame(frame2)
        
        # 计算相似度得分 (路径、函数名、代码行各占1/3权重)
        path_score = _compare_path(path1, path2)
        func_score = 1.0 if func1 == func2 else 0.0
        # 代码相似度
        code_score = difflib.SequenceMatcher(None, code1, code2).ratio()
        
        frame_score = (path_score + func_score + code_score) / 3.0
        if frame_score < threshold:
            return False

    return True


def _parse_frame(frame: str) -> Tuple[str, str, str]:
    """
    解析栈帧字符串，提取路径、函数名和代码行

    参数:
    - frame: 栈帧字符串。格式为"File {path}, line {line}, in {func}, {code}"
    
    返回:
    - path, func, code
    """
    path = func = code = ''
    stack_info = frame.split(' ')
    if len(stack_info) > 6:
        path = stack_info[1][:-1]
        func = stack_info[5][:-1]
        code = ' '.join(stack_info[6:])
    return path, func, code


def _compare_path(path1: str, path2: str) -> float:
    """比较两个路径的相似度，只考虑文件名"""
    if not path1 or not path2:
        return 0.0
    
    # 提取文件名（忽略目录路径）
    file1 = os.path.basename(path1)
    file2 = os.path.basename(path2)
    
    return 1.0 if file1 == file2 else 0.0


def find_matching_stack(bench_stack: List[str], cmp_stacks: List[Dict]) -> Optional[Dict]:
    """
    查找匹配的调用栈
    
    参数:
    - bench_stack: 基准侧的调用栈列表
    - cmp_stacks: 比较侧的调用栈条目列表，每个条目是{'count': 次数, 'stack': 调用栈列表}
    
    返回:
    - 匹配的调用栈条目或None
    """
    for cmp_entry in cmp_stacks:
        if stack_match(cmp_entry['stack'], bench_stack):
            return cmp_entry
    
    return None


def stack_list_to_string(stack_list):
    """
    将调用栈列表转换为换行分隔的字符串
    如果输入是特殊标记（如"no match stack"），则直接返回
    """
    if isinstance(stack_list, list):
        return '\n'.join(stack_list)
    return stack_list


def compare_random_calls(bench_dir: str = 'bench', cmp_dir: str = 'cmp') -> pd.DataFrame:
    """比较两个目录下的随机调用栈统计，生成详细比对结果"""
    bench_rank_data = load_stats_files(bench_dir)
    cmp_rank_data = load_stats_files(cmp_dir)
    
    # 获取所有rank
    all_ranks = sorted(set(bench_rank_data.keys()) | set(cmp_rank_data.keys()))
    
    results = []
    
    for rank in all_ranks:
        bench_data = bench_rank_data.get(rank, {})
        cmp_data = cmp_rank_data.get(rank, {})
        
        # 获取所有操作
        all_ops = set(bench_data.keys()) | set(cmp_data.keys())
        
        for op in all_ops:
            bench_stacks = bench_data.get(op, [])
            cmp_stacks = cmp_data.get(op, [])
            
            # 处理bench侧的每个调用栈
            for bench_entry in bench_stacks:
                bench_stack = bench_entry['stack']
                bench_count = bench_entry['count']
                
                # 查找匹配的cmp侧调用栈
                cmp_entry = find_matching_stack(bench_stack, cmp_stacks)
                
                if cmp_entry:
                    cmp_count = cmp_entry['count']
                    check_result = bench_count == cmp_count
                    results.append([op, rank, bench_stack, cmp_entry['stack'], bench_count, cmp_count, check_result])
                else:
                    # 没有匹配的调用栈
                    results.append([op, rank, bench_stack, "no match stack", bench_count, 0, False])
            
            # 处理cmp侧中没有在bench侧出现的调用栈
            for cmp_entry in cmp_stacks:
                cmp_stack = cmp_entry['stack']
                # 检查是否已经在上面处理过
                if not any(stack_match(bench_entry['stack'], cmp_stack) for bench_entry in bench_stacks):
                    results.append([op, rank, "no match stack", cmp_stack, 0, cmp_entry['count'], False])
    
    # 创建DataFrame
    df = pd.DataFrame(results, columns=RandomChecker.result_header)
    
    # 应用转换函数
    df['bench_stack'] = df['bench_stack'].apply(stack_list_to_string)
    df['cmp_stack'] = df['cmp_stack'].apply(stack_list_to_string)
    
    return df


def torch_patchs():
    """补丁Torch随机函数"""
    import torch
    torch_patches = {
        'rand': torch.rand,
        'randint': torch.randint,
        'randn': torch.randn,
        'rand_like': torch.rand_like,
        'randint_like': torch.randint_like,
        'randn_like': torch.randn_like,
        'manual_seed': torch.manual_seed
    }
    for name, func in torch_patches.items():
        setattr(torch, name, track_random_call(func, f"torch.{name}"))

    tensor_patches = {
        'exponential_': torch.Tensor.exponential_,
        'geometric_': torch.Tensor.geometric_,
        'log_normal_': torch.Tensor.log_normal_,
        'cauchy_': torch.Tensor.cauchy_
    }
    for name, func in tensor_patches.items():
        setattr(torch.Tensor, name, track_random_call(func, f"torch.Tensor.{name}"))


def mindspore_patchs():
    """补丁MindSpore随机函数"""
    import mindspore

    mindspore_ops_patches = {
        'rand': mindspore.ops.rand,
        'randint': mindspore.ops.randint,
        'randn': mindspore.ops.randn
    }
    for name, func in mindspore_ops_patches.items():
        setattr(mindspore.ops, name, track_random_call(func, f"mindspore.ops.{name}"))

    mindspore_patches = {
        'manual_seed': mindspore.set_seed
    }
    for name, func in mindspore_patches.items():
        setattr(mindspore, name, track_random_call(func, f"mindspore.{name}"))
    

@register_checker_item("random")
class RandomChecker(BaseChecker):
    input_needed = None
    target_name_in_zip = "random"
    result_header = ['op', 'rank', 'bench_stack', 'cmp_stack', 'bench_count', 'cmp_count', 'check_result']
    write_once = False

    @staticmethod
    def pack(pack_input):
        """打包随机调用统计到zip文件"""
        output_zip_path = pack_input.output_zip_path
        
        def collect_input(model, args, kwargs, step):
            if RandomChecker.write_once:
                return
                
            random_stats_dir = os.path.join(RandomChecker.target_name_in_zip)
            stats_filepath = os.path.join(random_stats_dir, f"rank{FmkAdp.get_rank_id()}.json")
            
            # 转换为JSON格式：{操作名: [{count: 次数, stack: 调用栈列表}]}
            stats_json = {}
            for op_name, entries in random_op_stats.items():
                stats_json[op_name] = entries
            
            create_file_in_zip(output_zip_path, stats_filepath, json.dumps(stats_json, indent=4))
            config_checking_print(f"已将随机调用统计打包到: {stats_filepath}")
            RandomChecker.write_once = True
            
        register_pre_forward_fun_list(collect_input)

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path, fmk):
        """比较两组随机调用统计"""
        bench_stats_path = os.path.join(bench_dir, RandomChecker.target_name_in_zip)
        cmp_stats_path = os.path.join(cmp_dir, RandomChecker.target_name_in_zip)
        
        df = compare_random_calls(bench_stats_path, cmp_stats_path)
        pass_check = Const.CONFIG_CHECK_PASS if False not in df['check_result'].values else Const.CONFIG_CHECK_ERROR
        
        return RandomChecker.target_name_in_zip, pass_check, df

    @staticmethod
    def apply_patches(fmk=Const.PT_FRAMEWORK):
        """应用随机函数补丁"""
        # 补丁Python random模块
        random_patches = {
            'random': random.random,
            'randint': random.randint,
            'uniform': random.uniform,
            'choice': random.choice
        }
        for name, func in random_patches.items():
            setattr(random, name, track_random_call(func, f"random.{name}"))

        # 补丁Numpy随机函数
        np_random_patches = {
            'rand': np.random.rand,
            'randint': np.random.randint,
            'choice': np.random.choice,
            'normal': np.random.normal
        }
        for name, func in np_random_patches.items():
            setattr(np.random, name, track_random_call(func, f"np.random.{name}"))

        # 补丁框架特定随机函数
        if fmk == Const.PT_FRAMEWORK:
            torch_patchs()
        elif fmk == Const.MS_FRAMEWORK:
            mindspore_patchs()
        else:
            raise Exception(f"不支持的框架: {fmk}, 支持的框架: {FmkAdp.supported_fmk}")
