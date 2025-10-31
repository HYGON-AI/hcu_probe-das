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

from typing import List

from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


class RankGroupGenerator(object):
    def __init__(self, tensor_parallel: int, expert_parallel: int, data_parallel: int,
                 pipeline_parallel: int, context_parallel: int, order: str) -> None:
        self.tensor_parallel = tensor_parallel
        self.expert_parallel = expert_parallel
        self.data_parallel = data_parallel
        self.pipeline_parallel = pipeline_parallel
        self.context_parallel = context_parallel
        self.total_size = tensor_parallel * data_parallel * pipeline_parallel * context_parallel

        self.parallel_sizes = {
            "tp": self.tensor_parallel,
            "pp": self.pipeline_parallel,
            "dp": self.data_parallel,
            "ep": self.expert_parallel,
            "cp": self.context_parallel,
        }
        self.original_order = order
        normalized_order = order.lower()

        # 检查ep和dp是否相邻
        if 'ep' in normalized_order:
            if 'ep-dp' not in normalized_order and 'dp-ep' not in normalized_order:
                logger.error(f"The ep and dp must be adjacent in order ({self.original_order}).")
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

        # 检查所有非1的并行维度是否都在order中
        for name in self.parallel_sizes.keys():
            size = self.parallel_sizes[name]
            if name not in normalized_order:
                if size != 1:
                    logger.error(f"The parallel size ({name}) is ({size}), "
                                 f"but it's not specified in order ({self.original_order}).")
                    raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
                else:
                    normalized_order += '-' + name

        self.order_with_ep = normalized_order
        self.order_without_ep = '-'.join([item for item in normalized_order.split('-') if item != 'ep'])

        self.size_list_with_ep = []
        self.size_list_without_ep = []

        for item in normalized_order.split('-'):
            if item == 'dp':
                self.size_list_with_ep.append(self.data_parallel // self.expert_parallel)
                self.size_list_without_ep.append(self.data_parallel)
            elif item == 'ep':
                self.size_list_with_ep.append(self.expert_parallel)
            else:
                self.size_list_with_ep.append(self.parallel_sizes[item])
                self.size_list_without_ep.append(self.parallel_sizes[item])

    @staticmethod
    def create_mask(order_str: str, target_tokens: str) -> List[bool]:
        order_elements = order_str.split('-')
        target_elements = target_tokens.split('-')
        mask = [False] * len(order_elements)
        for token in target_elements:
            mask[order_elements.index(token)] = True
        return mask

    @staticmethod
    def create_masked_rank_groups(
            total_size: int,
            parallel_dims: List[int],
            mask: List[bool],
    ) -> List[List[int]]:
        def compute_prefix_products(dimensions: List[int], initial: int = 1) -> List[int]:
            products = [initial]
            current = initial
            for dim in dimensions:
                current *= dim
                products.append(current)
            return products

        def calculate_inner_product(a: List[int], b: List[int]) -> int:
            return sum(x * y for x, y in zip(a, b))

        def decompose_index(index: int, shape: List[int], strides: List[int] = None) -> List[int]:
            if strides is None:
                strides = compute_prefix_products(shape)
            indices = [(index // stride) % dim for dim, stride in zip(shape, strides)]

            # 验证分解是否正确
            if calculate_inner_product(indices, strides[:-1]) != index:
                error_msg = f"The index {index} with shape {shape} doesn't match decomposed indices {indices}."
                logger.error(error_msg)
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

            return indices

        # 分离被掩码和未被掩码的维度
        masked_dims = [dim for dim, is_masked in zip(parallel_dims, mask) if is_masked]
        unmasked_dims = [dim for dim, is_masked in zip(parallel_dims, mask) if not is_masked]

        # 计算全局、掩码和未掩码的步长
        global_strides = compute_prefix_products(parallel_dims)
        masked_strides = [stride for stride, is_masked in zip(global_strides, mask) if is_masked]
        unmasked_strides = [stride for stride, is_masked in zip(global_strides, mask) if not is_masked]

        # 计算组大小和组数
        group_dim = compute_prefix_products(masked_dims)[-1]
        group_count = total_size // group_dim

        # 生成所有组的rank
        rank_groups = []
        for group_idx in range(group_count):
            decomposed_group = decompose_index(group_idx, unmasked_dims)
            current_group = []
            for in_group_idx in range(group_dim):
                decomposed_rank = decompose_index(in_group_idx, masked_dims)
                rank_value = (calculate_inner_product(decomposed_rank, masked_strides) +
                              calculate_inner_product(decomposed_group, unmasked_strides))
                current_group.append(rank_value)
            rank_groups.append(current_group)

        return rank_groups

    def generate_ranks(self, token: str, separate_ep: bool = False) -> List[List[int]]:
        if separate_ep:
            parallel_dims = self.size_list_with_ep
            current_order = self.order_with_ep
        else:
            parallel_dims = self.size_list_without_ep
            current_order = self.order_without_ep

        mask = self.create_mask(current_order, token)
        return self.create_masked_rank_groups(self.total_size, parallel_dims, mask)

    def generate_all_ranks(self) -> dict:
        result = {}
        for token in ["dp", "pp", "tp"]:
            result[token] = self.generate_ranks(token)
            result[f"{token}_size"] = self.parallel_sizes[token]
        return result


def get_tp_pp_default_groups(
        total_world_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        order: str = "tp-cp-ep-dp-pp",
) -> tuple:
    context_parallel_size = 1
    expert_parallel_size = 1

    # 检查world_size是否可被各并行维度的乘积整除
    product = tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    if total_world_size % product != 0:
        logger.error(f"The world size ({total_world_size}) is not divisible by "
                     f"{tensor_parallel_size} x {pipeline_parallel_size} x {context_parallel_size}.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    data_parallel_size = total_world_size // product

    # 检查数据并行是否可被专家并行整除
    if data_parallel_size % expert_parallel_size != 0:
        logger.error(f"The data parallel size ({data_parallel_size}) is not divisible by expert parallel size.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    # 生成rank组
    rank_creator = RankGroupGenerator(
        tensor_parallel=tensor_parallel_size,
        expert_parallel=expert_parallel_size,
        data_parallel=data_parallel_size,
        pipeline_parallel=pipeline_parallel_size,
        context_parallel=context_parallel_size,
        order=order,
    )

    return rank_creator.generate_ranks('tp'), rank_creator.generate_ranks('pp')
