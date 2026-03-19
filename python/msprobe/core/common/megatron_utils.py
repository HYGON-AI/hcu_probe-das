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

import os
from tqdm import tqdm
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from msprobe.core.common.utils import is_int
from msprobe.core.common.file_utils import create_directory, remove_path
from msprobe.core.common.log import logger
from msprobe.core.common.parallel_state import RankGroupGenerator


class MegatronStepInfo:
    is_megatron = False
    is_forward = False
    is_backward = False
    forward_micro_step = -1
    backward_micro_step = -1

    @classmethod
    def reset(cls):
        """重置所有类属性到初始状态"""
        cls.is_megatron = False
        cls.is_forward = False
        cls.is_backward = False
        cls.forward_micro_step = -1
        cls.backward_micro_step = -1


def wrap_megatron_step(func, is_forward=True):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if not MegatronStepInfo.is_megatron:
            MegatronStepInfo.is_megatron = True
        if is_forward:
            MegatronStepInfo.is_forward = True
            MegatronStepInfo.is_backward = False
            MegatronStepInfo.forward_micro_step += 1
        else:
            MegatronStepInfo.is_forward = False
            MegatronStepInfo.is_backward = True
            MegatronStepInfo.backward_micro_step += 1
        return func(*args, **kwargs)

    return wrapped_func


def get_micro_step():
    return MegatronStepInfo.forward_micro_step if MegatronStepInfo.is_forward else MegatronStepInfo.backward_micro_step


def is_megatron():
    return MegatronStepInfo.is_megatron


@dataclass
class VisualizationConfig:
    """可视化配置数据类"""
    height_per_tp: float = 0.5
    width_per_stage: float = 2
    min_width: int = 6
    max_width: int = 60
    min_height: int = 4
    max_height: int = 100
    legend_width: int = 5
    font_sizes: Dict[str, int] = None

    def __post_init__(self):
        if self.font_sizes is None:
            self.font_sizes = {
                'title': 14,
                'axis': 12,
                'tick': 10,
                'legend': 10,
                'text': 9
            }


class ModelParallelismVisualizer:

    def __init__(self, args, groups_dict, rank_models):
        """
        初始化可视化器

        Args:
            args: 并行配置参数
            groups_dict: 并行分组信息
            rank_models: 每个rank的模型配置字典
        """
        self.groups_dict = groups_dict
        self.rank_models = rank_models
        self.config = VisualizationConfig()

        # 计算布局参数
        self.layout = self._calculate_layout(args)

    def _calculate_layout(self, args) -> Dict[str, Any]:
        """计算布局参数"""
        tp_groups = self.groups_dict["tp"]
        dp_groups = self.groups_dict.get(
            "dp", [[rank] for rank in range(args.world_size)])

        dp_size = args.data_parallel_size
        dp_groups = list(zip(*dp_groups))  # 转置DP分组
        vpp_size = args.virtual_pipeline_parallel_size

        # 计算模型代表rank
        model_representatives = {}
        for dp_idx, dp_group in enumerate(dp_groups):
            for tp_idx, tp_group in enumerate(tp_groups):
                representative_rank = next(
                    (r for r in dp_group if r in tp_group), None)
                if representative_rank is not None:
                    model_representatives[(dp_idx, tp_idx)
                                          ] = representative_rank

        return {
            'num_layers': args.num_layers,
            'tp_groups': tp_groups,
            'dp_groups': dp_groups,
            'dp_size': dp_size,
            'tp_size': args.tensor_parallel_size,
            'pp_size': args.pipeline_parallel_size,
            'vpp_size': vpp_size,
            'num_tp': len(tp_groups),
            'num_model_copies': len(dp_groups),
            'model_representatives': model_representatives
        }

    def _calculate_figure_size(self) -> Tuple[float, float]:
        """动态计算图表尺寸"""
        fig_height = min(
            max(self.config.min_height,
                self.config.height_per_tp * self.layout['num_tp']),
            self.config.max_height
        )
        fig_width = min(
            max(self.config.min_width,
                self.config.width_per_stage * self.layout['vpp_size']),
            self.config.max_width
        )
        # 为图例预留0空间
        return fig_width + self.config.legend_width, fig_height

    def _setup_colors(self):
        """设置颜色配置"""
        num_copies = self.layout['num_model_copies']

        if num_copies <= 20:
            self.model_colors = plt.cm.tab20(range(num_copies))
        elif num_copies <= 40:
            # 40个以内使用tab20色板组合
            self.model_colors = np.vstack(
                [plt.cm.tab20b(range(20)), plt.cm.tab20c(range(20))])
        else:
            # 超过40个使用连续色板
            colormap = plt.get_cmap('viridis')
            norm = plt.Normalize(0, num_copies - 1)
            self.model_colors = [colormap(norm(i)) for i in range(num_copies)]

    def _setup_axes(self, ax):
        """设置坐标轴"""
        ax.set_xlabel('Virtual Pipeline Stage',
                      fontsize=self.config.font_sizes['axis'])
        ax.set_ylabel('TP Group', fontsize=self.config.font_sizes['axis'])

        # 设置标题
        title = (f'Model Parallelism Configuration | '
                 f'Total Layers: {self.layout["num_layers"]} | '
                 f'DP={self.layout["dp_size"]} | '
                 f'TP={self.layout["tp_size"]} | '
                 f'PP={self.layout["pp_size"]} | '
                 f'VPP={self.layout["vpp_size"]}')
        ax.set_title(title, fontsize=self.config.font_sizes['title'], pad=20)

        # 设置坐标轴范围
        ax.set_xlim(-0.5, self.layout['vpp_size'] - 0.5)
        ax.set_ylim(-0.5, self.layout['num_tp'] - 0.5)
        ax.grid(True, linestyle='--', alpha=0.3)

        # 设置刻度
        self._setup_ticks(ax)

    def _setup_ticks(self, ax):
        """设置坐标轴刻度"""
        # X轴刻度
        ax.set_xticks(range(self.layout['vpp_size']))
        ax.set_xticklabels(
            [f'Stage {i}' for i in range(self.layout['vpp_size'])])

        # Y轴刻度（TP组标签）
        tp_labels = []
        for tp_idx, tp_group in enumerate(self.layout['tp_groups']):
            min_rank = min(tp_group)
            max_rank = max(tp_group)
            if min_rank == max_rank:
                # 单卡情况
                tp_labels.append(f'Rank{min_rank}')
            else:
                tp_labels.append(f'Group{tp_idx}: Ranks{min_rank}-{max_rank}')

        ax.set_yticks(range(self.layout['num_tp']))
        ax.set_yticklabels(tp_labels, fontsize=self.config.font_sizes['tick'])

    def _create_layer_text(self, layers: List) -> str:
        """创建层描述文本"""
        text_parts = []
        trans_layers = [l for l in layers if isinstance(l, int)]
        if "Embedding" in layers:
            text_parts.append("Embed")
        if trans_layers:
            text_parts.append(f"L{min(trans_layers)}-{max(trans_layers)}")
        if "OutputLayer" in layers:
            text_parts.append("Out")
        return "+".join(text_parts)

    def _draw_model_blocks(self, ax):
        """绘制模型块"""
        # 计算总块数
        total_blocks = sum(len(self.rank_models.get(representative_rank, []))
                           for representative_rank in self.layout['model_representatives'].values())
        if total_blocks > 5000:
            logger.warning(
                f"Large-scale drawing: {total_blocks} model stages, may take a long time")
        with tqdm(total=total_blocks, desc="Drawing model stages", unit="stage") as pbar:
            for (dp_idx, tp_idx), representative_rank in self.layout['model_representatives'].items():
                model_configs = self.rank_models.get(representative_rank, [])

                for vpp_stage, layers in enumerate(model_configs):
                    self._draw_single_block(
                        ax, dp_idx, tp_idx, vpp_stage, layers)
                    pbar.update(1)
        if total_blocks > 20000:
            logger.warning(f"Large-scale visualization: {total_blocks} model stages detected, "
                           f"this may cause memory overflow")

    def _draw_single_block(self, ax, dp_idx: int, tp_idx: int, vpp_stage: int, layers: List):
        """绘制单个模型块"""
        # 解析层信息
        layer_text = self._create_layer_text(layers)

        # 绘制矩形区块
        rect = Rectangle(
            (vpp_stage - 0.45, tp_idx - 0.45), 0.9, 0.9,
            facecolor=self.model_colors[dp_idx],
            edgecolor='white',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(rect)

        # 添加文本标签
        fontsize = self.config.font_sizes['text']
        ax.text(
            vpp_stage, tp_idx,
            layer_text,
            ha='center', va='center',
            fontsize=fontsize,
            bbox=dict(facecolor='white', alpha=0.3, pad=1)
        )

    def _add_legend(self, ax):
        """添加图例"""
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=self.model_colors[i],
                          alpha=0.8,
                          label=f'Model Copy {i+1}')
            for i in range(self.layout['num_model_copies'])
        ]

        ax.legend(
            handles=legend_elements,
            loc='center left',  # 以左侧中心为基准
            bbox_to_anchor=(1.02, 0.5),
            title=f"Model Copies",
            title_fontsize=self.config.font_sizes['legend'],
            fontsize=self.config.font_sizes['legend']
        )

    def _visualize(self) -> plt.Figure:
        """
        执行可视化
        """
        # 创建图表
        fig_width, fig_height = self._calculate_figure_size()
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), clear=True)
        self._setup_colors()
        self._setup_axes(ax)

        # 调整布局
        right_margin = 1 - self.config.legend_width / fig_width
        plt.tight_layout(rect=[0, 0, right_margin, 1])

        # 添加图例
        self._add_legend(ax)

        # 绘制模型块
        self._draw_model_blocks(ax)
        return fig

    def save(self, filepath: str):
        """
        保存可视化结果到文件
        """
        fig = self._visualize()
        logger.info(f"Saving visualization result to {filepath}...")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)


@dataclass
class ParallelConfig:
    world_size: int
    num_layers: int
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # 优先级高于virtual_pipeline_parallel_size
    num_layers_per_virtual_pipeline_stage: int = None
    order: str = "tp-cp-ep-dp-pp"
    standalone_embedding_stage: bool = False
    output_path: str = './'
    # _calculate_derived_parameters
    virtual_pipeline_parallel_size: int = 1
    data_parallel_size: int = None
    transformer_pipeline_parallel_size: int = None

    def __post_init__(self):
        self._validate()
        self._calculate_derived_parameters()
        self._create_output_path()

    def _calculate_derived_parameters(self):
        """计算派生参数"""
        # Pipeline模型并行大小
        self.pipeline_parallel_size = min(
            self.pipeline_parallel_size,
            (self.world_size // self.tensor_parallel_size)
        )

        self.transformer_pipeline_parallel_size = (
            self.pipeline_parallel_size - 1
            if self.standalone_embedding_stage else
            self.pipeline_parallel_size
        )

        self._validate_pipeline_configs()
        # 通过num_layers_per_virtual_pipeline_stage配置vpp
        if self.num_layers_per_virtual_pipeline_stage is not None:
            num_layers_per_pipeline_stage = self.num_layers // self.transformer_pipeline_parallel_size

            self.virtual_pipeline_parallel_size = (
                num_layers_per_pipeline_stage // self.num_layers_per_virtual_pipeline_stage
            )
            if self.virtual_pipeline_parallel_size > 1 and self.pipeline_parallel_size <= 1:
                raise ValueError(
                    "pipeline-model-parallel size should be greater than 1 with virtual-pipeline"
                )

        # 计算数据并行大小
        self.data_parallel_size = self.world_size // (
            self.tensor_parallel_size * self.pipeline_parallel_size
        )

    def _validate_pipeline_configs(self):
        """验证流水线层数配置"""
        if self.transformer_pipeline_parallel_size == 0 or \
                self.num_layers % self.transformer_pipeline_parallel_size != 0:
            if self.standalone_embedding_stage:
                raise ValueError(
                    f'number of layers ({self.num_layers}) must be divisible by '
                    f'transformer_pipeline_parallel_size ({self.transformer_pipeline_parallel_size}), '
                    f'which is pipeline_parallel_size-1 when standalone_embedding_stage is enabled.'
                )
            else:
                raise ValueError(
                    f'number of layers ({self.num_layers}) must be divisible by '
                    f'pipeline_parallel_size ({self.pipeline_parallel_size}).'
                )

        num_layers_per_pipeline_stage = self.num_layers // self.transformer_pipeline_parallel_size
        if self.num_layers_per_virtual_pipeline_stage is not None:
            if num_layers_per_pipeline_stage % self.num_layers_per_virtual_pipeline_stage != 0:
                raise ValueError(
                    f'number of layers per pipeline stage ({num_layers_per_pipeline_stage}) must be divisible by '
                    f'number of layers per virtual pipeline stage ({self.num_layers_per_virtual_pipeline_stage}).'
                )

    def _create_output_path(self):
        create_directory(self.output_path)
        file_name = f"ws{self.world_size}_ln{self.num_layers}" \
                    f"_tp{self.tensor_parallel_size}" \
                    f"_pp{self.pipeline_parallel_size}" \
                    f"_vpp{self.virtual_pipeline_parallel_size}.png"
        png_path = os.path.join(self.output_path, file_name)

        if os.path.exists(png_path):
            logger.warning(f"Existing path will be recovered: {png_path}")
            remove_path(png_path)
        self.output_path = png_path

    def _validate(self):
        """
        验证所有配置输入的类型
        order 会在RankGroupGenerator类中验证
        data_parallel_size 会在 _calculate_derived_parameters中计算覆盖
        """
        per_vpp = self.num_layers_per_virtual_pipeline_stage if \
            self.num_layers_per_virtual_pipeline_stage is not None else 1
        if self.tensor_parallel_size is None:
            self.tensor_parallel_size = 1
        if self.pipeline_parallel_size is None:
            self.pipeline_parallel_size = 1
        for name, _input in {
            "world_size": self.world_size,
            "num_layers": self.num_layers,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "num_layers_per_virtual_pipeline_stage": per_vpp
        }.items():
            if not is_int(_input) or _input < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.world_size > 1024:
            raise ValueError(
                "world_size ({self.world_size}) exceeds the maximum supported value of 1024.")
        if self.num_layers > 256:
            raise ValueError(
                "num_layers ({self.num_layers}) exceeds the maximum supported value of 256.")

        if (self.world_size % (self.tensor_parallel_size * self.pipeline_parallel_size) != 0):
            raise ValueError(
                f"world_size ({self.world_size}) is not divisible by "
                f"tensor_parallel_size({self.tensor_parallel_size}) x "
                f"pipeline_parallel_size ({self.pipeline_parallel_size})"
            )
        if not isinstance(self.standalone_embedding_stage, bool):
            raise ValueError(f"standalone_embedding_stage must be a boolean")


class ModelLayerSimulator:
    """
    模型仿真 - 统一处理模型层信息生成
    """

    def __init__(self, args: 'ParallelConfig', pp_stage_list: list):
        self.args = args
        self.pp_stage_list = pp_stage_list

    def get_model_for_rank(self, rank: int) -> List[List[str]]:
        """
        为指定rank生成模型层
        """
        if self._should_use_virtual_pipeline():
            return self._create_virtual_pipeline_model(rank)
        else:
            return self._create_standard_pipeline_model(rank)

    def _should_use_virtual_pipeline(self) -> bool:
        """判断是否使用虚拟流水线策略"""
        return self.args.virtual_pipeline_parallel_size > 1

    def _create_virtual_pipeline_model(self, rank: int) -> List[List[str]]:
        """虚拟流水线模式：生成多个虚拟阶段的模型配置"""
        model_configs = []
        for vpp_stage in range(self.args.virtual_pipeline_parallel_size):
            # 确定当前阶段是否为流水线的首尾阶段
            is_first_stage = self._is_pipeline_first_stage(rank, vpp_stage)
            is_last_stage = self._is_pipeline_last_stage(rank, vpp_stage)

            # 生成单个虚拟阶段的模型
            stage_config = self._create_single_stage_config(
                rank, vpp_stage, is_first_stage, is_last_stage
            )
            model_configs.append(stage_config)

        return model_configs

    def _create_standard_pipeline_model(self, rank: int) -> List[List[str]]:
        """标准流水线模式：生成单个模型配置"""
        is_first_stage = self._is_pipeline_first_stage(rank, 0)
        is_last_stage = self._is_pipeline_last_stage(rank, -1)

        model_config = self._create_single_stage_config(
            rank, 0, is_first_stage, is_last_stage
        )
        return [model_config]

    def _create_single_stage_config(self, rank: int, vpp_stage: int,
                                    is_first_stage: bool, is_last_stage: bool) -> List[str]:
        """
        生成单个阶段（虚拟阶段或普通阶段）的模型配置
        """
        components = []

        # 前处理组件（嵌入层）
        if is_first_stage:
            components.append("Embedding")
        # Transformer层组件
        components.extend(self._create_transformer_layers(rank, vpp_stage))
        # 后处理组件（输出层）
        if is_last_stage:
            components.append("OutputLayer")

        return components

    def _create_transformer_layers(self, rank: int, vpp_stage: int) -> List[str]:
        """创建Transformer层组件"""
        num_layers = self._calculate_transformer_layers(rank)
        if num_layers == 0:
            return []

        # 计算层偏移量
        offset = self._calculate_layer_offset(rank, vpp_stage)

        # 生成层名称
        return [i + 1 + offset for i in range(num_layers)]

    def _calculate_transformer_layers(self, rank: int) -> int:
        """计算当前rank需要处理的Transformer层数"""
        # 流水线并行场景
        if self.args.pipeline_parallel_size > 1:
            if (self.args.standalone_embedding_stage and
                    self._is_pipeline_first_stage(rank, 0)):
                return 0  # 嵌入阶段不包含Transformer层
            num_layers = self.args.num_layers // self.args.transformer_pipeline_parallel_size
            if self._should_use_virtual_pipeline():
                return num_layers // self.args.virtual_pipeline_parallel_size
            return num_layers
        # 非流水线并行场景
        return self.args.num_layers

    def _calculate_layer_offset(self, rank: int, vpp_stage: int) -> int:
        """计算层偏移量"""
        pipeline_rank = self._get_pipeline_rank(rank)
        if self.args.standalone_embedding_stage:
            pipeline_rank -= 1
        base_layers_per_stage = self.args.num_layers // self.args.transformer_pipeline_parallel_size

        if self._should_use_virtual_pipeline():
            # 虚拟流水线：进一步分割每个流水线阶段
            layers_per_virtual_stage = base_layers_per_stage // self.args.virtual_pipeline_parallel_size
            return vpp_stage * (self.args.num_layers // self.args.virtual_pipeline_parallel_size) + \
                (pipeline_rank * layers_per_virtual_stage)
        else:
            # 普通流水线：直接按流水线阶段划分
            return pipeline_rank * base_layers_per_stage

    def _is_pipeline_first_stage(self, rank: int, vpp_stage: int) -> bool:
        """判断是否为流水线首阶段"""
        return (rank in self.pp_stage_list[0] and vpp_stage == 0)

    def _is_pipeline_last_stage(self, rank: int, vpp_stage: int) -> bool:
        """判断是否为流水线尾阶段"""
        # 虚拟流水线：只有最后一个虚拟阶段才算是真正的尾阶段
        if (self._should_use_virtual_pipeline() and
                vpp_stage != (self.args.virtual_pipeline_parallel_size - 1)):
            return False

        return rank in self.pp_stage_list[-1]

    def _get_pipeline_rank(self, rank: int) -> int:
        """获取在流水线中的局部rank"""
        for local_rank, ranks in enumerate(self.pp_stage_list):
            if rank in ranks:
                return local_rank
        return 0

    # 批量生成所有rank的模型配置
    def generate_all_models(self) -> Dict[int, List[List[str]]]:
        """批量生成所有rank的模型配置"""
        return {rank: self.get_model_for_rank(rank) for rank in range(self.args.world_size)}


def plot_model_parallelism(args: 'ParallelConfig'):
    logger.info("Starting model parallelism visualization")
    logger.info("Generating model configurations for each rank...")
    groups_dict = RankGroupGenerator(
        args.tensor_parallel_size,
        1,  # expert_parallel
        args.data_parallel_size,
        args.pipeline_parallel_size,
        1,  # context_parallel
        args.order
    ).generate_all_ranks()
    pp_stage_list = list(zip(*groups_dict.get("pp", [])))
    rank_models = ModelLayerSimulator(
        args, pp_stage_list).generate_all_models()
    logger.info(f"Model configuration generation completed: "
                f"{len(groups_dict.get('tp', []))} TP groups, "
                f"{len(groups_dict.get('dp', []))} DP groups, "
                f"{len(groups_dict.get('pp', []))} PP groups")
    logger.info("Starting visualization...")
    ModelParallelismVisualizer(
        args, groups_dict, rank_models).save(args.output_path)
    logger.info("Done")
