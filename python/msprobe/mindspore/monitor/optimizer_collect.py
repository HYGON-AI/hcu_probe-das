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
from abc import abstractmethod

from mindspore import mint, ops

from msprobe.mindspore.common.log import logger
from msprobe.core.common.const import MonitorConst


class OptimizerMon(object):
    def __init__(self, optim) -> None:
        self.fp16_to_fp32_param = {}
        self.optim = optim
        self.state = {}

    def narrow_from_flatten(self, param, flatten_state):
        return flatten_state

    def get_state(self, optim):
        if hasattr(optim, 'chained_optimizers'):
            for opt in optim.chained_optimizers:
                self._get_single_state(opt)
        else:
            self._get_single_state(optim)

    def fetch_grad(self, monitor, params2name):
        if not self.fp16_to_fp32_param:
            self.map_fp16_to_fp32_param(self.optim)

        grad_dict = {}
        first_param = True
        for param, name in params2name.items():
            if monitor.duplicate_param.get(name, False):
                continue
            if self.fp16_to_fp32_param and param not in self.fp16_to_fp32_param:
                continue
            grad = param.main_grad if monitor.params_have_main_grad else param.grad
            element_in_cur_partition = self.fp16_to_fp32_param.get(param, param).numel()
            if param.numel() != element_in_cur_partition:
                if first_param:
                    grad = grad.flatten()[-element_in_cur_partition:]
                else:  # supposed to be the last one
                    grad = grad.flatten()[:element_in_cur_partition]
            first_param = False
            if grad is None:
                continue
            tag = monitor.name2tag.get(name, {}).get(MonitorConst.POST_GRAD)
            monitor.register_param_call_id("hook_optimizer", tag)
            grad_dict[tag] = grad
        return grad_dict

    def map_fp16_to_fp32_param(self, optim):
        pass

    def fetch_mv(self, monitor, params2name):
        if not self.fp16_to_fp32_param:
            self.map_fp16_to_fp32_param(self.optim)
        if not self.state:
            self.get_state(self.optim)

        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        update_dict = {}
        ratio_dict = {}

        if not self.state:
            logger.warning('optimizer state can not accessed')
            return exp_avg_dict, exp_avg_sq_dict, update_dict, ratio_dict

        for lp_param, name in params2name.items():
            if lp_param in self.fp16_to_fp32_param:
                hp_param = self.fp16_to_fp32_param[lp_param]
            else:
                hp_param = lp_param

            if hp_param in self.state:
                state_param = self.state.get(hp_param, {})
                exp_avg = self.narrow_from_flatten(lp_param, state_param.get("exp_avg", None))
                exp_avg_sq = self.narrow_from_flatten(lp_param, state_param.get("exp_avg_sq", None))
                if monitor.mv_distribution:
                    exp_avg_dict[name] = exp_avg
                    exp_avg_sq_dict[name] = exp_avg_sq
                if monitor.mg_direction:
                    exp_avg_dict[name] = exp_avg
                if monitor.ur_distribution:
                    if len(self.optim.param_groups) > 1:
                        logger.info(f"the length of optim.param_groups is {len(self.optim.param_groups)}.")
                    if 'step' in state_param:
                        step = state_param['step']  # Optimizer from pytorch or FusedAdam from apex(used by megatron)
                    elif 'step' in self.optim.param_groups[0]:
                        step = self.optim.param_groups[0]['step']  # AdamW from mindspeed
                    else:
                        logger.warning(f"step of {name} is None, maybe something wrong happened.")
                        continue
                    if exp_avg is None or exp_avg_sq is None:
                        logger.warning(f"exp_avg or exp_avg_sq of {name} is None, skip calculation.")
                        continue
                    exp_avg_hat = exp_avg / (1 - self.optim.defaults['betas'][0] ** step)
                    exp_avg_sq_hat = exp_avg_sq / (1 - self.optim.defaults['betas'][1] ** step)
                    update_dict[name] = exp_avg_hat / (mint.sqrt(exp_avg_sq_hat) + self.optim.defaults['eps'])
                    ratio_dict[name] = exp_avg_hat / mint.sqrt(exp_avg_sq_hat)
                    monitor.update_heatmap_visualizer[name].pre_cal(update_dict[name])
                    monitor.ratio_heatmap_visualizer[name].pre_cal(ratio_dict[name])
        return exp_avg_dict, exp_avg_sq_dict, update_dict, ratio_dict

    def _get_single_state(self, optim):
        state = {}
        if hasattr(optim, 'param_to_cpu_states_map'):
            state = optim.param_to_cpu_states_map
        elif hasattr(optim, 'state'):
            state = optim.state
        elif hasattr(optim, 'optimizer') and hasattr(optim.optimizer, 'state'):
            state = optim.optimizer.state
        self.state.update(state)


class MixPrecisionOptimizerMon(OptimizerMon):
    """
    混合精度优化器监控类。在混合精度训练中监控和管理优化器。
    混合精度训练通过适当降低某些计算的精度来加速训练过程并减少内存消耗。
    """

    def map_fp16_to_fp32_param(self, optim):
        for fp16_group, fp32_group in zip(optim.float16_groups, optim.fp32_from_float16_groups):
            for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                self.fp16_to_fp32_param[fp16_param] = fp32_param


class MegatronDistributedOptimizerMon(OptimizerMon):
    def map_fp16_to_fp32_param(self, optim):
        if not (hasattr(optim, "model_float16_groups") and
                hasattr(optim, "shard_fp32_from_float16_groups")):
            raise Exception(
                "megatron distributed optimizer should have model_float16_groups and shard_fp32_from_float16_groups, "
                "if not, please check megatron-lm version")
        for fp16_group, shard_fp32_group in zip(optim.model_float16_groups,
                                                optim.shard_fp32_from_float16_groups):
            for fp16_param, shard_fp32_param in zip(fp16_group, shard_fp32_group):
                self.fp16_to_fp32_param[fp16_param] = shard_fp32_param


class MegatronChainedDistributedOptimizerMon(MegatronDistributedOptimizerMon):
    def map_fp16_to_fp32_param(self, optim):
        for opt in optim.chained_optimizers:
            super().map_fp16_to_fp32_param(opt)


class MegatronChainedMixPrecisionOptimizerMon(MixPrecisionOptimizerMon):
    def map_fp16_to_fp32_param(self, optim):
        for opt in optim.chained_optimizers:
            super().map_fp16_to_fp32_param(opt)


class DeepSpeedZeroOptimizerMon(OptimizerMon):
    """
    Base monitor class for DeepSpeed ZeRO optimizer.
    ZeRO stage 0 no partition
    ZeRO stage 1 partitions optimizer states across data parallel processes.
    ZeRO stage 2 additionally partitions gradients.
    ZeRO stage 3 additionally partitions parameters.

    This class provides monitoring capabilities for ZeRO optimizers by:
    - Handling gradient collection for different ZeRO stages
    - Managing optimizer state access for monitoring
    """

    def __init__(self, optim):
        super().__init__(optim)
        self.stage = ''
        self.bit16_groups = []
        self.fp32_flat_groups = []
        self.param2group = ()
        self.param2index = []
        self.group_offset = {}

    @abstractmethod
    def get_grad_for_param(self, lp_param, group_idx, param_id):
        raise NotImplementedError

    def param_not_in_partition(self, lp_param, group_idx):
        param_slice_mapping = self.optim.state_dict()['param_slice_mappings'][group_idx]
        hp_address = param_slice_mapping.get(self.optim.param_names.get(lp_param))
        return hp_address is None

    def get_position(self, lp_param, group_idx):
        param_slice_mapping = self.optim.state_dict()['param_slice_mappings'][group_idx]
        hp_address = param_slice_mapping.get(self.optim.param_names.get(lp_param))
        return hp_address.start, hp_address.numel

    def get_group_index(self):
        param2group = {}
        for group_idx, bit16_group in enumerate(self.bit16_groups):
            for param in bit16_group:
                param2group[param] = group_idx
        return param2group

    def get_param_index(self, lp_param, group_idx):
        if not self.param2index:
            for group in self.bit16_groups:
                param2index = {}
                for index, param in enumerate(group):
                    param2index[param] = index
                self.param2index.append(param2index)

        return self.param2index[group_idx][lp_param]

    def narrow_from_flatten(self, param, flatten_state):
        if flatten_state is None:
            return flatten_state
        group_idx = self.param2group[param]
        if self.param_not_in_partition(param, group_idx):
            return None
        start, numel = self.get_position(param, group_idx)
        return flatten_state.narrow(0, start, numel)

    def map_fp16_to_fp32_param(self, optim):
        for group_idx, group in enumerate(self.bit16_groups):
            for param in group:
                self.fp16_to_fp32_param[param] = self.fp32_flat_groups[group_idx]

    def fetch_grad(self, monitor, params2name):
        grad_dict = {}
        for lp_param, name in params2name.items():
            group_idx = self.param2group[lp_param]
            param_id = self.get_param_index(lp_param, group_idx)
            if self.param_not_in_partition(lp_param, group_idx):
                continue
            if self.stage == '1or2':
                param_id = param_id - self.group_offset[group_idx] - 1
            grad = self.get_grad_for_param(lp_param, group_idx, param_id)
            tag = monitor.name2tag.get(name, {}).get(MonitorConst.POST_GRAD)
            monitor.register_param_call_id("hook_optimizer", tag)
            grad_dict[tag] = grad

        return grad_dict


class DeepSpeedZeroOptimizerStage0Mon(DeepSpeedZeroOptimizerMon):
    def __init__(self, optim):
        super().__init__(optim)
        self.stage = '0'
        self.bit16_groups = optim.bf16_groups
        self.fp32_flat_groups = optim.fp32_groups_flat_partition
        self.param2group = self.get_group_index()

    def get_grad_for_param(self, lp_param, group_idx, param_id):
        return self.optim.fp32_groups_gradient_dict[group_idx][param_id]


class DeepSpeedZeroOptimizerStage1or2Mon(DeepSpeedZeroOptimizerMon):
    def __init__(self, optim):
        super().__init__(optim)
        self.stage = '1or2'
        self.bit16_groups = optim.bit16_groups
        self.fp32_flat_groups = optim.single_partition_of_fp32_groups
        self.param2group = self.get_group_index()
        self.group_offset = {}
        self.get_group_offset()

    def get_grad_for_param(self, lp_param, group_idx, param_id):
        if getattr(self.optim, "cpu_offload", False):
            grads = self.optim.single_partition_of_fp32_groups[group_idx].grad
            start, numel = self.get_position(lp_param, group_idx)
            grad = grads.narrow(0, start, numel)
        else:
            grad = self.optim.averaged_gradients[group_idx][param_id]
        return grad

    def get_group_offset(self):
        for group_idx, group in enumerate(self.bit16_groups):
            self.group_offset[group_idx] = -1
            for lp_param in group:
                if self.param_not_in_partition(lp_param, group_idx):
                    self.group_offset[group_idx] = self.get_param_index(lp_param, group_idx)
                else:
                    break


class DeepSpeedZeroOptimizerStage3Mon(DeepSpeedZeroOptimizerMon):
    def __init__(self, optim):
        super().__init__(optim)
        self.stage = '3'
        self.bit16_groups = optim.fp16_groups
        self.fp32_flat_groups = optim.fp32_partitioned_groups_flat
        self.param2group = self.get_group_index()

    def param_not_in_partition(self, lp_param, group_idx):
        """Each param partioned across all zero ranks"""
        return False

    def get_position(self, lp_param, group_idx):
        param_id = self.optim.get_param_id(lp_param)
        return self.optim.grad_position[param_id][1:]

    def get_grad_for_param(self, lp_param, group_idx, param_id):
        return self.optim.averaged_gradients[group_idx][param_id]


class OptimizerMonFactory:
    _optimizer_mon_map = {
        "FP32Optimizer": OptimizerMon,
        "Float16OptimizerWithFloat16Params": MixPrecisionOptimizerMon,
        "DistributedOptimizer": MegatronDistributedOptimizerMon,
        "SwapDistributedOptimizer": MegatronDistributedOptimizerMon,
        "ChainedDistributedOptimizer": MegatronChainedDistributedOptimizerMon,
        "ChainedSwapDistributedOptimizer": MegatronChainedDistributedOptimizerMon,
        "ChainedFloat16OptimizerWithFloat16Params": MegatronChainedMixPrecisionOptimizerMon,
        "BF16_Optimizer": DeepSpeedZeroOptimizerStage0Mon,
        "DeepSpeedZeroOptimizer": DeepSpeedZeroOptimizerStage1or2Mon,
        "DeepSpeedZeroOptimizer_Stage3": DeepSpeedZeroOptimizerStage3Mon,
        "Adam": OptimizerMon
    }

    @staticmethod
    def create_optimizer_mon(optimizer):
        # auto replace opt_ty
        optimizer_class = optimizer.__class__.__name__
        if optimizer_class == "ChainedOptimizer":
            optimizer_class = "Chained" + optimizer.chained_optimizers[0].__class__.__name__
        logger.info(f'The optimizer type is {optimizer_class}')

        optimizer_mon_class = OptimizerMonFactory._optimizer_mon_map.get(optimizer_class, OptimizerMon)
        return optimizer_mon_class(optimizer)
