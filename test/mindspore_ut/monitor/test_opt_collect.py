import pytest
import numpy as np
from mindspore import Tensor, nn, ops
from unittest.mock import MagicMock, patch

from msprobe.core.common.const import MonitorConst
# Import the classes to test
from msprobe.core.common.log import logger
from msprobe.mindspore.monitor.optimizer_collect import (
    OptimizerMon,
    MixPrecisionOptimizerMon,
    MegatronDistributedOptimizerMon,
    MegatronChainedDistributedOptimizerMon,
    MegatronChainedMixPrecisionOptimizerMon,
    DeepSpeedZeroOptimizerMon,
    DeepSpeedZeroOptimizerStage0Mon,
    DeepSpeedZeroOptimizerStage1or2Mon,
    DeepSpeedZeroOptimizerStage3Mon,
    OptimizerMonFactory
)

class TestOptimizerMon:
    @classmethod
    def setup_class(cls):
        """Setup once for all tests in this class"""
        cls.mock_monitor = MagicMock()
        cls.mock_monitor.name2tag = {"test_param": {MonitorConst.POST_GRAD: "test_tag"}}
        cls.mock_monitor.duplicate_param = {}
        cls.mock_monitor.params_have_main_grad = False
        cls.mock_monitor.fsdp_wrapped_module = False
        cls.mock_monitor.mv_distribution = True
        cls.mock_monitor.mg_direction = True
        cls.mock_monitor.ur_distribution = True
        cls.mock_monitor.update_heatmap_visualizer = {"test_param": MagicMock()}
        cls.mock_monitor.ratio_heatmap_visualizer = {"test_param": MagicMock()}

    def test_fetch_grad_if_param_has_valid_grad_then_return_correct_grad_values(self):
        # Setup
        param = MagicMock()
        expected_grad = Tensor([1.0, 2.0, 3.0])
        param.grad = expected_grad
        params2name = {param: "test_param"}
        optimizer = MagicMock()
        mon = OptimizerMon(optimizer)
        
        # Execute
        result = mon.fetch_grad(self.mock_monitor, params2name)
        
        # Verify
        assert len(result) == 1
        assert (result["test_tag"] == expected_grad).all()
        self.mock_monitor.register_param_call_id.assert_called_once_with("hook_optimizer", "test_tag")

    def test_fetch_grad_if_param_has_main_grad_then_return_main_grad_values(self):
        # Setup
        param = MagicMock()
        expected_grad = Tensor(np.array([1.5, 2.5]))
        param.main_grad = expected_grad
        param.grad = None
        params2name = {param: "test_param"}
        optimizer = MagicMock()
        self.mock_monitor.params_have_main_grad = True
        mon = OptimizerMon(optimizer)
        
        # Execute
        result = mon.fetch_grad(self.mock_monitor, params2name)
        
        # Verify
        assert len(result) == 1
        assert (result["test_tag"] == expected_grad).all()

    def test_fetch_mv_if_state_complete_then_return_correct_momentum_values(self):
        # Setup
        param = MagicMock()
        params2name = {param: "test_param"}
        optimizer = MagicMock()
        optimizer.state = {
            param: {
                "exp_avg": Tensor([0.1]),
                "exp_avg_sq": Tensor([0.2]),
                "step": 10
            }
        }
        del optimizer.chained_optimizers
        del optimizer.param_to_cpu_states_map
        optimizer.defaults = {'betas': (0.9, 0.999), 'eps': 1e-8}
        optimizer.param_groups = [{}]
        
        mon = OptimizerMon(optimizer)
        mon.fp16_to_fp32_param = {}
        
        # Execute
        exp_avg, exp_avg_sq, update, ratio = mon.fetch_mv(self.mock_monitor, params2name)
        
        # Verify
        beta1, beta2 = optimizer.defaults['betas']
        step = optimizer.state[param]['step']
        
        expected_exp_avg_hat = 0.1 / (1 - beta1**step)
        expected_exp_avg_sq_hat = 0.2 / (1 - beta2**step)
        expected_update = expected_exp_avg_hat / (np.sqrt(expected_exp_avg_sq_hat) + optimizer.defaults['eps'])
        expected_ratio = expected_exp_avg_hat / np.sqrt(expected_exp_avg_sq_hat)
        
        assert exp_avg["test_param"] == Tensor([0.1])
        assert exp_avg_sq["test_param"] == Tensor([0.2])
        assert update["test_param"] == Tensor([expected_update])
        assert ratio["test_param"] == Tensor([expected_ratio])

    def test_narrow_from_flatten_if_state_not_partitioned_then_return_original_state(self):
        # Setup
        param = MagicMock()
        flatten_state = Tensor([1.0, 2.0, 3.0])
        mon = OptimizerMon(MagicMock())
        
        # Execute
        result = mon.narrow_from_flatten(param, flatten_state)
        
        # Verify
        assert (result == flatten_state).all()

class TestMixPrecisionOptimizerMon:
    @classmethod
    def setup_class(cls):
        cls.mock_monitor = MagicMock()
        cls.mock_monitor.mv_distribution = True
        cls.mock_monitor.mg_direction = True
        cls.mock_monitor.ur_distribution = True
        cls.mock_monitor.update_heatmap_visualizer = {'param1': MagicMock(), 'param2': MagicMock()}
        cls.mock_monitor.ratio_heatmap_visualizer = {'param1': MagicMock(), 'param2': MagicMock()}

    def test_map_fp16_to_fp32_param_if_multiple_groups_then_create_correct_mappings(self):
        # Setup
        optimizer = MagicMock()
        fp16_params = [MagicMock(), MagicMock(), MagicMock()]
        fp32_params = [MagicMock(), MagicMock(), MagicMock()]
        optimizer.float16_groups = [fp16_params[:2], [fp16_params[2]]]
        optimizer.fp32_from_float16_groups = [fp32_params[:2], [fp32_params[2]]]
        
        mon = MixPrecisionOptimizerMon(optimizer)
        
        # Execute
        mon.map_fp16_to_fp32_param(optimizer)
        
        # Verify
        assert len(mon.fp16_to_fp32_param) == 3
        for fp16, fp32 in zip(fp16_params, fp32_params):
            assert mon.fp16_to_fp32_param[fp16] == fp32

class TestDeepSpeedZeroOptimizerStage1or2Mon:
    @classmethod
    def setup_class(cls):
        """Setup once for all tests in this class"""
        cls.mock_monitor = MagicMock()
        cls.mock_monitor.name2tag = {"test_param": {MonitorConst.POST_GRAD: "test_tag"}}
        cls.mock_monitor.duplicate_param = {}
        cls.mock_monitor.params_have_main_grad = False
        cls.mock_monitor.mg_direction = True
        cls.mock_monitor.ur_distribution = True

    def test_fetch_grad_if_param_in_partition_then_return_correct_grad_slice(self):
        # Setup
        optimizer = MagicMock()
        param = MagicMock()
        params2name = {param: "test_param"}
        expected_grad = Tensor(np.array([1.0, 2.0, 3.0]))
        param.main_grad = expected_grad
        param.grad = None
        optimizer.bit16_groups = [[param]]
        optimizer.cpu_offload = False
        mon = DeepSpeedZeroOptimizerStage1or2Mon(optimizer)
        mon.param2group = {param: 0}
        mon.get_param_index = MagicMock(return_value=1)
        mon.param_not_in_partition = MagicMock(return_value=False)
        mon.get_position = MagicMock(return_value=(3, 3))  # start at index 3, length 3
        
        # MagicMock the averaged_gradients structure
        optimizer.averaged_gradients = {
            0: [
                None,  # index 0
                Tensor(np.array([1.0, 2.0, 3.0]))  # index 1
            ]
        }
        
        # Execute
        result = mon.fetch_grad(self.mock_monitor, params2name)
        
        # Verify
        assert len(result) == 1
        assert (result["test_tag"] == expected_grad).all()

class TestOptimizerMonFactory:
    @classmethod
    def setup_class(cls):
        cls.mock_monitor = MagicMock()
        cls.mock_monitor.mv_distribution = True
        cls.mock_monitor.mg_direction = True
        cls.mock_monitor.ur_distribution = True
        cls.mock_monitor.update_heatmap_visualizer = {'param1': MagicMock(), 'param2': MagicMock()}
        cls.mock_monitor.ratio_heatmap_visualizer = {'param1': MagicMock(), 'param2': MagicMock()}

    def test_create_optimizer_mon_if_chained_optimizer_then_return_correct_monitor_type(self):
        # Setup
        base_optimizer = MagicMock()
        base_optimizer.__class__.__name__ = "DistributedOptimizer"
        optimizer = MagicMock()
        optimizer.__class__.__name__ = "ChainedOptimizer"
        optimizer.chained_optimizers = [base_optimizer]
        
        # Execute
        result = OptimizerMonFactory.create_optimizer_mon(optimizer)
        
        # Verify
        assert isinstance(result, MegatronChainedDistributedOptimizerMon)

    def test_create_optimizer_mon_if_deepspeed_stage3_then_return_stage3_monitor(self):
        # Setup
        optimizer = MagicMock()
        optimizer.__class__.__name__ = "DeepSpeedZeroOptimizer_Stage3"
        
        # Execute
        result = OptimizerMonFactory.create_optimizer_mon(optimizer)
        
        # Verify
        assert isinstance(result, DeepSpeedZeroOptimizerStage3Mon)
        assert result.stage == '3'
