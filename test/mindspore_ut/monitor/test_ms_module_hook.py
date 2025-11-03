import os
import unittest
from unittest.mock import MagicMock, patch
import mindspore as ms

from msprobe.mindspore.monitor.module_hook import TrainerMon, ModuleHookContext, OptimizerContext, GradContext

class TestTrainerMon(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(base_dir, "config/test_config.json")
        self.mock_config = {
            "start_step": 0,
            "collect_times": 2,
            "step_interval": 1,
            "targets": {"layer1": {}, "layer2": {}},
            "format": "csv",
            "ops": ["max", "min", "mean"],
            "xy_distribution": True,
            "mv_distribution": True,
            "wg_distribution": True,
            "param_distribution": True
        }
        self.trainer = TrainerMon(self.config_path)

    def test_init_given_valid_config_when_initialized_then_sets_correct_attributes(self):
        self.assertEqual(self.trainer.config_file_path, self.config_path)
        self.assertEqual(self.trainer.start_step, 0)
        self.assertEqual(self.trainer.collect_times, 2)
        self.assertTrue(self.trainer.monitoring)

    @patch('os.getenv', return_value='/custom/output')
    def test_get_output_base_dir_given_env_set_when_called_then_returns_custom_dir(self, mock_getenv):
        from msprobe.mindspore.monitor.module_hook import get_output_base_dir
        self.assertEqual(get_output_base_dir(), '/custom/output')

    @patch('os.path.getmtime', return_value=123456)
    @patch('json.load', return_value={})
    def test_dynamic_monitor_given_updated_config_when_called_then_updates_config(self, mock_load, mock_mtime):
        self.trainer.dynamic_enable = True
        self.trainer.config_timestamp = 0
        self.trainer.monitoring = False
        optimizer = MagicMock()
        self.trainer.optimizer_context[optimizer] = OptimizerContext()
        self.trainer.dynamic_monitor(optimizer)
        self.assertEqual(self.trainer.config_timestamp, 123456)

    def test_is_target_rank_given_rank_in_list_when_called_then_returns_true(self):
        self.trainer.module_rank_list = [0, 1]
        self.trainer.rank = 0
        self.assertTrue(self.trainer.is_target_rank())

    def test_is_target_rank_given_rank_not_in_list_when_called_then_returns_false(self):
        self.trainer.module_rank_list = [1, 2]
        self.trainer.rank = 0
        self.assertFalse(self.trainer.is_target_rank())

    def test_hook_optimizer_given_valid_optimizer_when_called_then_adds_hooks(self):
        optimizer = MagicMock()
        self.trainer.hook_optimizer(optimizer)
        self.assertEqual(len(self.trainer.pre_step_hooks), 1)
        self.assertEqual(len(self.trainer.post_step_hooks), 1)

    def test_write_xy_tb_given_activation_data_when_called_then_writes_metrics(self):
        context = ModuleHookContext("test_module")
        context.actv = {"key": ms.Tensor(1.0)}
        self.trainer.module_fwd_hook_context_by_module[MagicMock()] = context
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_xy_tb(1)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_write_grad_tb_given_grad_data_when_called_then_writes_metrics(self):
        self.trainer.grad_context.acc_metric = {"grad1": ms.Tensor(0.5)}
        self.trainer.grad_context.post = {"grad2": ms.Tensor(0.8)}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_grad_tb(1)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_write_mv_tb_given_mv_data_when_called_then_writes_metrics(self):
        context = OptimizerContext()
        context.exp_avg_metric = {"m1": ms.Tensor(0.1)}
        context.exp_avg_sq_metric = {"v1": ms.Tensor(0.2)}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_mv_tb(context)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_write_param_tb_given_param_data_when_called_then_writes_metrics(self):
        context = OptimizerContext()
        context.param_metric = {"param_pre": ms.Tensor(1.0), "param_post": ms.Tensor(2.0)}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_param_tb(context)
        self.trainer.summary_writer.write_metrics.assert_called()


class TestModuleHookContext(unittest.TestCase):
    def test_reset_clears_activation_data(self):
        context = ModuleHookContext("test")
        context.actv = {"data": ms.Tensor(1.0)}
        context.actvgrad = [ms.Tensor(2.0)]
        context.reset()
        self.assertEqual(len(context.actv), 0)
        self.assertEqual(len(context.actvgrad), 0)


class TestOptimizerContext(unittest.TestCase):
    def test_reset_clears_all_metrics(self):
        context = OptimizerContext()
        context.param_mg_direction = {"p1": 0.5}
        context.param_adam_update = {"p1": ms.Tensor(0.1)}
        context.reset()
        self.assertEqual(len(context.param_mg_direction), 0)
        self.assertEqual(len(context.param_adam_update), 0)


class TestGradContext(unittest.TestCase):
    def test_reset_clears_grad_data(self):
        context = GradContext()
        context.pre = {"g1": ms.Tensor(0.1)}
        context.post = {"g2": ms.Tensor(0.2)}
        context.reset()
        self.assertEqual(len(context.pre), 0)
        self.assertEqual(len(context.post), 0)


if __name__ == '__main__':
    unittest.main()