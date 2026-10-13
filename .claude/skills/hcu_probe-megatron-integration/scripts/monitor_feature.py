from ..feature import AbstractFeature


class MonitorFeature(AbstractFeature):
    def __init__(self):
        super().__init__('hcu-probe-monitor-config', 2)

    def register_args(self, parser):
        group = parser.add_argument_group(title='hcu_probe monitor')
        group.add_argument('--hcu-probe-monitor-config', type=str, default=None,
                           help='Path to monitor_v2 config JSON file for lightweight '
                                'training-state monitoring (weight_grad, module, optimizer, '
                                'param, cc). Uses TrainerMonitorV2 API.')

    def validate_args(self, args):
        import os
        cfg = getattr(args, 'hcu_probe_monitor_config', None)
        if cfg and not os.path.isfile(cfg):
            raise FileNotFoundError(
                f'--hcu-probe-monitor-config file not found: {cfg}'
            )

    def register_patches(self, patch_manager, args):
        from hcu_megatron.adaptor.features_manager.probe.monitor_wrapper import (
            monitor_setup_model_and_optimizer_wrapper,
            monitor_train_wrapper,
            monitor_train_step_wrapper,
        )

        patch_manager.register_patch(
            'megatron.training.training.setup_model_and_optimizer',
            monitor_setup_model_and_optimizer_wrapper,
            apply_wrapper=True,
        )

        patch_manager.register_patch(
            'megatron.training.training.train',
            monitor_train_wrapper,
            apply_wrapper=True,
        )

        patch_manager.register_patch(
            'megatron.training.training.train_step',
            monitor_train_step_wrapper,
            apply_wrapper=True,
        )
