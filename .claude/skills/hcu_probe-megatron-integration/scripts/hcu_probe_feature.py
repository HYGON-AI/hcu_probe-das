from ..feature import AbstractFeature


def _train_step_target():
    """hcu_megatron(Megatron-LM-das)的 MegatronBasicFeature 用 hcu_megatron.training.training.train 替换了
    megatron 的 train,而它调用的是本模块内的 train_step;只 patch megatron.training.training.train_step 不会被调用。"""
    try:
        import hcu_megatron.training.training  # noqa: F401
        return 'hcu_megatron.training.training.train_step'
    except ImportError:
        return 'megatron.training.training.train_step'


class HcuProbeFeature(AbstractFeature):
    def __init__(self):
        super().__init__('hcu-probe-config', 2)

    def register_args(self, parser):
        group = parser.add_argument_group(title='hcu_probe')
        group.add_argument('--hcu-probe-config', type=str, default=None,
                           help='Path to hcu_probe config JSON file for precision debugging. '
                                'When set, tensor collection is automatically injected '
                                'around each train_step via the adaptor patch system.')
        group.add_argument('--hcu-probe-seed', type=int, default=None,
                           help='Fix random seed for reproducible precision debugging. '
                                'When set, calls hcu_probe seed_all() before training.')

    def validate_args(self, args):
        import os
        cfg = getattr(args, 'hcu_probe_config', None)
        if cfg and not os.path.isfile(cfg):
            raise FileNotFoundError(
                f'--hcu-probe-config file not found: {cfg}'
            )

    def register_patches(self, patch_manager, args):
        from hcu_megatron.features_manager.probe.hcu_probe_wrapper import (
            hcu_probe_train_wrapper,
            hcu_probe_train_step_wrapper,
        )

        patch_manager.register_patch(
            'megatron.training.training.train',
            hcu_probe_train_wrapper,
            apply_wrapper=True,
        )

        patch_manager.register_patch(
            _train_step_target(),
            hcu_probe_train_step_wrapper,
            apply_wrapper=True,
        )
