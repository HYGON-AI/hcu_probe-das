"""Backward-compat shim for the loss tracker.

Real implementation moved to `hcu_probe.integrations._common.loss` (framework-
agnostic). This module re-exports the same public symbols so existing
`from hcu_probe.integrations.megatron.loss import LossTracker, get_tracker`
imports keep working unchanged.

New code SHOULD import from `_common.loss` directly.
"""

# Re-export framework-agnostic symbols
from .._common.loss import (  # noqa: F401
    LossTracker,
    close_tracker,
    get_existing_tracker,
)
from .._common.loss import get_tracker as _common_get_tracker
from .._common.config import resolve_loss_output_dir as _resolve_loss_output_dir


def get_tracker(args):
    """Megatron-style facade preserved for backward compatibility.

    Reads `args.hcu_probe_config` / `args.hcu_probe_monitor_config` and forwards
    to the framework-agnostic `_common.loss.get_tracker(output_dir=...)`.

    Returns None when neither --hcu-probe-config nor --hcu-probe-monitor-config
    is set, making all downstream record(...) calls no-ops.
    """
    from .dump import _is_megatron_dp_rank_0

    output_dir = _resolve_loss_output_dir(
        dump_cfg_path=getattr(args, "hcu_probe_config", None),
        monitor_cfg_path=getattr(args, "hcu_probe_monitor_config", None),
    )
    return _common_get_tracker(
        output_dir=output_dir, is_writing_rank_fn=_is_megatron_dp_rank_0
    )
