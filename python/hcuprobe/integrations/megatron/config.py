"""Megatron-specific arg translation + backward-compat shim.

Generic JSON config parsing was moved to
`hcu_probe.integrations._common.config`. This module re-exports the same
public symbols so existing imports keep working.

Megatron-specific helpers (those reading megatron-style argparse fields)
remain here:
    - resolve_dump_output_dir(args)
    - resolve_monitor_output_dir(args)
    - resolve_loss_output_dir(args)
    - compute_grad_acc_steps(args)
"""

# Re-export framework-agnostic helpers
from .._common.config import (  # noqa: F401
    read_json,
    get_dump_output_dir_from_cfg,
    get_monitor_output_dir_from_cfg,
)
from .._common.config import resolve_loss_output_dir as _common_resolve_loss_output_dir


# ---------------------------------------------------------------------------
#  Megatron-style facades (preserve backward compatibility)
# ---------------------------------------------------------------------------

def resolve_dump_output_dir(args):
    """Read output directory from `args.hcu_probe_config` JSON (L0/L1 dump)."""
    return get_dump_output_dir_from_cfg(getattr(args, "hcu_probe_config", None))


def resolve_monitor_output_dir(args):
    """Read output directory from `args.hcu_probe_monitor_config` JSON (Monitor V2)."""
    return get_monitor_output_dir_from_cfg(getattr(args, "hcu_probe_monitor_config", None))


def resolve_loss_output_dir(args):
    """Single source of truth for LossTracker output directory.

    Priority: --hcu-probe-config dump_path > --hcu-probe-monitor-config output_dir.
    Returns None when neither is set (LossTracker becomes a no-op).
    """
    return _common_resolve_loss_output_dir(
        dump_cfg_path=getattr(args, "hcu_probe_config", None),
        monitor_cfg_path=getattr(args, "hcu_probe_monitor_config", None),
    )


def compute_grad_acc_steps(args):
    """Compute gradient accumulation steps from Megatron batch config."""
    gbs = getattr(args, "global_batch_size", None)
    dp = getattr(args, "data_parallel_size", None)
    mbs = getattr(args, "micro_batch_size", None)
    if gbs and dp and mbs:
        return gbs // dp // mbs
    return 1
