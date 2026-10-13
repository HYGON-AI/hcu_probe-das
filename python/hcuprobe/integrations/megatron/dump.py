"""Megatron-side translator for the dump (PrecisionDebugger) lifecycle.

Reads Megatron `args` namespace and forwards keyword-only calls to the
framework-agnostic `_common.lifecycle.DumpLifecycle`. All business logic
lives in `_common`; this file is just argument translation.

Hook contract (called from the hcu_megatron probe wrapper):
    on_train_start(args)
    on_step_start(model, args)
    on_step_end(result, args)
    on_train_end()

`result` is the tuple returned by Megatron's `train_step`:
    (loss_dict, skipped_iter, should_checkpoint, should_exit,
     exit_code, grad_norm, num_zeros_in_grad, ...)
"""

from .._common.lifecycle import get_dump_lifecycle
from .._common.config import resolve_loss_output_dir
from .config import compute_grad_acc_steps  # re-exported below for backward compat  # noqa: F401


def _is_megatron_dp_rank_0():
    """Megatron-aware DP-rank check (override default torch rank=0)."""
    try:
        import torch
        from megatron.core import parallel_state as mpu
        if not torch.distributed.is_initialized():
            return True
        return mpu.get_data_parallel_rank() == 0
    except Exception:
        return True


def on_train_start(args):
    """Invoked once at train() entry. Safe to call when dump is disabled."""
    cfg_path = getattr(args, "hcu_probe_config", None)
    if not cfg_path:
        return
    get_dump_lifecycle().on_train_start(
        dump_cfg_path=cfg_path,
        seed=getattr(args, "hcu_probe_seed", None),
        reproduce=getattr(args, "reproduce", False),
        loss_output_dir=resolve_loss_output_dir(
            dump_cfg_path=cfg_path,
            monitor_cfg_path=getattr(args, "hcu_probe_monitor_config", None),
        ),
        is_writing_rank_fn=_is_megatron_dp_rank_0,
    )


def on_step_start(model, args):
    """Invoked at the top of each train_step when dump is active."""
    if not getattr(args, "hcu_probe_config", None):
        return
    model_obj = model[0] if isinstance(model, list) else model
    get_dump_lifecycle().on_step_start(model=model_obj)


def on_step_end(result, args):
    """Invoked at the bottom of each train_step. Advances dump + records loss."""
    if not getattr(args, "hcu_probe_config", None):
        return

    loss_dict = grad_norm = skipped_iter = None
    if isinstance(result, tuple) and len(result) >= 6:
        loss_dict = result[0]
        skipped_iter = result[1]
        grad_norm = result[5]

    lifecycle = get_dump_lifecycle()
    debugger = lifecycle.debugger
    step = getattr(debugger, "iteration", None) if debugger is not None else None

    lifecycle.on_step_end(
        step=step,
        loss_dict=loss_dict,
        grad_norm=grad_norm,
        skipped_iter=skipped_iter,
    )


def on_train_end():
    """Invoked in train() finally block. Flushes debugger and tracker."""
    get_dump_lifecycle().on_train_end()
