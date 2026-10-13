"""Megatron-side translator for the Monitor V2 lifecycle.

Reads Megatron `args` namespace + `models`/`optimizer` from
`setup_model_and_optimizer` and forwards keyword-only calls to the
framework-agnostic `_common.lifecycle.MonitorLifecycle`. All business logic
lives in `_common`; this file is just argument translation.

Important invariant (see CLAUDE.md / monitor v2 docstring):
    The monitor_v2 JSON config MUST set "patch_optimizer_step": false under
    Megatron. Megatron's nested DistributedOptimizer → ChainedOptimizer
    defeats upstream auto-patching, so we advance `mon.step()` manually
    from `on_step_end` instead.

Hook contract (called from hcu_megatron monitor_wrapper):
    on_setup_done(models, optimizer, args)
    on_step_end(result, args)
    on_train_end()
"""

from .._common.lifecycle import get_monitor_lifecycle
from .._common.config import resolve_loss_output_dir
from .config import compute_grad_acc_steps
from .dump import _is_megatron_dp_rank_0


def on_setup_done(models, optimizer, args):
    """Invoked after Megatron's setup_model_and_optimizer returns."""
    cfg_path = getattr(args, "hcu_probe_monitor_config", None)
    if not cfg_path:
        return
    get_monitor_lifecycle().on_setup_done(
        models=models,
        optimizer=optimizer,
        monitor_cfg_path=cfg_path,
        grad_acc_steps=compute_grad_acc_steps(args),
        loss_output_dir=resolve_loss_output_dir(
            dump_cfg_path=getattr(args, "hcu_probe_config", None),
            monitor_cfg_path=cfg_path,
        ),
        is_writing_rank_fn=_is_megatron_dp_rank_0,
    )


def on_step_end(result, args):
    """Invoked at the bottom of each train_step when monitor is active."""
    if not getattr(args, "hcu_probe_monitor_config", None):
        return

    loss_dict = grad_norm = skipped_iter = None
    if isinstance(result, tuple) and len(result) >= 6:
        loss_dict = result[0]
        skipped_iter = result[1]
        grad_norm = result[5]

    get_monitor_lifecycle().on_step_end(
        loss_dict=loss_dict,
        grad_norm=grad_norm,
        skipped_iter=skipped_iter,
    )


def on_train_end():
    """Invoked in train() finally block."""
    get_monitor_lifecycle().on_train_end()
