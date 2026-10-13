"""
Thin adaptor shim: Monitor V2 lifecycle wired into Megatron's
setup_model_and_optimizer(), train() and train_step() via the adaptor
patch system.

Actual implementation lives in the installed hcu_probe whl package at
`hcu_probe.integrations.megatron.monitor`. This file is kept intentionally
small so the full contract is visible in ~30 lines.
"""

from functools import wraps

from megatron.training import get_args
from megatron.training.utils import print_rank_0

from hcu_probe.integrations.megatron import monitor


def monitor_setup_model_and_optimizer_wrapper(original_func):
    print_rank_0("[monitor-wrapper] setup_model_and_optimizer wrapper applied.")

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        models, optimizer, opt_param_scheduler = original_func(*args, **kwargs)
        monitor.on_setup_done(models, optimizer, get_args())
        return models, optimizer, opt_param_scheduler

    return wrapper


def monitor_train_wrapper(original_train):
    print_rank_0("[monitor-wrapper] train() wrapper applied for monitor stop.")

    def wrapped_train(*args, **kwargs):
        try:
            return original_train(*args, **kwargs)
        finally:
            monitor.on_train_end()

    return wrapped_train


def monitor_train_step_wrapper(original_train_step):
    print_rank_0("[monitor-wrapper] train_step() wrapper applied for monitor step.")

    def wrapped_train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        monitor.on_step_end(result, get_args())
        return result

    return wrapped_train_step