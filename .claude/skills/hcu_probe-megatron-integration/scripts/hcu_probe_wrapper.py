"""
Thin adaptor shim: hcu_probe dump lifecycle wired into Megatron's train() and
train_step() via the adaptor patch system.

Actual implementation lives in the installed hcu_probe whl package at
`hcuprobe.integrations.megatron.dump`. This file is kept intentionally
small so that reading it reveals
the entire contract at a glance.
"""

from megatron.training import get_args
from megatron.training.utils import print_rank_0

from hcuprobe.integrations.megatron import dump


def hcu_probe_train_wrapper(original_train):
    print_rank_0("[hcu_probe-wrapper] train() wrapper applied via Feature mechanism.")

    def wrapped_train(*args, **kwargs):
        dump.on_train_start(get_args())
        try:
            return original_train(*args, **kwargs)
        finally:
            dump.on_train_end()

    return wrapped_train


def hcu_probe_train_step_wrapper(original_train_step):
    print_rank_0("[hcu_probe-wrapper] train_step() wrapper applied via Feature mechanism.")

    def wrapped_train_step(forward_step_func, data_iterator, model, *args, **kwargs):
        dump.on_step_start(model, get_args())
        result = original_train_step(forward_step_func, data_iterator, model, *args, **kwargs)
        dump.on_step_end(result, get_args())
        return result

    return wrapped_train_step