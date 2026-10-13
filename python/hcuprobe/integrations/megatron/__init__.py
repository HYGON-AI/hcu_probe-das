"""
Megatron-LM integration for hcu_probe.

Two independent subsystems, each activated by its own CLI flag:

    hcu_probe.integrations.megatron.dump     — L0/L1 tensor dump (PrecisionDebugger)
                                                  Activated by --hcu-probe-config
    hcu_probe.integrations.megatron.monitor  — lightweight training-state monitor
                                                  (TrainerMonitorV2)
                                                  Activated by --hcu-probe-monitor-config

Both share:
    hcu_probe.integrations.megatron.loss     — per-step loss/grad_norm CSV logger
                                                  (deduplicated across subsystems)
    hcu_probe.integrations.megatron.config   — shared JSON parsing helpers

Typical framework usage (from hcu_megatron adaptor wrappers):

    from hcu_probe.integrations.megatron import dump, monitor
    dump.on_train_start(args)
    dump.on_step_start(model, args)
    result = original_train_step(...)
    dump.on_step_end(result, args)
    dump.on_train_end()
"""

from . import config, dump, loss, monitor

__all__ = ["config", "dump", "loss", "monitor"]
