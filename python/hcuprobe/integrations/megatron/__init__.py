"""
Megatron-LM integration for hcu_probe.

Two independent subsystems, each activated by its own CLI flag:

    hcuprobe.integrations.megatron.dump     — L0/L1 tensor dump (PrecisionDebugger)
                                                  Activated by --hcu-probe-config
    hcuprobe.integrations.megatron.monitor  — lightweight training-state monitor
                                                  (TrainerMonitorV2)
                                                  Activated by --hcu-probe-monitor-config

Both share:
    hcuprobe.integrations.megatron.loss     — per-step loss/grad_norm CSV logger
                                                  (deduplicated across subsystems)
    hcuprobe.integrations.megatron.config   — shared JSON parsing helpers

Typical framework usage (from hcu_megatron adaptor wrappers):

    from hcuprobe.integrations.megatron import dump, monitor
    dump.on_train_start(args)
    dump.on_step_start(model, args)
    result = original_train_step(...)
    dump.on_step_end(result, args)
    dump.on_train_end()
"""

from . import config, dump, loss, monitor

__all__ = ["config", "dump", "loss", "monitor"]
