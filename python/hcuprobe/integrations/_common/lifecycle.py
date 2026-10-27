"""Framework-agnostic dump / monitor lifecycle classes.

These classes wrap the upstream PrecisionDebugger and TrainerMonitorV2 with
keyword-only APIs that do not depend on any specific training framework.
Per-framework `integrations/<framework>/` namespaces translate framework-
specific argument shapes (e.g. Megatron `args` namespace, HuggingFace
`TrainingArguments`) into the keyword-only calls below.

Hook lifecycle:

  Dump (--hcu-probe-config):
    on_train_start(...)         seed + debugger init, optional tracker open
    on_step_start(model)        debugger.start(model)
    on_step_end(...)            debugger.step() + loss record
    on_train_end()              debugger.stop() + tracker close

  Monitor V2 (--hcu-probe-monitor-config):
    on_setup_done(...)          monitor init + .start(model, optimizer, grad_acc_steps)
    on_step_end(...)            mon.step() + loss record
    on_train_end()              monitor.stop() + tracker close

A process-wide singleton getter is exposed at the bottom so per-framework
namespaces share one instance:

    from hcuprobe.integrations._common.lifecycle import get_dump_lifecycle, get_monitor_lifecycle
"""

from . import config as _cfg
from . import loss as _loss


# ---------------------------------------------------------------------------
#  Logging helper (framework-agnostic; degrades to print)
# ---------------------------------------------------------------------------

def _log(component, msg):
    """Best-effort rank-0 logger.

    Tries Megatron's `print_rank_0` first (if Megatron is loaded in this
    process). Falls back to plain print otherwise.
    """
    line = f"[hcu_probe.{component}] {msg}"
    try:
        from megatron.training.utils import print_rank_0
        print_rank_0(line)
    except Exception:
        print(line)


# ---------------------------------------------------------------------------
#  Dump lifecycle
# ---------------------------------------------------------------------------

class DumpLifecycle:
    """Encapsulates PrecisionDebugger lifecycle for tensor dump.

    Stateful: holds a single PrecisionDebugger reference. Safe to call hooks
    when dump is disabled (becomes no-op).
    """

    def __init__(self):
        self._debugger = None

    def on_train_start(
        self,
        *,
        dump_cfg_path,
        seed=None,
        reproduce=False,
        loss_output_dir=None,
        is_writing_rank_fn=None,
    ):
        if not dump_cfg_path:
            return

        if seed is not None:
            try:
                from hcuprobe.pytorch import seed_all
                seed_all(seed=seed, mode=reproduce, rm_dropout=reproduce)
                _log("dump", f"seed fixed to {seed} (mode={reproduce}, rm_dropout={reproduce})")
            except Exception as e:
                _log("dump", f"seed_all() failed: {e}")

        if self._debugger is None:
            try:
                from hcuprobe.pytorch import PrecisionDebugger
                self._debugger = PrecisionDebugger(config_path=dump_cfg_path)
                _log("dump", f"PrecisionDebugger initialized with {dump_cfg_path}")
            except Exception as e:
                _log("dump", f"PrecisionDebugger init failed: {e}")
                self._debugger = None

        # Pre-open the loss tracker so the first record() does not race with
        # mkdir on a parallel filesystem.
        if loss_output_dir:
            _loss.get_tracker(output_dir=loss_output_dir, is_writing_rank_fn=is_writing_rank_fn)

    def on_step_start(self, *, model):
        if self._debugger is None:
            return
        try:
            from hcuprobe.pytorch import PrecisionDebugger
            PrecisionDebugger.start(model=model)
        except Exception as e:
            _log("dump", f"start() failed: {e}")

    def on_step_end(
        self,
        *,
        step=None,
        loss_dict=None,
        grad_norm=None,
        skipped_iter=None,
    ):
        if self._debugger is not None:
            try:
                from hcuprobe.pytorch import PrecisionDebugger
                PrecisionDebugger.step()
            except Exception as e:
                _log("dump", f"step() failed: {e}")

        tracker = _loss.get_existing_tracker()
        if tracker is not None and loss_dict:
            effective_step = step if step is not None else _fallback_step(tracker)
            tracker.record(effective_step, loss_dict, grad_norm, skipped_iter)

    def on_train_end(self):
        if self._debugger is not None:
            try:
                from hcuprobe.pytorch import PrecisionDebugger
                PrecisionDebugger.stop()
                _log("dump", "PrecisionDebugger stopped.")
            except Exception as e:
                _log("dump", f"stop() failed: {e}")
            finally:
                self._debugger = None
        _loss.close_tracker()

    @property
    def debugger(self):
        return self._debugger


# ---------------------------------------------------------------------------
#  Monitor V2 lifecycle
# ---------------------------------------------------------------------------

class MonitorLifecycle:
    """Encapsulates TrainerMonitorV2 lifecycle.

    Important invariant: the monitor JSON config MUST set
    `"patch_optimizer_step": false`. Megatron's nested optimizer chain defeats
    upstream auto-patching, so we advance `mon.step()` manually from
    `on_step_end`. trl / lightning callbacks should follow the same convention.
    """

    def __init__(self):
        self._monitor = None
        self._counter = 0

    def on_setup_done(
        self,
        *,
        models,
        optimizer,
        monitor_cfg_path,
        grad_acc_steps=1,
        loss_output_dir=None,
        is_writing_rank_fn=None,
    ):
        if not monitor_cfg_path:
            return
        try:
            import os
            # P4:TrainerMonitorV2 不会递归创建 output_dir,缺目录会静默 init 失败、无 csv;这里预先建好。
            _mon_out = _cfg.get_monitor_output_dir_from_cfg(monitor_cfg_path)
            if _mon_out:
                os.makedirs(_mon_out, exist_ok=True)
            from hcuprobe.core.monitor_v2.trainer import TrainerMonitorV2
            self._monitor = TrainerMonitorV2(monitor_cfg_path, fr="pytorch")
            model_obj = models[0] if isinstance(models, (list, tuple)) else models
            self._monitor.start(
                model=model_obj, optimizer=optimizer, grad_acc_steps=grad_acc_steps
            )
            _log("monitor", f"TrainerMonitorV2 initialized and started (grad_acc_steps={grad_acc_steps}).")
        except Exception as e:
            _log("monitor", f"init/start failed: {e}")
            self._monitor = None

        if loss_output_dir:
            _loss.get_tracker(output_dir=loss_output_dir, is_writing_rank_fn=is_writing_rank_fn)

    def on_step_end(
        self,
        *,
        step=None,
        loss_dict=None,
        grad_norm=None,
        skipped_iter=None,
    ):
        if self._monitor is None:
            return
        try:
            self._monitor.step()
        except Exception as e:
            _log("monitor", f"step() failed: {e}")

        tracker = _loss.get_existing_tracker()
        if tracker is not None and loss_dict:
            effective_step = step if step is not None else self._next_local_step()
            tracker.record(effective_step, loss_dict, grad_norm, skipped_iter)

    def on_train_end(self):
        if self._monitor is not None:
            try:
                self._monitor.stop()
                _log("monitor", "TrainerMonitorV2 stopped.")
            except Exception as e:
                _log("monitor", f"stop() failed: {e}")
            finally:
                self._monitor = None
        _loss.close_tracker()

    def _next_local_step(self):
        n = self._counter
        self._counter += 1
        return n

    @property
    def monitor(self):
        return self._monitor


# ---------------------------------------------------------------------------
#  Process-wide singletons (so dump and monitor can be coordinated cross-namespace)
# ---------------------------------------------------------------------------

_dump_lifecycle = None
_monitor_lifecycle = None


def get_dump_lifecycle():
    global _dump_lifecycle
    if _dump_lifecycle is None:
        _dump_lifecycle = DumpLifecycle()
    return _dump_lifecycle


def get_monitor_lifecycle():
    global _monitor_lifecycle
    if _monitor_lifecycle is None:
        _monitor_lifecycle = MonitorLifecycle()
    return _monitor_lifecycle


def reset_lifecycles():
    """Test-only: reset module-level state so tests can re-init."""
    global _dump_lifecycle, _monitor_lifecycle
    _dump_lifecycle = None
    _monitor_lifecycle = None
    _loss.close_tracker()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _fallback_step(tracker):
    """Derive a monotonic step counter from already-recorded rows
    when the upstream debugger does not expose one."""
    return len({s for (s, _) in tracker._recorded})  # noqa: SLF001


# Re-export key utilities
from .config import (  # noqa: E402, F401
    resolve_loss_output_dir,
    get_dump_output_dir_from_cfg,
    get_monitor_output_dir_from_cfg,
)
