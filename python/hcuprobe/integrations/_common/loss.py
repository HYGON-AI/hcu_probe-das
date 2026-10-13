"""Framework-agnostic per-step loss / grad_norm CSV logger.

A single module-level singleton holds the CSV file. Both the dump lifecycle
and the monitor lifecycle call `record(...)` on the same tracker — addressing
review R3 (one global, not two).

Deduplication: the tracker remembers (step, loss_key) pairs already written,
so simultaneous activation of both subsystems does not duplicate rows.

DP rank filtering: the default `is_writing_rank_fn` checks
`torch.distributed.get_rank() == 0` (framework-agnostic). Megatron-style
DP-group rank can be supplied via the `is_writing_rank_fn` injection.
"""

import csv
import os
import time


def _default_is_writing_rank():
    """Default DP-rank check using torch.distributed (framework-agnostic).

    Per-framework integrations may inject a more precise check
    (e.g. Megatron's data-parallel-group rank vs world rank).
    """
    try:
        import torch

        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


class LossTracker:
    def __init__(self, output_dir, *, is_writing_rank_fn=None):
        self._output_dir = output_dir
        self._csv_path = None
        self._file = None
        self._writer = None
        self._recorded = set()
        self._is_writing_rank_fn = is_writing_rank_fn or _default_is_writing_rank

    def _ensure_open(self):
        if self._writer is not None:
            return
        loss_dir = os.path.join(self._output_dir, "loss_tracking")
        os.makedirs(loss_dir, exist_ok=True)
        self._csv_path = os.path.join(loss_dir, "loss_log.csv")
        need_header = not os.path.exists(self._csv_path)
        self._file = open(self._csv_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if need_header:
            self._writer.writerow(
                ["step", "loss_key", "loss_value", "grad_norm", "skipped", "timestamp"]
            )

    def record(self, step, loss_dict, grad_norm, skipped_iter):
        if not loss_dict:
            return
        if not self._is_writing_rank_fn():
            return
        self._ensure_open()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        grad_norm_val = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
        for key, val in loss_dict.items():
            marker = (step, key)
            if marker in self._recorded:
                continue
            self._recorded.add(marker)
            loss_val = val.item() if hasattr(val, "item") else val
            self._writer.writerow(
                [step, key, f"{loss_val:.8f}", grad_norm_val, skipped_iter, ts]
            )
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


_tracker = None


def get_tracker(*, output_dir=None, is_writing_rank_fn=None):
    """Return the process-wide LossTracker, initializing it lazily.

    Returns None when `output_dir` is None and no tracker has been created.
    Once initialized, subsequent calls ignore further arguments.
    """
    global _tracker
    if _tracker is not None:
        return _tracker
    if not output_dir:
        return None
    _tracker = LossTracker(output_dir, is_writing_rank_fn=is_writing_rank_fn)
    return _tracker


def get_existing_tracker():
    """Return the existing tracker without creating one."""
    return _tracker


def close_tracker():
    global _tracker
    if _tracker is not None:
        _tracker.close()
        _tracker = None
