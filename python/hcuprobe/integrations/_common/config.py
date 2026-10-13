"""Framework-agnostic JSON config parsing for hcu_probe.

Provides:
- read_json(path)              -- safe JSON loader, returns None on failure
- get_dump_output_dir_from_cfg(cfg_path)
- get_monitor_output_dir_from_cfg(cfg_path)
- resolve_loss_output_dir(*, dump_cfg_path=None, monitor_cfg_path=None)
"""

import json
import os


def read_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_dump_output_dir_from_cfg(cfg_path):
    cfg = read_json(cfg_path)
    if not cfg:
        return None
    return cfg.get("dump_path") or cfg.get("output_path") or cfg.get("output_dir")


def get_monitor_output_dir_from_cfg(cfg_path):
    cfg = read_json(cfg_path)
    if not cfg:
        return None
    return cfg.get("output_dir")


def resolve_loss_output_dir(*, dump_cfg_path=None, monitor_cfg_path=None):
    """Single source of truth for LossTracker output directory.

    Priority: dump cfg dump_path > monitor cfg output_dir.
    Returns None when neither is set.
    """
    if dump_cfg_path:
        d = get_dump_output_dir_from_cfg(dump_cfg_path)
        if d:
            return d
    if monitor_cfg_path:
        return get_monitor_output_dir_from_cfg(monitor_cfg_path)
    return None
