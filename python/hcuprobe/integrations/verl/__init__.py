"""
verl integration for hcuprobe (HCU-maintained, not part of upstream msprobe).

    hcuprobe.integrations.verl.compare  — training/inference consistency compare
                                          (wraps upstream pt_compare without modifying core/)

Typical usage:

    from hcuprobe.integrations.verl import consistent_compare
    consistent_compare(train_dump_json, infer_dump_json, out_dir, backend="fsdp")   # or "megatron"
"""

from .compare import consistent_compare

__all__ = ["consistent_compare"]
