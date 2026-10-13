#!/usr/bin/env python3
# WORKAROUND(容器内临时 patch,不进 image):
# das vllm 的 rocm.py `_sync_hip_cuda_env_vars()` 在 HIP(ray 给单卡)≠ CUDA(verl 设 0-7)时 raise。
# 改成"冲突时优先 HIP"——ray 给每个 rollout worker 分的单卡才是该 worker 要用的卡。
# 这样 vllm 的 model-inspect 子进程也不再崩。
# ⚠️ 此为先跑通 base verl 的权宜;正确的 env/config 修法待定。
f = "/usr/local/lib/python3.10/dist-packages/vllm/platforms/rocm.py"
s = open(f).read()
if "hcu-patch: prefer ray-assigned HIP" in s:
    print("ALREADY_PATCHED"); raise SystemExit(0)
old = (
    "    if hip_val is not None and cuda_val is not None:\n"
    "        if hip_val != cuda_val:\n"
    "            raise ValueError(\n"
    '                f"Inconsistent GPU visibility env vars: "\n'
    "                f\"HIP_VISIBLE_DEVICES='{hip_val}' vs \"\n"
    "                f\"CUDA_VISIBLE_DEVICES='{cuda_val}'. \"\n"
    '                f"Please set only one, or ensure they match."\n'
    "            )"
)
new = (
    "    if hip_val is not None and cuda_val is not None:\n"
    "        if hip_val != cuda_val:\n"
    '            os.environ["CUDA_VISIBLE_DEVICES"] = hip_val  # hcu-patch: prefer ray-assigned HIP'
)
if old not in s:
    print("ANCHOR_NOT_FOUND"); raise SystemExit(1)
open(f, "w").write(s.replace(old, new, 1))
print("PATCHED_ROCM")
