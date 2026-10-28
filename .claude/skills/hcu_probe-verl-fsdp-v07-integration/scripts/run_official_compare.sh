#!/bin/bash
# ============================================================================
# 官方 pt_compare --backend fsdp 复核 verl 0.7 既有 dump(过滤版)
# ----------------------------------------------------------------------------
# 前提(容器内):
#   1. hcu_probe 含"修法B"= VerlShapeErrChecker + VERL_STATISTICS_CHECKERS
#      (FSDP 训练 [1,S,H] vs 推理 [S,H] 的 shape squeeze;含修法B 的 compare 容器已含)
#   2. Megatron-LM 在 PYTHONPATH(VerlShapeErrChecker 链路依赖,缺则 import 崩)
#   3. openpyxl 已装(pt_compare 落 xlsx)
#   4. 有 HCU 设备挂载(pt_compare import torch → dlopen flash_attn,缺设备 SIGABRT 134)
#
# 用法(容器内):
#   bash run_official_compare.sh <run目录> <VERL根>
#   例:bash run_official_compare.sh /data/hcu_probe_verl/dump_frozen \
#                                    /data/hcu_probe_verl/hcu_verl
#   <run目录> 下须有 dump_actor/  dump_generate/(0.7 冻结件即是)
#
# 产物:<run目录>_filtered/compare_out/compare_result_*.xlsx
# ============================================================================
set -e
RUN=${1:-/data/hcu_probe_verl/dump_frozen}
V=${2:-/data/hcu_probe_verl/hcu_verl}

export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
export PYTHONPATH=$V:$V/verl:$V/Megatron-LM:$PYTHONPATH
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libamdhip64.so HYHAL_PATH=${HYHAL_PATH:-/opt/hyhal}
export HSA_NO_SCRATCH_RECLAIM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 环境自检:hcu_probe 修法B / Megatron-LM / openpyxl / pt_compare ==="
DP=$(python3 -c "import hcuprobe,os;print(os.path.dirname(hcu_probe.__file__))")
grep -rlq VerlShapeErrChecker "$DP/core/compare/" && echo "  [ok] VerlShapeErrChecker(修法B)在" || echo "  [WARN] 修法B 缺失 → compare 会 shape error 配对≈0"
python3 -c "import megatron; import openpyxl; from hcuprobe.pytorch.compare.pt_compare import pt_compare; print('  [ok] megatron/openpyxl/pt_compare import OK')"

echo "=== 跑官方 compare(过滤 bench-only 交错模块 → pt_compare backend=fsdp consistent_check)==="
python3 "$(dirname "$0")/run07_filtered.py" "$RUN"