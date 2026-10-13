#!/bin/bash
# 单节点跑 base verl 0.5b(绕开 run.sh 的 mpirun/NET_TYPE/PORT,手动做等价步骤)
# 容器内执行:bash /data/hcu_probe_verl/run_0.5b.sh
set +e
VERL=/data/hcu_probe_verl/hcu_verl
ASSETS=/data/hcu_probe_verl/verl_assets
OUT=/data/hcu_probe_verl/verl_run
mkdir -p ${OUT}

echo "==================== [0/4] 清残留僵尸(上次崩的 vllm/ray 孤儿进程占显存,ray stop 杀不掉)===================="
ray stop --force 2>/dev/null || true
# 容器有独立 PID namespace,只杀本容器进程,不误伤同节点别的用户;这些 pattern 不含脚本自身路径,pkill 也排除自己
for pat in main_ppo EngineCore vllm_async WorkerDict "ray::" vLLMHttpServer; do pkill -9 -f "$pat" 2>/dev/null; done
sleep 6

echo "==================== [1/4] ray hycu 补丁(ray 默认认 hcugpu,海光是 hycu)===================="
ray stop --force 2>/dev/null || true
RAY_PY=$(pip show ray | grep Location | awk -F 'Location: ' '{print $2}')/ray/_private/thirdparty/pyamdsmi/pyamdsmi.py
if [[ -f "$RAY_PY" ]]; then
    sed -i "s|/sys/module/amdgpu/initstate|/sys/module/hycu/initstate|g" "$RAY_PY"
    sed -i "s|rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')|rocm_path = os.environ.get('HYHAL_PATH', '/opt/rocm')|g" "$RAY_PY"
    echo "[ray-patch] applied: $RAY_PY"
else
    echo "[ray-patch] WARN not found: $RAY_PY"
fi

echo "==================== [2/4] hcu_verl patch verl(cp patch_init.py)===================="
cp ${VERL}/hcu_verl/patch_init.py ${VERL}/verl/verl/__init__.py && echo "[hcu_verl-patch] OK"

echo "==================== [3/4] env ===================="
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export VERL_PATH=${VERL}
export PYTHONPATH=${VERL}:${VERL}/verl:${VERL}/Megatron-LM:${PYTHONPATH}
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libamdhip64.so
export HYHAL_PATH=${HYHAL_PATH:-/opt/hyhal}
export TENSORBOARD_DIR=${OUT}/tensorboard
# ★ HCU GPU 可见性(否则 das vllm 报 HIP vs CUDA 冲突)——抄自 hcu_verl examples/recipe/run_dapo_qwen3_30b_a3b_npu.sh
unset CUDA_VISIBLE_DEVICES || true
unset ROCR_VISIBLE_DEVICES || true
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
# B-debug: 解开被 ray dedup / vllm 吞掉的 WorkerProc 根因
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=DEBUG
# vllm worker HIP OOM 对策 —— hcu_slime das env + 关 sleep_mode 后 expandable 可用
export HSA_NO_SCRATCH_RECLAIM=1                            # ROCm 不回收 scratch
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # 治碎片(已关 sleep_mode,不再撞 memory pool 断言)
echo "PYTHONPATH=$PYTHONPATH"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES  NOSET_HIP=$RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"

echo "==================== [4/4] run 0.5b L4 端到端收敛(关 dump,正常大 batch 看 reward/ppo_kl)===================="
# L4 复用 dump 用的 run_qwen2_5_0.5b.sh(patch 02 改成了小 batch+单请求的 dump 配方),
# 用 hydra CLI override 覆盖回正常吞吐配置(大 batch + 正常 prefix caching),避免 max_num_seqs=1 串行卡。
# 注:不设 DUMP_ON/PROMPTS_ONLY/MONITOR_CONFIG(纯训练);override 优先级高于脚本内 +actor_rollout_ref.* 默认值。
cd ${VERL}/examples/grpo_trainer || exit 1
echo "localhost slots=8" > ./hostfile
bash run_qwen2_5_0.5b.sh \
    --data_path=${ASSETS}/data \
    --host_file=./hostfile \
    --hf_model_path=${ASSETS}/models \
    --mcore_model_path=${ASSETS}/models \
    --save_ckpt_path=${OUT} \
    --profiling= \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True
echo "==================== run_qwen2_5_0.5b.sh (L4) EXIT=$? ===================="
