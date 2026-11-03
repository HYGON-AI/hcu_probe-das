#!/bin/bash
# 训练健康验证(eager):普通 FSDP 训练,【不带 dump、不带 PROMPTS_ONLY】+ attn=eager。
# 对照默认 flash(grad_norm 每步 NaN)。看 grad_norm/pg_loss/reward 是否有限且正常。
set +e
ROOT=/your/work/dir          # ← 示例路径,改成你的工作目录
VERL=$ROOT/hcu_verl_v0.8.0
OUT=$ROOT/verl_run_fsdp_v08_normal_eager
mkdir -p $OUT
exec > $OUT/train_normal_eager.log 2>&1

ray stop --force 2>/dev/null || true
for pat in main_ppo EngineCore vllm_async WorkerDict vLLMHttp; do pkill -9 -f "$pat" 2>/dev/null; done
sleep 6

RAY_PY=$(pip show ray | grep Location | awk -F 'Location: ' '{print $2}')/ray/_private/thirdparty/pyamdsmi/pyamdsmi.py
if [[ -f "$RAY_PY" ]]; then
  sed -i "s|/sys/module/amdgpu/initstate|/sys/module/hycu/initstate|g" "$RAY_PY"
  sed -i "s|rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')|rocm_path = os.environ.get('HYHAL_PATH', '/opt/rocm')|g" "$RAY_PY"
fi
cp ${VERL}/hcu_verl/patch_init.py ${VERL}/verl/verl/__init__.py

# 普通训练:不导出 DUMP_ON / PROMPTS_ONLY / DUMP_*_CONFIG
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error VERL_PATH=${VERL}
export PYTHONPATH=${VERL}:${VERL}/verl:${VERL}/Megatron-LM:${PYTHONPATH}
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libamdhip64.so HYHAL_PATH=/opt/hyhal
export TENSORBOARD_DIR=${OUT}/tensorboard
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 GPUS_PER_NODE=8
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
export HSA_NO_SCRATCH_RECLAIM=1 HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_FORCE_DEV_KERNARG=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${VERL}/examples/grpo_trainer || exit 1
echo "localhost slots=8" > ./hostfile

RQ=run_qwen2_5_0.5b.sh
RE=run_qwen2_5_0.5b_normal_eager.sh
cp $RQ $RE
sed -i 's/^max_response_length=.*/max_response_length=$((256 * 1))/' $RE
sed -i '/${TRAINER_CONFIG\[@\]}/a\    data.train_batch_size=16 \\\n    actor_rollout_ref.actor.ppo_mini_batch_size=16 \\\n    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\\n    trainer.balance_batch=False \\\n    trainer.total_epochs=1 \\\n    +actor_rollout_ref.model.override_config.attn_implementation=eager' $RE

echo "=== 生效 override(eager + 无 dump)==="
grep -nE "train_batch_size=16|ppo_micro_batch_size_per_gpu=1|balance_batch=False|attn_implementation=eager|^max_response_length=" $RE
echo "DUMP_ON=$DUMP_ON PROMPTS_ONLY=$PROMPTS_ONLY (应为空)"

bash $RE \
  --data_path=$ROOT/data \
  --host_file=./hostfile \
  --hf_model_path=$ROOT/verl_assets/models \
  --save_ckpt_path=${OUT} \
  --profiling=
echo RUN_EXIT=$?
