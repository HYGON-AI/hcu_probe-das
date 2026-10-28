#!/bin/bash
# 收官验证 run:对齐 megatron 通过配置(balance_batch=False, prefix_caching 默认 True)+ FSDP 专属修法 attn=eager。
# 小 batch(16/mini16/micro1/n5):① 跑完整 step 拿 loss/grad_norm(训练健康)② dump 量小不 OOM
# ③ 留够 num_requests=1 配对样本给 compare。不早停,让它跑 2~3 个 step。
set +e
ROOT=/your/work/dir          # ← 示例路径,改成你的工作目录
VERL=$ROOT/hcu_verl_v0.8.0
OUT=$ROOT/verl_run_fsdp_v08_clean
mkdir -p $OUT
exec > $OUT/train_clean.log 2>&1

ray stop --force 2>/dev/null || true
for pat in main_ppo EngineCore vllm_async WorkerDict vLLMHttp; do pkill -9 -f "$pat" 2>/dev/null; done
sleep 6

RAY_PY=$(pip show ray | grep Location | awk -F 'Location: ' '{print $2}')/ray/_private/thirdparty/pyamdsmi/pyamdsmi.py
if [[ -f "$RAY_PY" ]]; then
  sed -i "s|/sys/module/amdgpu/initstate|/sys/module/hycu/initstate|g" "$RAY_PY"
  sed -i "s|rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')|rocm_path = os.environ.get('HYHAL_PATH', '/opt/rocm')|g" "$RAY_PY"
fi
cp ${VERL}/hcu_verl/patch_init.py ${VERL}/verl/verl/__init__.py

printf '%s\n' '{"task":"statistics","dump_path":"'$OUT'/dump_actor","rank":[],"step":[],"level":"L0","async_dump":false,"statistics":{"scope":[],"list":[],"tensor_list":[],"data_mode":["all"],"summary_mode":"statistics"}}' > $OUT/cfg_actor.json
printf '%s\n' '{"task":"statistics","dump_path":"'$OUT'/dump_generate","rank":[],"step":[],"level":"L0","async_dump":false,"statistics":{"scope":[],"list":[],"tensor_list":[],"data_mode":["all"],"summary_mode":"statistics"}}' > $OUT/cfg_generate.json

export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error VERL_PATH=${VERL}
export PYTHONPATH=${VERL}:${VERL}/verl:${VERL}/Megatron-LM:${PYTHONPATH}
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libamdhip64.so HYHAL_PATH=/opt/hyhal
export TENSORBOARD_DIR=${OUT}/tensorboard
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 GPUS_PER_NODE=8
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
export HSA_NO_SCRATCH_RECLAIM=1 HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_FORCE_DEV_KERNARG=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DUMP_ON=1 PROMPTS_ONLY=1 DUMP_PHASE=log_prob
export DUMP_ACTOR_CONFIG=$OUT/cfg_actor.json
export DUMP_CONFIG_PATH=$OUT/cfg_generate.json

cd ${VERL}/examples/grpo_trainer || exit 1
echo "localhost slots=8" > ./hostfile

RQ=run_qwen2_5_0.5b.sh
RE=run_qwen2_5_0.5b_clean.sh
cp $RQ $RE
sed -i 's/^max_response_length=.*/max_response_length=$((256 * 1))/' $RE
# 末尾追加 override(hydra 后者胜,覆盖 config 数组里的默认)
sed -i '/${TRAINER_CONFIG\[@\]}/a\    data.train_batch_size=16 \\\n    actor_rollout_ref.actor.ppo_mini_batch_size=16 \\\n    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\\n    trainer.balance_batch=False \\\n    trainer.total_epochs=1 \\\n    +actor_rollout_ref.model.override_config.attn_implementation=eager' $RE

echo "=== 生效 override 检查 ==="
grep -nE "train_batch_size=16|ppo_micro_batch_size_per_gpu=1|balance_batch=False|attn_implementation=eager|^max_response_length=" $RE

bash $RE \
  --data_path=$ROOT/data \
  --host_file=./hostfile \
  --hf_model_path=$ROOT/verl_assets/models \
  --save_ckpt_path=${OUT} \
  --profiling=
echo RUN_EXIT=$?
