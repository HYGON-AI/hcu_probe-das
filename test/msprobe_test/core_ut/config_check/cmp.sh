MASTER_PORT=6001
NNODES=1
NODE_RANK=0
CKPT_SAVE_DIR="./aaa"
DATA_PATH="./aaa"
TOKENIZER_MODEL="./aaa"
CKPT_LOAD_DIR="./aaa"
TP=2

DISTRIBUTED_ARGS="
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --sequence-parallel \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    --distributed-backend nccl \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    | tee logs/train_llama2_7b.log