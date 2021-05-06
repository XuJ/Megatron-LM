#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=${HOME}/data/ShangjianTech_concat_txt/gpt2_test_text_document
CHECKPOINT_PATH=checkpoints/gpt2_distributed_with_mp_sentencepiece
TOKENIZER_PATH=${HOME}/data/bpe_3w_new/vocab.json
VOCAB_MODEL_FILE=${HOME}/data/bpe_3w_new/chinese_vocab.model

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       --num-layers 32 \
       --hidden-size 2560 \
       --num-attention-heads 16 \
       --micro-batch-size 32 \
       --global-batch-size 512 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100000 \
       --lr-decay-iters 100000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path "$DATA_PATH" \
       --vocab-file "$TOKENIZER_PATH" \
       --vocab-model-file "$VOCAB_MODEL_FILE" \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 1.5e-4 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 50 \
       --eval-interval 10 \
       --eval-iters 5 \
       --fp16
