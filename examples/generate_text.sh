#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

INPUT_FILE=${HOME}/data/cpm_generate_test.txt
CHECKPOINT_PATH=checkpoints/gpt2_distributed_with_mp_sentencepiece
#CHECKPOINT_PATH=${HOME}/model/CPM-large_MP8_Megatron
VOCAB_FILE=${HOME}/data/bpe_3w_new/vocab.json
VOCAB_MODEL_FILE=${HOME}/data/bpe_3w_new/chinese_vocab.model

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       tools/generate_samples_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       --num-layers 32 \
       --hidden-size 2560 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --fp16 \
       --micro-batch-size 32 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file "$VOCAB_FILE" \
       --vocab-model-file "$VOCAB_MODEL_FILE" \
       --sample-input-file "$INPUT_FILE" \
       --num-samples 0 \
       --top_p 0.9 \
       --recompute
