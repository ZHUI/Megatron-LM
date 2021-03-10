#! /bin/bash

# Runs the "345M" parameter model
export CUDA_VISIBLE_DEVICES=2
export MASTER_PORT=6001
RANK=0
WORLD_SIZE=1

DATA_PATH=/ssd1/zhonghui03/Megatron-LM/data_dir/my-gpt2_text_document
#DATA_PATH=/home/qiujinxuan/Megatron-LM/data_dir/
CHECKPOINT_PATH=./checkpoints

# MAX= 200W/32=62500, 62500 * 0.09/3600=193h  8d
# SAVE= 4W/32=1250, 1250 * 0.09/3600=3.85h

MAX_STEPS=70000
SAVE_STEPS=2000
EVAL_STEPS=500


python pretrain_gpt.py\
       --num-layers 24 \
       --hidden-size 1024\
       --num-attention-heads 16\
       --micro-batch-size 32 \
       --global-batch-size 32 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 70000\
       --lr-decay-iters 320000\
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 2000 \
       --eval-interval 500 \
       --eval-iters 10 \
       --no-scaled-masked-softmax-fusion\
       --fp16

