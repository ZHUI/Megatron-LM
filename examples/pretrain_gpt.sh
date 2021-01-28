#! /bin/bash

# Runs the "345M" parameter model
export CUDA_VISIBLE_DEVICES=2
export MASTER_PORT=6001
RANK=0
WORLD_SIZE=1

DATA_PATH=/ssd1/zhonghui03/Megatron-LM/data_dir/my-gpt2_text_document
#DATA_PATH=/home/qiujinxuan/Megatron-LM/data_dir/
CHECKPOINT_PATH=./checkpoints


python pretrain_gpt.py\
       --num-layers 4 \
       --hidden-size 1024\
       --num-attention-heads 4\
       --micro-batch-size 8 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000\
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
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --no-scaled-masked-softmax-fusion
