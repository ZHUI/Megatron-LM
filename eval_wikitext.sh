TASK="WIKITEXT103"

VALID_DATA='/ssd1/zhonghui03/models/PaddleNLP/examples/language_model/gpt2/eval_data/medium-345M.test.tokens'
#VALID_DATA='/ssd1/zhonghui03/models/PaddleNLP/examples/language_model/gpt2/eval_data/medium-345M.test.tokens'
#VALID_DATA='/ssd1/zhonghui03/models/PaddleNLP/examples/language_model/gpt2/eval_data/wikitext-103/wiki.test.tokens'
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
#CHECKPOINT_PATH=/ssd1/zhonghui03/models/PaddleNLP/examples/language_model/gpt2/
#CHECKPOINT_PATH=/ssd1/zhonghui03/GPT2ModelCheckpoints
CHECKPOINT_PATH=megatron_0010000

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --checkpoint-activations \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng

