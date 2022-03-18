#!/usr/bin/env bash

CHECKPOINT_PATH=checkpoints/finetune_vsa_heap
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH
cp checkpoints/pretrain/checkpoint_best.pt $CHECKPOINT_PATH/

TOTAL_UPDATES=90000   # Total number of training steps
TOTAL_EPOCHS=10       #
WARMUP_UPDATES=100    # Warmup the learning rate over this many updates
PEAK_LR=1e-5          # Peak learning rate, adjust as needed, official suggested: 1e-4
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
MAX_SENTENCES=64      # Number of sequences per batch (batch size)
ENCODER_LAYERS=8
NUM_CLASSES=2

CUDA_VISIBLE_DEVICES=2 python train.py \
  data-bin/finetune_vsa \
  --label heap \
  --num-classes $NUM_CLASSES \
  --task vsa --criterion vsa \
  --arch xdep --max-positions $TOKENS_PER_SAMPLE \
  --reset-optimizer --reset-dataloader --reset-meters \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
  --dropout 0.1 --attention-dropout 0.1 --pooler-dropout 0.1 --weight-decay 0.01 \
  --best-checkpoint-metric F1 --maximize-best-checkpoint-metric \
  --max-sentences $MAX_SENTENCES \
  --max-update $TOTAL_UPDATES --max-epoch $TOTAL_EPOCHS --log-format json --log-interval 10 \
  --no-epoch-checkpoints --save-dir $CHECKPOINT_PATH/ \
  --encoder-layers $ENCODER_LAYERS \
  --memory-efficient-fp16 --batch-size-valid 32 \
  --restore-file $CHECKPOINT_PATH/checkpoint_best.pt \
  --input-combine fuse --fuse-layer 1 --beta-shift 1e-1 --fuse-dropout 1e-1 \
  --byte-combine cnn --num-workers 4 --skip-invalid-size-inputs-valid-test |
  tee result/finetune_vsa_heap
