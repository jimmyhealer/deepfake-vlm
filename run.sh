#! /bin/bash

# --- Training ---
python train.py \
  --base_model "openai/clip-vit-base-patch32" \
  --model "clip_lora" \
  --root_dir "datasets" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --lora_target_modules "q_proj,v_proj" \
  --epochs 1 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --output_dir "results/clip_lora" \
  --log_path "logs/training_clip_lora.log" \
  --device "cuda" \
  --random_seed 42

# --- Evaluation ---
python evaluate.py \
  --model_dir "results/clip_lora" \
  --model "clip_lora" \
  --root_dir "datasets" \
  --log_path "logs/evaluation_clip_lora.log" \
  --split "test" \
  --roc_curve_path "results/roc_curve_clip_lora.png" \
  --device "cuda" \
  --random_seed 42