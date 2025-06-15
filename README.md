# Deepfake Detection with VLM

> [!NOTE]
> The assets include ROC Curve, result, and `report.pdf`.
> Thank you for your time.

## Overview
A deepfake detection system powered by Vision-Language Models (VLM). This project leverages pre-trained VLMs and parameter-efficient tuning methods to detect deepfakes across different manipulation techniques.

## Dataset

### FaceForensics++ (C40)
Download FaceForensics++ C40 [here](https://www.dropbox.com/t/2Amyu4D5TulaIofv). If the link is invalid, please contact Chih-Chung Hsu (National Yang Ming Chiao Tung University).

### Face2Face
Available on [Kaggle](https://www.kaggle.com/datasets/mdhadiuzzaman/face2face).

```bash
curl -L -o ~/Downloads/face2face.zip \
  https://www.kaggle.com/api/v1/datasets/download/mdhadiuzzaman/face2face
```

## Setup

Create and activate a virtual environment:

```bash
conda create -n deepfake-vlm python=3.11
conda activate deepfake-vlm
pip install -r requirements.txt
```

## Quick Start

Run training and evaluation:

```bash
bash run.sh
```

## Validate Dataset

This script validates the dataset, prints the number of videos in each split, and checks for identity leakage.

```bash
python dataset.py
```

## Training

You can train the model with the following command:

```bash
python train.py
  --base_model "openai/clip-vit-base-patch32" \
  --model "clip_lora" \ # "clip_linear" or "clip_lora" or "clip_vpt"
  --root_dir "datasets" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --lora_target_modules "q_proj,v_proj" \
  --prompt_size 120 \
```

You can select `clip_linear`, `clip_lora`, or `clip_vpt` as the model train strategy.

- `clip_linear`: Use the linear classifier on the CLIP features.
- `clip_lora`: Use the LoRA adapter on the CLIP features.
- `clip_vpt`: Use the visual prompt mask on the CLIP features.

### Qwen

You can train the Qwen model with the following command:

```bash
cd qwen
# Check your environment has unsloth
python train.py \
   --model_id "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit" \
   --load_in_4bit \
   --root_dir "../datasets" \
   --lora_r 16 \
   --lora_alpha 16 \
   --lora_dropout 0 \
   --use_rslora \
   --epochs 1 \
   --scale 4 \ # Scale factor for resizing images.
   --log_path "logs/training_qwen_lora.log" \
   --output_dir "results/qwen2.5-vl-lora-ft"
```

## Evaluation

You can evaluate the model with the following command:

```bash
# Check your environment has unsloth
python eval.py \
  --model_dir "results/clip_lora" \
  --model "clip_lora" \
  --root_dir "datasets" \
  --log_path "logs/evaluation_clip_lora.log" \
  --split "test" \
  --roc_curve_path "results/roc_curve_clip_lora.png" \
  --device "cuda" \
  --random_seed 42
```

### Qwen

You can evaluate the Qwen model with the following command:

```bash
cd qwen
python eval.py \
  --modemodel_pathl_dir "results/qwen2.5-vl-lora-ft" \
  --load_in_4bit \
  --root_dir "../datasets" \
  --split "test" \
  --scale 4 \ # Scale factor for resizing images.
  --log_path "logs/evaluation_qwen_lora.log" \
  --output_file "qwen_lora.txt"
```

## Qwen Zero-shot

You can test the Qwen Zero-shot with the following command:

```bash
cd qwen
python inference.py \
  --model_id "Qwen/Qwen2.5-VL-7B-Instruct-awq" \
  --tokenizer_id "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" \
  --quantization "awq" \
  --dtype "float16" \
  --root_dir "../datasets" \
  --split "test" \
  --scale 4 \ # Scale factor for resizing images.
  --temperature 0.1 \
  --top_p 0.001 \
  --repetition_penalty 1.05 \
  --max_tokens 1024 \
  --output_file "qwen_zero_shot.txt" \
  --log_path "logs/qwen_inference.log"
```

## Hyperparameter Search

You can search the hyperparameters with the following command:

```bash
python hyperparameter_search.py
```

## Results

| Model         | AUC    | EER    | ACC    | F1     | Runtime |
|---------------|--------|--------|--------|--------|---------|
| CLIP Baseline | 0.6100 | 0.4000 | 0.6273 | 0.7545 | 1 min   |
| CLIP + LoRA   | 0.8690 | **0.2000** | 0.8364 | 0.9043 | 2 min   |
| CLIP + VPT    | **0.8790** | **0.2000** | 0.8364 | 0.9032 | 1 min   |
| Qwen (zero-shot) | -      | -      | **0.9090** | **0.9523** | 28 min  |
| Qwen + QLoRA  | -      | -      | 0.8182 | 0.8980 | 1 hr    |

## Pretrained Weights

Download model weights from the [Releases](https://github.com/jimmyhealer/deepfake-vlm/releases/latest):
- CLIP Baseline
- CLIP + LoRA
- CLIP + VPT
- Qwen + QLoRA

---

For detailed implementation and results analysis, please refer to the full [project report](./assets/report.pdf).