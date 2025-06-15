import argparse
import logging
import random
import sys
import os

from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from tqdm import tqdm
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DeepfakeVideoDataset
from utils.helpers import setup_logging, set_seed


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen-VL model for Deepfake Detection."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        help="Model ID from HuggingFace.",
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", default=True, help="Load model in 4bit."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../datasets",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA attention dimension (rank)."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0, help="LoRA dropout probability."
    )
    parser.add_argument(
        "--use_rslora",
        action="store_true",
        default=False,
        help="Use rank-stabilized LoRA.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=3407,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps.")
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate."
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps.")
    parser.add_argument(
        "--optim", type=str, default="adamw_8bit", help="Optimizer to use."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear", help="LR scheduler type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/qwen2.5-vl-lora-ft-output-2",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/qwen_train.log",
        help="Path to save the log file.",
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Scale factor for resizing images."
    )
    return parser.parse_args()


def main(args):
    setup_logging(args.log_path)
    set_seed(args.random_state)

    logging.info("Starting Qwen-VL training...")
    logging.info(f"Arguments: {args}")

    logging.info(f"Loading model: {args.model_id}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    logging.info("Setting up PEFT model...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((1280 // args.scale, 720 // args.scale)),
        ]
    )

    logging.info(f"Loading dataset from: {args.root_dir}")
    train_dataset = DeepfakeVideoDataset(
        root_dir=args.root_dir, split="train", transform=transform
    )
    logging.info(f"Loaded {len(train_dataset)} training samples.")

    INSTRUCTION = "Please classify the following image as real or fake."

    def convert_to_conversation(sample):
        image = sample["image"]
        label = sample["label"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"The class of this image is {'real' if label else 'fake'}.",
                    }
                ],
            },
        ]

        return {"messages": conversation}

    logging.info("Converting dataset to conversation format...")
    converted_dataset = []
    for i in tqdm(train_dataset):
        converted_dataset.append(convert_to_conversation(i))

    random.seed(args.random_state)
    random.shuffle(converted_dataset)

    FastVisionModel.for_training(model)

    logging.info("Starting training with SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.random_state,
            output_dir=args.output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )

    trainer.train()
    logging.info(f"Training complete. Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    logging.info("Model saved successfully.")


if __name__ == "__main__":
    args = get_args()
    main(args)
