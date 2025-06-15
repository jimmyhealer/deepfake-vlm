import argparse
import logging
import sys
import os
from unsloth import FastVisionModel
from tqdm import tqdm
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DeepfakeVideoDataset
from utils.helpers import setup_logging


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen-VL model for Deepfake Detection."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/qwen2.5-vl-lora-ft",
        help="Path to the trained model.",
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
        "--split", type=str, default="test", help="Dataset split to evaluate on."
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Scale factor for resizing images."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/qwen_eval.log",
        help="Path to save the log file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.txt",
        help="File to save predictions.",
    )
    return parser.parse_args()


def main(args):
    setup_logging(args.log_path)
    logging.info("Starting Qwen-VL evaluation...")
    logging.info(f"Arguments: {args}")

    logging.info(f"Loading model from: {args.model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=args.load_in_4bit
    )

    transform = transforms.Compose(
        [
            transforms.Resize((1280 // args.scale, 720 // args.scale)),
        ]
    )

    logging.info(f"Loading dataset from: {args.root_dir}, split: {args.split}")
    test_dataset = DeepfakeVideoDataset(
        root_dir=args.root_dir, split=args.split, transform=transform
    )
    logging.info(f"Loaded {len(test_dataset)} samples from the test dataset.")

    instruction = "Please classify the following image as real or fake."

    def convert_to_conversation(sample):
        image = sample["image"]
        label = sample["label"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image"},
                ],
            },
        ]

        return {"messages": conversation, "image": image, "label": label}

    logging.info("Converting dataset to conversation format...")
    converted_dataset = []
    for i in tqdm(range(len(test_dataset))):
        converted_dataset.append(convert_to_conversation(test_dataset[i]))

    predictions = []
    acc = 0
    logging.info("Generating predictions...")
    for i in tqdm(converted_dataset):
        input_text = tokenizer.apply_chat_template(
            i["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            i["image"],
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            repetition_penalty=1.05,
        )

        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(prediction)
        prediction_text = prediction.split("assistant")[-1].strip()
        label = "real" if i["label"] else "fake"
        if label in prediction_text:
            acc += 1

    accuracy = acc / len(predictions) if predictions else 0
    logging.info(f"Final Accuracy: {accuracy:.4f}")

    logging.info(f"Saving predictions to {args.output_file}")
    with open(args.output_file, "w") as f:
        for i in predictions:
            f.write(i + "\n")
    logging.info("Predictions saved.")


if __name__ == "__main__":
    args = get_args()
    main(args)
