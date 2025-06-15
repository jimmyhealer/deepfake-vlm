import argparse
import logging
import os
import sys
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DeepfakeVideoDataset
from utils.helpers import setup_logging


def get_args():
    parser = argparse.ArgumentParser(description="Inference with Qwen-VL model.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct-awq",
        help="Model ID for VLLM.",
    )
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        help="Tokenizer ID from HuggingFace.",
    )
    parser.add_argument(
        "--quantization", type=str, default="awq", help="Quantization method."
    )
    parser.add_argument("--dtype", type=str, default="float16", help="Data type.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="datasets",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use."
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Scale factor for resizing images."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature."
    )
    parser.add_argument("--top_p", type=float, default=0.001, help="Sampling top_p.")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Sampling repetition penalty.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum new tokens to generate."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="qwen_zero_shot.txt",
        help="File to save predictions.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/qwen_inference.log",
        help="Path to save the log file.",
    )
    return parser.parse_args()


def main(args):
    setup_logging(args.log_path)
    logging.info("Starting Qwen-VL inference...")
    logging.info(f"Arguments: {args}")

    logging.info(f"Loading LLM: {args.model_id}")
    llm = LLM(
        model=args.model_id,
        limit_mm_per_prompt={"image": 1, "video": 0},
        quantization=args.quantization,
        dtype=args.dtype,
        tokenizer_mode="auto",
    )

    logging.info(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    transform = transforms.Compose(
        [
            transforms.Resize((1280 // args.scale, 720 // args.scale)),
        ]
    )

    logging.info(f"Loading dataset from: {args.root_dir}, split: {args.split}")
    test_dataset = DeepfakeVideoDataset(
        root_dir=args.root_dir, split=args.split, transform=transform, max_length=None
    )
    logging.info(f"Loaded {len(test_dataset)} samples.")

    system_prompt = """
You are a world-class digital forensics expert specializing in the detection of AI-generated media, commonly known as Deepfakes. Your analysis must be rigorous, objective, and strictly based on visual evidence. You will focus on identifying specific artifacts and inconsistencies that distinguish synthetic media from authentic photographs.

For every image analysis request, you MUST respond ONLY with a well-formed XML structure. Do not include any introductory text, explanations, conversational filler, or the XML declaration (`<?xml ... ?>`) outside of the root `<analysis_report>` tag. The XML output must conform to the following schema:

<analysis_report>
    <!-- Your final verdict: "AI-Generated" or "Authentic" -->
    <verdict>...</verdict>

    <!-- Your confidence in the verdict, from 0 to 100 -->
    <confidence_score>...</confidence_score>
    
    <!-- A container for all identified artifacts. Should be empty if none are found. -->
    <detected_artifacts>
        <!-- Repeat this block for each artifact found -->
        <artifact>
            <id>...</id>
            <description>...</description>
        </artifact>
    </detected_artifacts>
    
    <!-- A concise, one-sentence summary of your key findings -->
    <reasoning>...</reasoning>
</analysis_report>
"""
    instruction = """
As a digital forensics expert, analyze the provided image.

Using the artifact checklist below, identify all present forgery indicators. Then, formulate your response strictly in the XML format defined in your system instructions.

Artifact Checklist:
[1] Asymmetric eye irises
[2] Irregular glasses shape or lens reflections
[3] Irregular teeth shape or texture
[4] Irregular ears or earrings
[5] Strange or unnatural hair texture
[6] Inconsistent or overly smooth skin texture
[7] Inconsistent lighting and shadows
[8] Strange or distorted background
[9] Deformed or illogical hands
[10] Unnatural blurring or blending at object edges
"""

    def convert_to_conversation(sample):
        image = sample["image"]
        label = sample["label"]

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            },
        ]
        return {"messages": conversation, "image": image, "label": label}

    logging.info("Converting dataset to conversation format...")
    converted_dataset = [
        convert_to_conversation(test_dataset[i])
        for i in tqdm(range(len(test_dataset)), desc="Converting dataset")
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stop_token_ids=[],
    )

    logging.info("Generating prompts...")
    prompts = []
    for i in tqdm(converted_dataset, desc="Generating prompts"):
        image_inputs, _ = process_vision_info(i["messages"], return_video_kwargs=False)
        prompt = tokenizer.apply_chat_template(
            i["messages"], tokenize=False, add_generation_prompt=True
        )
        prompts.append({"prompt": prompt, "multi_modal_data": {"image": image_inputs}})

    logging.info("Generating results from LLM...")
    results = llm.generate(prompts, sampling_params=sampling_params)

    predictions = []
    acc = 0
    for result, sample in zip(results, converted_dataset):
        output = result.outputs[0].text
        predictions.append(output)
        label = "AI-Generated" if sample["label"] else "Authentic"

        prediction = output.split("<verdict>")[1].split("</verdict>")[0]
        if label == prediction:
            acc += 1

    accuracy = acc / len(results) if results else 0
    logging.info(f"Final Accuracy: {accuracy:.4f}")

    logging.info(f"Saving predictions to {args.output_file}")
    with open(args.output_file, "w") as f:
        for i in predictions:
            f.write(i + "\n")
    logging.info("Predictions saved.")


if __name__ == "__main__":
    args = get_args()
    main(args)
