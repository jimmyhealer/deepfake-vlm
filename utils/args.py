import argparse


def get_base_parser():
    """
    Returns a base parser with arguments common to both training and evaluation.
    """
    parser = argparse.ArgumentParser(add_help=False)

    # Model and LoRA parameters
    parser.add_argument(
        "--model",
        type=str,
        default="clip_linear",
        choices=["clip_linear", "clip_lora", "clip_vpt"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Base CLIP model ID from HuggingFace or OpenAI.",
    )
    parser.add_argument(
        "--use_processor",
        action="store_true",
        help="Use the processor for the model.",
    )

    # Dataset and Directories
    parser.add_argument(
        "--root_dir",
        type=str,
        default="datasets",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequence.",
    )
    parser.add_argument(
        "--prompt_size",
        type=int,
        default=30,
        help="The size (height and width) of the visual prompt.",
    )

    # Common execution parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training (cuda or cpu).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser 