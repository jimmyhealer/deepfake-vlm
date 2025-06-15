import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import logging
import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import DeepfakeVideoDataset
from utils.helpers import set_seed, setup_logging
from utils.args import get_base_parser
from utils.model_utils import select_model


def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    model.to(device)
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, data in enumerate(progress_bar):
        images, labels = (
            data["image"],
            data["label"].to(device),
        )

        images = model.preprocess(images)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{running_loss / (i + 1):.4f}"})

    return running_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating"):
            if data is None:
                continue
            images, labels = (
                data["image"],
                data["label"].to(device),
            )

            images = model.preprocess(images)
            outputs = model(images)
            scores = torch.sigmoid(outputs.data)
            predicted = (scores > 0.5).long()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    if not all_labels:
        logging.warning("No data was validated.")
        return {}

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.0
        logging.warning(
            "AUC score could not be calculated because only one class is present in the validation set."
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def collate_fn(batch):
    batch = [b for b in batch if b and b["image"] is not None]
    if not batch:
        return None

    # Handle cases where images are PIL Images and not Tensors
    if not isinstance(batch[0]["image"], torch.Tensor):
        images = [b["image"] for b in batch]
    else:
        images = torch.stack([b["image"] for b in batch])

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
    video_ids = [b["video_id"] for b in batch]
    return {"image": images, "label": labels, "video_id": video_ids}


def main(args):
    """
    Main function to train and validate the selected model.
    """
    # Setup logging and reproducibility
    setup_logging(args.log_path)
    set_seed(args.random_seed)

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available. Switching to CPU.")
        args.device = "cpu"
    logging.info(f"Using device: {args.device}")

    # Load model
    logging.info(f"Loading {args.model} model with base model: {args.base_model}")
    try:
        model = select_model(args)
        total_params, trainable_params, percent = model.print_trainable_parameters()
        logging.info(
            f"Total parameters: {total_params:,} || Trainable parameters: {trainable_params:,} ({percent:.2f}%)"
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Load datasets
    logging.info(f"Loading dataset from: {args.root_dir}")
    try:
        train_dataset = DeepfakeVideoDataset(
            root_dir=args.root_dir, split="train", transform=model.train_transform
        )
        val_dataset = DeepfakeVideoDataset(
            root_dir=args.root_dir, split="val", transform=model.eval_transform
        )
        logging.info(
            f"Dataset loaded with {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    logging.info(
        f"Optimizer: AdamW, LR: {args.lr}, Weight Decay: {args.weight_decay}, Epochs: {args.epochs}, Batch Size: {args.batch_size}"
    )

    # Training and validation loop
    logging.info("Starting training...")
    try:
        for epoch in range(args.epochs):
            avg_loss = train(
                model,
                train_loader,
                criterion,
                optimizer,
                args.device,
                epoch,
                args.epochs,
            )

            logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            metrics = validate(model, validation_loader, args.device)
            logging.info(
                f"Validation Accuracy: {metrics.get('accuracy', 0) * 100:.2f} %"
            )
            logging.info(f"Validation F1-Score: {metrics.get('f1', 0):.4f}")
            logging.info(f"Validation AUC: {metrics.get('auc', 0):.4f}")
            logging.info(f"Validation Precision: {metrics.get('precision', 0):.4f}")
            logging.info(f"Validation Recall: {metrics.get('recall', 0):.4f}")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

    # Save the model
    logging.info(f"Saving model and metrics to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    model.save(args.output_dir)
    # if args.model == "clip_lora":
    #     model.clip.vision_model.save_pretrained(os.path.join(args.output_dir, "lora_adapters"))
    #     torch.save(
    #         model.classifier.state_dict(),
    #         os.path.join(args.output_dir, "classifier.pth"),
    #     )
    # else:  # For clip_linear or other similar models
    #     torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))

    # Save metrics and config
    final_metrics = {
        "final_train_loss": avg_loss,
        "validation_metrics": metrics,
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    logging.info("Training finished successfully.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a model for Deepfake Detection.",
        parents=[get_base_parser()],
    )

    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA attention dimension (rank)."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, help="LoRA dropout probability."
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated target modules of the LoRA adapter.",
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/clip_linear",
        help="Directory to save the trained model and metrics.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/training.log",
        help="Path to save the log file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
