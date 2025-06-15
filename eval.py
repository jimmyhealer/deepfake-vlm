import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DeepfakeVideoDataset
from utils.helpers import setup_logging, set_seed
from utils.args import get_base_parser
from utils.model_utils import select_model


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


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader and returns raw scores, labels, and predictions grouped by video.
    """
    model.eval()
    all_labels, all_scores, all_video_ids = [], [], []
    video_data = {}

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating on test set"):
            if (
                not data
                or "image" not in data
                or "label" not in data
                or "video_id" not in data
            ):
                logging.warning("Skipping batch due to missing data keys.")
                continue

            images, labels_tensor, video_ids = (
                data["image"],
                data["label"],
                data["video_id"],
            )

            images = model.preprocess(images)
            outputs = model(images)
            logging.debug(f"outputs: {outputs[:5]}, labels: {labels_tensor[:5]}")
            # The model might return 2 outputs for binary classification
            if outputs.ndim > 1 and outputs.shape[1] == 2:
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
            else:
                scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            labels = labels_tensor.cpu().numpy()

            all_labels.extend(labels)
            all_scores.extend(scores)
            all_video_ids.extend(video_ids)

            for i in range(len(video_ids)):
                video_id = video_ids[i]
                if video_id not in video_data:
                    video_data[video_id] = {"scores": [], "labels": []}
                video_data[video_id]["scores"].append(scores[i])
                video_data[video_id]["labels"].append(labels[i])
    return all_labels, all_scores, all_video_ids, video_data


def calculate_eer(y_true, y_score):
    """
    Calculates the Equal Error Rate (EER).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    return fpr[eer_index]


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """
    Plots and saves the ROC curve.
    """
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (area = {roc_auc:0.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    # Ensure directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(save_path)
    logging.info(f"ROC curve saved to {save_path}")
    plt.close()


def calculate_and_log_metrics(y_true, y_scores, level, roc_curve_path=None):
    """
    Calculates and logs metrics (Accuracy, F1, AUC, EER) and plots ROC curve.
    """
    if not y_true or not y_scores:
        logging.warning(f"No data to calculate {level}-level metrics.")
        return {}

    logging.info(f"--- {level}-level Metrics ---")
    y_pred = (np.array(y_scores) > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Accuracy: {accuracy * 100:.2f} %")

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    logging.info(f"F1 Score: {f1:.4f}")

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        logging.info(f"AUC: {roc_auc:.4f}")
    except ValueError:
        roc_auc = 0.0
        logging.warning(
            f"Could not calculate AUC for {level}-level, only one class present."
        )
        # Create dummy fpr, tpr for plotting if needed
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    try:
        eer = calculate_eer(y_true, y_scores)
        logging.info(f"EER: {eer:.4f}")
    except ValueError:
        eer = 0.0
        logging.warning(f"Could not calculate EER for {level}-level.")

    if roc_curve_path:
        plot_roc_curve(fpr, tpr, roc_auc, roc_curve_path)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc": roc_auc,
        "eer": eer,
    }


def main(args):
    """
    Main function to evaluate a trained model.
    """
    setup_logging(args.log_path)
    set_seed(args.random_seed)

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available. Switching to CPU.")
        args.device = "cpu"
    logging.info(f"Using device: {args.device}")

    # Load model
    try:
        model = select_model(args)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Load test dataset
    logging.info(f"Loading test dataset from {args.root_dir}")
    try:
        test_dataset = DeepfakeVideoDataset(
            root_dir=args.root_dir,
            split=args.split,
            transform=model.eval_transform,
            max_length=args.max_length,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        logging.info(f"Test dataset loaded with {len(test_dataset)} samples.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Evaluate the model
    frame_labels, frame_scores, frame_video_ids, video_data = evaluate_model(
        model, test_loader, args.device
    )

    all_metrics = {}

    # Frame-level data saving
    frame_df = pd.DataFrame(
        {"video_id": frame_video_ids, "score": frame_scores, "label": frame_labels}
    )
    frame_csv_path = os.path.join(args.model_dir, f"{args.split}_frame_scores.csv")
    frame_df.to_csv(frame_csv_path, index=False)
    logging.info(f"Frame-level scores saved to {frame_csv_path}")

    # Frame-level metrics
    frame_metrics = calculate_and_log_metrics(
        frame_labels, frame_scores, level="Frame", roc_curve_path=args.roc_curve_path
    )
    all_metrics["frame_level"] = frame_metrics

    # Video-level metrics
    video_labels, video_scores, video_ids_list = [], [], []
    if video_data:
        logging.debug("--- Video-level Predictions ---")
        for video_id, data in sorted(video_data.items()):
            if data["labels"]:
                avg_score = np.mean(data["scores"])
                prediction = "Fake" if avg_score > 0.5 else "Real"
                true_label = "Fake" if data["labels"][0] == 1 else "Real"

                logging.debug(
                    f"Video: {video_id}, Avg Score: {avg_score:.4f}, Prediction: {prediction}, True Label: {true_label}"
                )
                video_ids_list.append(video_id)
                video_labels.append(data["labels"][0])
                video_scores.append(avg_score)

    # Video-level data saving
    if video_ids_list:
        video_df = pd.DataFrame(
            {"video_id": video_ids_list, "score": video_scores, "label": video_labels}
        )
        video_csv_path = os.path.join(args.model_dir, f"{args.split}_video_scores.csv")
        video_df.to_csv(video_csv_path, index=False)
        logging.info(f"Video-level scores saved to {video_csv_path}")

    video_roc_path = None
    if args.roc_curve_path:
        base, ext = os.path.splitext(args.roc_curve_path)
        video_roc_path = f"{base}_video{ext}"

    video_metrics = calculate_and_log_metrics(
        video_labels, video_scores, level="Video", roc_curve_path=video_roc_path
    )
    all_metrics["video_level"] = video_metrics

    # Save metrics to json
    metrics_filename = f"{args.split}_metrics.json"
    metrics_path = os.path.join(args.model_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

    logging.info("Evaluation finished.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model for Deepfake Detection.",
        parents=[get_base_parser()],
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model artifacts.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "test_face2face"],
        help="Split to evaluate on.",
    )
    parser.add_argument(
        "--roc_curve_path",
        type=str,
        default="results/roc_curve.png",
        help="Path to save the ROC curve plot.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/evaluation.log",
        help="Path to save the log file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
