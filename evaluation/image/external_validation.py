#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent external validation entry point for the M0_IMG image
classification model.

Loads a trained checkpoint and evaluates it against an external labels
file plus a directory of images, using the pipeline's own model
architecture, configuration, and validation-time image transform. No
training, fitting, or data re-splitting is performed.

Usage:
    python -m evaluation.image.external_validation \\
        --labels-file <path.csv|path.xlsx> \\
        --image-dir <path> \\
        --checkpoint <path.pth> \\
        --image-column <column name> \\
        --label-column <column name> \\
        [--group-column <column name>] \\
        [--output-dir <path>] \\
        [--device auto|cpu|cuda] \\
        [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from image_pipeline.src.config import Config
from image_pipeline.src.data_utils import create_data_transforms
from image_pipeline.src.main import resolve_device, set_global_seed
from image_pipeline.src.models import create_model
from image_pipeline.src.train import load_model

THIS_DIR = Path(__file__).resolve().parent
NUM_CLASSES = len(Config.LABEL_MAPPING)
CLASS_INDICES = sorted(Config.LABEL_MAPPING.keys())
CLASS_DISPLAY_NAMES = [
    Config.ENGLISH_LABEL_MAPPING[Config.LABEL_MAPPING[i]] for i in CLASS_INDICES
]
LABEL_VALUE_TO_INDEX = {v: k for k, v in Config.LABEL_MAPPING.items()}


def build_arg_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluation.image.external_validation",
        description=(
            "Independent external validation for the M0_IMG image classification "
            "model. Evaluates every valid record in the labels file against the "
            "images in --image-dir, using an already-trained checkpoint. No "
            "training, fitting, or re-splitting of the external data is performed."
        ),
    )
    parser.add_argument(
        "--labels-file", type=str, required=True,
        help="Path to a labels table (.csv, .xlsx, or .xls) listing one row "
             "per image.",
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing the external images. Image-column values "
             "are resolved as paths relative to this directory.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a model checkpoint (.pth) saved by the M0_IMG training "
             "pipeline (a dict containing 'model_state_dict').",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write validation outputs to. Created only when "
             "the program runs (default: <this package>/outputs).",
    )
    parser.add_argument(
        "--image-column", type=str, required=True,
        help="Name of the labels-file column containing each image's path "
             "relative to --image-dir.",
    )
    parser.add_argument(
        "--label-column", type=str, required=True,
        help="Name of the labels-file column containing the ground-truth "
             "class label for each image.",
    )
    parser.add_argument(
        "--group-column", type=str, default=None,
        help="Optional name of a labels-file column identifying which "
             "records belong to the same subject/cluster (e.g. for later "
             "grouped resampling). If given, predictions.csv includes an "
             "anonymous 'group_id' column (sequential group_NNNNNN labels "
             "assigned in first-seen order; original values are never "
             "written out and no mapping file is produced). If omitted, "
             "predictions.csv has no group_id column at all.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Device selection: auto picks cuda if available else cpu; cpu "
             "forces CPU; cuda fails loudly if unavailable (default: auto).",
    )
    parser.add_argument(
        "--seed", type=int, default=Config.RANDOM_STATE,
        help="Random seed for reproducibility (default: the image pipeline's "
             "configured RANDOM_STATE).",
    )
    return parser


def resolve_output_dir(output_dir_arg):
    """Resolve --output-dir, defaulting to <this package>/outputs (never hardcoded)."""
    if output_dir_arg is not None:
        return Path(output_dir_arg).resolve()
    return (THIS_DIR / "outputs").resolve()


def load_labels_table(labels_path):
    """Read a CSV or Excel labels table into a DataFrame."""
    suffix = labels_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(labels_path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(labels_path)
    raise ValueError(
        f"Unsupported labels-file extension '{suffix}'. Supported extensions: "
        ".csv, .xlsx, .xls"
    )


def validate_and_build_records(df, image_dir, image_column, label_column, group_column):
    """
    Validate the labels table against the required contract and return a
    list of per-record dicts: {'image_path': Path, 'label': int, 'group': raw value or None}.

    Raises ValueError / FileNotFoundError with a specific, actionable message
    on any contract violation. Never silently drops or reorders records.
    """
    if df.empty:
        raise ValueError("Labels file contains no records.")

    required_columns = {image_column, label_column}
    if group_column is not None:
        required_columns.add(group_column)
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Labels file is missing required column(s): {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    duplicate_mask = df[image_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(set(df.loc[duplicate_mask, image_column].astype(str)))
        raise ValueError(
            f"Labels file contains duplicate values in image column "
            f"'{image_column}': {duplicates}"
        )

    missing_label_mask = df[label_column].isna()
    if missing_label_mask.any():
        rows = df.index[missing_label_mask].tolist()
        raise ValueError(
            f"Labels file has missing values in label column '{label_column}' "
            f"at row(s): {rows}"
        )

    invalid_labels = sorted(
        set(df[label_column].astype(str)) - set(map(str, LABEL_VALUE_TO_INDEX.keys()))
    )
    if invalid_labels:
        allowed = sorted(map(str, LABEL_VALUE_TO_INDEX.keys()))
        raise ValueError(
            f"Labels file contains unsupported label value(s) in column "
            f"'{label_column}': {invalid_labels}. Allowed values: {allowed}"
        )

    unsupported_ext = []
    for value in df[image_column]:
        ext = Path(str(value)).suffix.lower()
        if ext not in Config.SUPPORTED_EXTENSIONS:
            unsupported_ext.append(str(value))
    if unsupported_ext:
        raise ValueError(
            f"Labels file references image file(s) with an unsupported "
            f"extension: {unsupported_ext}. Supported extensions: "
            f"{Config.SUPPORTED_EXTENSIONS}"
        )

    missing_images = []
    resolved_paths = []
    for value in df[image_column]:
        candidate = image_dir / str(value)
        resolved_paths.append(candidate)
        if not candidate.is_file():
            missing_images.append(str(candidate))
    if missing_images:
        raise FileNotFoundError(
            f"Labels file references image file(s) not found under "
            f"--image-dir: {missing_images}"
        )

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        records.append({
            "image_path": resolved_paths[i],
            "label": LABEL_VALUE_TO_INDEX[row[label_column]],
            "group": row[group_column] if group_column is not None else None,
        })
    return records


def run_inference(model, device, records, val_transform):
    """Run forward-only inference over every record. Returns (labels, preds, probs)."""
    labels = np.empty(len(records), dtype=int)
    preds = np.empty(len(records), dtype=int)
    probs = np.empty((len(records), NUM_CLASSES), dtype=float)

    model.eval()
    with torch.no_grad():
        for i, record in enumerate(records):
            image = Image.open(record["image_path"]).convert("RGB")
            image_tensor = val_transform(image).unsqueeze(0).to(device)

            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(output, dim=1).item()

            labels[i] = record["label"]
            preds[i] = predicted
            probs[i] = probabilities[0].cpu().numpy()

    return labels, preds, probs


def compute_metrics(labels, preds):
    """Compute accuracy, macro/weighted precision/recall/F1, per-class metrics, and the confusion matrix."""
    accuracy = accuracy_score(labels, preds)
    report_dict = classification_report(
        labels, preds,
        labels=CLASS_INDICES,
        target_names=CLASS_DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        labels, preds,
        labels=CLASS_INDICES,
        target_names=CLASS_DISPLAY_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=CLASS_INDICES)

    metrics = {
        "n_samples": int(len(labels)),
        "class_names": CLASS_DISPLAY_NAMES,
        "accuracy": accuracy,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1_score": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }
            for name in CLASS_DISPLAY_NAMES
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics, report_text, cm


def build_predictions_frame(records, labels, preds, probs, group_column):
    """Build the predictions DataFrame containing only anonymous fields."""
    data = {
        "sample_index": np.arange(len(records)),
        "true_label": labels,
        "predicted_label": preds,
    }
    for class_idx in CLASS_INDICES:
        data[f"prob_class_{class_idx}"] = probs[:, class_idx]

    if group_column is not None:
        group_id_map = {}
        group_ids = []
        for record in records:
            raw_group = record["group"]
            if raw_group not in group_id_map:
                group_id_map[raw_group] = f"group_{len(group_id_map) + 1:06d}"
            group_ids.append(group_id_map[raw_group])
        data["group_id"] = group_ids

    return pd.DataFrame(data)


def write_outputs(output_dir, metrics, report_text, cm, predictions_df):
    """Write metrics.json, classification_report.txt, confusion_matrix.csv, and predictions.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write(report_text)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_DISPLAY_NAMES],
        columns=[f"pred_{name}" for name in CLASS_DISPLAY_NAMES],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    set_global_seed(args.seed)

    labels_path = Path(args.labels_file).resolve()
    image_dir = Path(args.image_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    if not labels_path.is_file():
        raise FileNotFoundError(f"--labels-file not found: {labels_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"--image-dir not found or not a directory: {image_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)

    df = load_labels_table(labels_path)
    records = validate_and_build_records(
        df, image_dir, args.image_column, args.label_column, args.group_column
    )

    model = create_model(num_classes=NUM_CLASSES, input_size=Config.IMAGE_SIZE, pretrained=False)
    model, _checkpoint = load_model(model, str(checkpoint_path), device)

    _, val_transform = create_data_transforms(Config.IMAGE_SIZE)

    labels, preds, probs = run_inference(model, device, records, val_transform)
    metrics, report_text, cm = compute_metrics(labels, preds)
    predictions_df = build_predictions_frame(records, labels, preds, probs, args.group_column)

    output_dir = resolve_output_dir(args.output_dir)
    write_outputs(output_dir, metrics, report_text, cm, predictions_df)

    print(f"External validation complete. N={metrics['n_samples']}  Accuracy={metrics['accuracy']:.4f}")
    print(f"Outputs written to: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
