#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent external validation entry point for the M0_CLIN tabular
(XGBoost surgery-decision) classification model.

Loads an already-fitted FeatureEngineer and an already-trained model and
scores a caller-supplied external dataset, using the pipeline's own
cleaning, feature-transform, and prediction code. No fitting, training,
resampling, or re-splitting is performed.

Usage:
    python -m evaluation.text.external_validation \\
        --input-data <path.csv|path.xlsx|path.xls> \\
        --model <path to a trained model .pkl> \\
        --feature-engineer <path to a fitted feature_engineer .pkl> \\
        [--output-dir <path>] \\
        [--group-column <column name>] \\
        [--seed 42]
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader, _read_table
from text_pipeline.src.feature_engineering import FeatureEngineer
from text_pipeline.src.models import SurgeryClassifier

THIS_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = THIS_DIR.parent

POSITIVE_CLASS = 1
LABEL_NAMES_EN = [Config.LABEL_NAMES_EN[0], Config.LABEL_NAMES_EN[1]]


def build_arg_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluation.text.external_validation",
        description=(
            "Independent external validation for the M0_CLIN tabular "
            "classification model. Evaluates every valid record in "
            "--input-data using an already-fitted FeatureEngineer and an "
            "already-trained model. No fitting, training, resampling, or "
            "re-splitting of the external data is performed."
        ),
    )
    parser.add_argument(
        "--input-data", type=str, required=True,
        help="Path to an external data table (.csv, .xlsx, or .xls) "
             "containing the model's required feature columns and target column.",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to an already-trained model file, saved by SurgeryClassifier.save().",
    )
    parser.add_argument(
        "--feature-engineer", type=str, required=True,
        help="Path to an already-fitted FeatureEngineer file, saved by FeatureEngineer.save().",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write validation outputs to. Created only when "
             "the program runs (default: <evaluation package>/outputs/text_external_validation).",
    )
    parser.add_argument(
        "--group-column", type=str, default=None,
        help="Optional name of an --input-data column identifying which "
             "records belong to the same subject/cluster. If given, "
             "predictions.csv includes an anonymous 'group_id' column "
             "(sequential group_NNNNNN labels assigned in first-seen order; "
             "original values are never written out and no mapping file is "
             "produced). If omitted, predictions.csv has no group_id column.",
    )
    parser.add_argument(
        "--seed", type=int, default=Config.RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {Config.RANDOM_STATE}).",
    )
    return parser


def resolve_output_dir(output_dir_arg):
    """Resolve --output-dir, defaulting to <evaluation package>/outputs/text_external_validation."""
    if output_dir_arg is not None:
        return Path(output_dir_arg).resolve()
    return (EVALUATION_DIR / "outputs" / "text_external_validation").resolve()


def load_input_table(path):
    """Read the external input data table (.csv, .xlsx, or .xls)."""
    return _read_table(str(path))


# Alternate raw-label spellings accepted for the target column, on top of
# Config.LABEL_MAPPING's own keys. Built with chr() from a codepoint
# (never written as a literal character) so this file's source text
# contains no CJK bytes. Each alias maps to the exact raw label
# Config.LABEL_MAPPING already recognizes, derived from Config.LABEL_NAMES
# rather than a new literal.
_TARGET_LABEL_ALIASES = {
    chr(0x662F): Config.LABEL_NAMES[1],  # alias for the positive-class raw label
    chr(0x5426): Config.LABEL_NAMES[0],  # alias for the negative-class raw label
}


def _normalize_target_to_raw_label(series):
    """
    Normalize a target column to the exact raw label strings
    Config.LABEL_MAPPING expects, so it can be mapped to 0/1 exactly once
    (by DataLoader._clean_dataframe(), the single point where that mapping
    happens).

    Accepts Config.LABEL_MAPPING's own raw label strings, a recognized
    alternate spelling (_TARGET_LABEL_ALIASES), or already-encoded 0/1
    integers (Config.LABEL_NAMES keys). Returns None if the column's values
    are mixed or outside all of these sets.
    """
    aliased = series.replace(_TARGET_LABEL_ALIASES)

    raw_values = set(Config.LABEL_MAPPING.keys())
    encoded_values = set(Config.LABEL_NAMES.keys())
    unique_values = set(aliased.unique().tolist())

    if unique_values and unique_values <= raw_values:
        return aliased
    if unique_values and unique_values <= encoded_values:
        return aliased.map(Config.LABEL_NAMES)
    return None


def validate_and_prepare(df, group_column):
    """
    Validate the input table, apply the pipeline's own cleaning and feature
    transform, and return (clean_df, y_true, group_values). Raises
    ValueError with a specific, actionable message on any contract
    violation. Never silently drops or reorders records.
    """
    if df.empty:
        raise ValueError("Input data file contains no records.")

    required_columns = list(Config.FEATURE_COLUMNS) + [Config.TARGET_COLUMN]
    if group_column is not None:
        required_columns.append(group_column)
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Input data file is missing required column(s): {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    if group_column is not None:
        group_series = df[group_column]
        if group_series.isna().any():
            raise ValueError(f"Input data file has missing values in group column '{group_column}'.")
        if (group_series.astype(str).str.strip() == "").any():
            raise ValueError(f"Input data file has empty-string values in group column '{group_column}'.")
        group_values = group_series.tolist()
    else:
        group_values = None

    target_series = df[Config.TARGET_COLUMN]
    if target_series.isna().any():
        raise ValueError(f"Input data file has missing values in target column '{Config.TARGET_COLUMN}'.")

    normalized_target = _normalize_target_to_raw_label(target_series)
    if normalized_target is None:
        raise ValueError(
            f"Input data file target column '{Config.TARGET_COLUMN}' contains value(s) "
            f"outside the supported label set. Allowed raw labels: "
            f"{sorted(Config.LABEL_MAPPING.keys())}; allowed encoded values: "
            f"{sorted(Config.LABEL_NAMES.keys())}."
        )

    working_df = df[list(Config.FEATURE_COLUMNS) + [Config.TARGET_COLUMN]].copy()
    working_df[Config.TARGET_COLUMN] = normalized_target

    loader = DataLoader(config=Config())
    cleaned_df, _preserved = loader._clean_dataframe(
        working_df, preserve_name="external_validation", verbose=False
    )

    if len(cleaned_df) != len(df):
        raise ValueError(
            "Cleaning removed record(s) from the input data; refusing to "
            "silently drop records during external validation."
        )

    if cleaned_df[Config.NUMERICAL_FEATURES].isna().all().all():
        raise ValueError("All numerical feature values are invalid after cleaning.")

    final_target = cleaned_df[Config.TARGET_COLUMN]
    if final_target.isna().any():
        raise ValueError(
            f"Target column '{Config.TARGET_COLUMN}' contains missing values "
            f"after cleaning and label encoding."
        )
    unexpected_encoded = set(final_target.unique().tolist()) - {0, 1}
    if unexpected_encoded:
        raise ValueError(
            f"Target column '{Config.TARGET_COLUMN}' contains value(s) outside "
            f"{{0, 1}} after cleaning and label encoding: {sorted(unexpected_encoded)}."
        )

    return cleaned_df, group_values


def load_model_and_engineer(model_path, feature_engineer_path):
    """Load the trained model and fitted feature engineer, verifying they are fitted."""
    classifier = SurgeryClassifier.load(str(model_path))
    engineer = FeatureEngineer.load(str(feature_engineer_path))

    if not getattr(engineer, "is_fitted", False):
        raise ValueError(f"Loaded FeatureEngineer is not fitted: {feature_engineer_path}")
    if not getattr(classifier, "is_fitted", False):
        raise ValueError(f"Loaded model is not fitted: {model_path}")
    if not hasattr(classifier, "predict") or not hasattr(classifier, "predict_proba"):
        raise ValueError("Loaded model does not support predict() and predict_proba().")

    return classifier, engineer


def transform_and_predict(cleaned_df, classifier, engineer):
    """Transform-only (never fit) and predict-only (never fit). Returns (y_true, y_pred, y_proba)."""
    X, y = engineer.transform(cleaned_df, verbose=False)

    if list(X.columns) != list(classifier.feature_names):
        raise ValueError(
            f"FeatureEngineer output feature order {list(X.columns)} does not match "
            f"the loaded model's expected feature order {classifier.feature_names}."
        )

    y_pred = classifier.predict(X)
    proba = classifier.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(
            f"predict_proba() must return binary-class probabilities with shape "
            f"(n_samples, 2); got shape {proba.shape}."
        )

    y_proba = proba[:, POSITIVE_CLASS]
    if not np.isfinite(y_proba).all() or (y_proba < 0).any() or (y_proba > 1).any():
        raise ValueError("predict_proba() returned non-finite or out-of-range probability values.")

    if len(y_pred) != len(cleaned_df):
        raise ValueError("Number of predictions does not match the number of cleaned input records.")

    y_true = y.to_numpy(dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return y_true, y_pred, proba


def compute_metrics(y_true, y_pred, y_proba):
    """Compute the fixed metric set using the model's own predict()/predict_proba() output."""
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    unique_true = set(np.unique(y_true).tolist())
    if len(unique_true) < 2:
        roc_auc = None
        average_precision = None
    else:
        roc_auc = float(roc_auc_score(y_true, y_proba[:, POSITIVE_CLASS]))
        average_precision = float(average_precision_score(y_true, y_proba[:, POSITIVE_CLASS]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report_text = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES_EN, zero_division=0
    )

    metrics = {
        "n_samples": int(len(y_true)),
        "label_names": LABEL_NAMES_EN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "confusion_matrix": cm.tolist(),
    }
    return metrics, report_text, cm


def build_predictions_frame(y_true, y_pred, y_proba, group_values):
    """Build the predictions DataFrame containing only anonymous fields."""
    import pandas as pd

    data = {
        "sample_index": np.arange(len(y_true)),
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_class_0": y_proba[:, 0],
        "prob_class_1": y_proba[:, 1],
    }

    if group_values is not None:
        group_id_map = {}
        group_ids = []
        for raw_group in group_values:
            if raw_group not in group_id_map:
                group_id_map[raw_group] = f"group_{len(group_id_map) + 1:06d}"
            group_ids.append(group_id_map[raw_group])
        data["group_id"] = group_ids

    return pd.DataFrame(data)


def _sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(output_dir, metrics, report_text, cm, predictions_df,
                   input_file_path, model_file_path, feature_engineer_file_path, seed):
    """Write metrics.json, classification_report.txt, confusion_matrix.csv, and predictions.csv."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = dict(metrics)
    metrics["seed"] = seed
    metrics["input_file_sha256"] = _sha256_of_file(input_file_path)
    metrics["model_file_sha256"] = _sha256_of_file(model_file_path)
    metrics["feature_engineer_file_sha256"] = _sha256_of_file(feature_engineer_file_path)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write(report_text)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in LABEL_NAMES_EN],
        columns=[f"pred_{name}" for name in LABEL_NAMES_EN],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input_data).resolve()
    model_path = Path(args.model).resolve()
    feature_engineer_path = Path(args.feature_engineer).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"--input-data not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"--model not found: {model_path}")
    if not feature_engineer_path.is_file():
        raise FileNotFoundError(f"--feature-engineer not found: {feature_engineer_path}")

    df = load_input_table(input_path)
    cleaned_df, group_values = validate_and_prepare(df, args.group_column)

    classifier, engineer = load_model_and_engineer(model_path, feature_engineer_path)
    y_true, y_pred, y_proba = transform_and_predict(cleaned_df, classifier, engineer)

    metrics, report_text, cm = compute_metrics(y_true, y_pred, y_proba)
    predictions_df = build_predictions_frame(y_true, y_pred, y_proba, group_values)

    output_dir = resolve_output_dir(args.output_dir)
    write_outputs(
        output_dir, metrics, report_text, cm, predictions_df,
        input_path, model_path, feature_engineer_path, args.seed,
    )

    print(f"External validation complete. N={metrics['n_samples']}  Accuracy={metrics['accuracy']:.4f}")
    print(f"Outputs written to: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
