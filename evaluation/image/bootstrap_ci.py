#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic clustered/record-level bootstrap confidence-interval tool for
classification predictions.

Consumes a predictions CSV (such as the one produced by
evaluation.image.external_validation) and computes percentile bootstrap
confidence intervals for a fixed set of multiclass classification metrics.
Performs no inference, training, or fitting -- it only reads already-saved
predictions and probabilities.

Usage:
    python -m evaluation.image.bootstrap_ci \\
        --predictions-file <path to predictions.csv> \\
        [--output-dir <path>] \\
        [--n-bootstrap 10000] \\
        [--confidence-level 0.95] \\
        [--seed 20260909] \\
        [--group-column group_id] \\
        [--resampling-unit group|sample]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = THIS_DIR.parent

DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEED = 20260909
DEFAULT_GROUP_COLUMN = "group_id"

REQUIRED_BASE_COLUMNS = ["sample_index", "true_label", "predicted_label"]
PROB_COLUMN_PATTERN = re.compile(r"^prob_class_(\d+)$")
MIN_CLASSES = 3
PROBABILITY_SUM_TOLERANCE = 1e-2


def build_arg_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluation.image.bootstrap_ci",
        description=(
            "Compute percentile bootstrap confidence intervals for "
            "multiclass classification metrics from a predictions CSV. "
            "Reads only the given predictions file; performs no inference, "
            "training, or fitting."
        ),
    )
    parser.add_argument(
        "--predictions-file", type=str, required=True,
        help="Path to a predictions CSV containing sample_index, true_label, "
             "predicted_label, and prob_class_0.. probability columns "
             "(such as the output of evaluation.image.external_validation).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write bootstrap outputs to. Created only when "
             "the program runs (default: <evaluation package>/outputs/bootstrap_ci).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap resamples (default: {DEFAULT_N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL,
        help=f"Confidence level for the percentile interval, in (0, 1) "
             f"(default: {DEFAULT_CONFIDENCE_LEVEL}).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Seed for the NumPy random generator used for resampling "
             f"(default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--group-column", type=str, default=DEFAULT_GROUP_COLUMN,
        help=f"Name of the predictions-file column identifying which "
             f"records belong to the same resampling cluster. Only used "
             f"when --resampling-unit is 'group' (default: "
             f"'{DEFAULT_GROUP_COLUMN}').",
    )
    parser.add_argument(
        "--resampling-unit", choices=["group", "sample"], default="group",
        help="Resampling unit for the bootstrap. 'group' (default) draws "
             "whole clusters (identified by --group-column) with "
             "replacement, keeping every record of a drawn cluster "
             "together -- the formal default behavior. 'sample' draws "
             "individual records with replacement and must be requested "
             "explicitly; it is never selected automatically.",
    )
    return parser


def resolve_output_dir(output_dir_arg):
    """Resolve --output-dir, defaulting to <evaluation package>/outputs/bootstrap_ci."""
    if output_dir_arg is not None:
        return Path(output_dir_arg).resolve()
    return (EVALUATION_DIR / "outputs" / "bootstrap_ci").resolve()


def load_predictions_table(path):
    """Read the predictions CSV into a DataFrame."""
    return pd.read_csv(path)


def _detect_probability_columns(df):
    """Return the sorted list of class indices for which a prob_class_<i> column exists."""
    indices = []
    for column in df.columns:
        match = PROB_COLUMN_PATTERN.match(column)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def validate_predictions(df, group_column, resampling_unit):
    """
    Validate the predictions table against the required contract and return
    (records_df, num_classes). Raises ValueError with a specific, actionable
    message on any contract violation. Never silently drops or reorders rows.
    """
    if df.empty:
        raise ValueError("Predictions file contains no records.")

    required_columns = list(REQUIRED_BASE_COLUMNS)
    if resampling_unit == "group":
        required_columns.append(group_column)
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Predictions file is missing required column(s): {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    class_indices = _detect_probability_columns(df)
    if len(class_indices) < MIN_CLASSES or class_indices != list(range(len(class_indices))):
        raise ValueError(
            f"Predictions file must contain contiguous probability columns "
            f"prob_class_0 .. prob_class_{{K-1}} for at least {MIN_CLASSES} classes. "
            f"Found class indices: {class_indices}"
        )
    num_classes = len(class_indices)
    prob_columns = [f"prob_class_{i}" for i in class_indices]

    if df["sample_index"].isna().any():
        raise ValueError("Predictions file has missing values in column 'sample_index'.")
    if df["sample_index"].duplicated().any():
        duplicates = sorted(set(df.loc[df["sample_index"].duplicated(), "sample_index"]))
        raise ValueError(f"Predictions file has duplicate 'sample_index' values: {duplicates}")

    allowed_labels = set(class_indices)
    for column in ("true_label", "predicted_label"):
        if df[column].isna().any():
            raise ValueError(f"Predictions file has missing values in column '{column}'.")
        invalid = sorted(set(df[column].unique()) - allowed_labels)
        if invalid:
            raise ValueError(
                f"Predictions file column '{column}' contains value(s) outside "
                f"the allowed class range {sorted(allowed_labels)}: {invalid}"
            )

    prob_values = df[prob_columns].to_numpy(dtype=float)
    if not np.isfinite(prob_values).all():
        raise ValueError(
            f"Predictions file probability columns {prob_columns} contain "
            f"non-finite (NaN/Inf) values."
        )
    if (prob_values < 0).any() or (prob_values > 1).any():
        raise ValueError(
            f"Predictions file probability columns {prob_columns} contain "
            f"values outside the [0, 1] range."
        )
    row_sums = prob_values.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > PROBABILITY_SUM_TOLERANCE):
        raise ValueError(
            f"Predictions file rows exist where {prob_columns} do not sum to "
            f"1 within tolerance {PROBABILITY_SUM_TOLERANCE}."
        )

    if resampling_unit == "group":
        group_series = df[group_column]
        if group_series.isna().any():
            raise ValueError(f"Predictions file has missing values in group column '{group_column}'.")
        if (group_series.astype(str).str.strip() == "").any():
            raise ValueError(f"Predictions file has empty-string values in group column '{group_column}'.")

    return df.reset_index(drop=True), num_classes


def compute_point_metrics(y_true, y_pred, y_prob, num_classes):
    """Compute the fixed metric set for one true/predicted/probability array triple.

    Returns a dict of metric_name -> value, using NaN for any metric that is
    mathematically undefined for the given sample (e.g. a class with no
    positive or no negative examples for its one-vs-rest AUC/AP).
    """
    n = len(y_true)
    out = {"accuracy": float(np.mean(y_true == y_pred))}

    precision = np.full(num_classes, np.nan)
    recall = np.full(num_classes, np.nan)
    f1 = np.full(num_classes, np.nan)
    support = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support[c] = tp + fn
        if tp + fp > 0:
            precision[c] = tp / (tp + fp)
        if tp + fn > 0:
            recall[c] = tp / (tp + fn)
        if not np.isnan(precision[c]) and not np.isnan(recall[c]):
            f1[c] = 0.0 if (precision[c] + recall[c]) == 0 else \
                2 * precision[c] * recall[c] / (precision[c] + recall[c])

    for c in range(num_classes):
        out[f"precision_class_{c}"] = precision[c]
        out[f"recall_class_{c}"] = recall[c]
        out[f"f1_class_{c}"] = f1[c]

    out["precision_macro"] = np.nan if np.any(np.isnan(precision)) else float(np.mean(precision))
    out["recall_macro"] = np.nan if np.any(np.isnan(recall)) else float(np.mean(recall))
    out["f1_macro"] = np.nan if np.any(np.isnan(f1)) else float(np.mean(f1))

    present = support > 0
    total_support = int(support.sum())
    if total_support == 0:
        out["precision_weighted"] = np.nan
        out["recall_weighted"] = np.nan
        out["f1_weighted"] = np.nan
    else:
        out["precision_weighted"] = (
            np.nan if np.any(np.isnan(precision[present]))
            else float(np.sum(precision[present] * support[present]) / total_support)
        )
        out["recall_weighted"] = (
            np.nan if np.any(np.isnan(recall[present]))
            else float(np.sum(recall[present] * support[present]) / total_support)
        )
        out["f1_weighted"] = (
            np.nan if np.any(np.isnan(f1[present]))
            else float(np.sum(f1[present] * support[present]) / total_support)
        )

    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = n - n_pos
        if n_pos > 0 and n_neg > 0:
            out[f"auc_class_{c}"] = float(roc_auc_score(y_bin, y_prob[:, c]))
            out[f"ap_class_{c}"] = float(average_precision_score(y_bin, y_prob[:, c]))
        else:
            out[f"auc_class_{c}"] = np.nan
            out[f"ap_class_{c}"] = np.nan

    return out


def metric_order(num_classes):
    """Fixed, deterministic ordering of metric names for a given class count."""
    return (
        ["accuracy", "precision_macro", "recall_macro", "f1_macro",
         "precision_weighted", "recall_weighted", "f1_weighted"]
        + [f"{m}_class_{c}" for c in range(num_classes) for m in ("precision", "recall", "f1")]
        + [f"{m}_class_{c}" for c in range(num_classes) for m in ("auc", "ap")]
    )


def check_point_estimate_self_consistency(y_true, y_pred, point_metrics, num_classes):
    """
    Recompute the confusion matrix independently from y_true/y_pred and
    verify it is internally consistent with the point-estimate accuracy
    before any bootstrap resampling begins. Raises RuntimeError on failure.
    """
    n = len(y_true)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t in range(num_classes):
        for p in range(num_classes):
            confusion[t, p] = int(np.sum((y_true == t) & (y_pred == p)))

    if int(confusion.sum()) != n:
        raise RuntimeError(
            "Point-estimate consistency check failed: recomputed confusion "
            "matrix does not account for all input records."
        )

    recomputed_accuracy = float(np.trace(confusion)) / n if n > 0 else float("nan")
    if not np.isclose(recomputed_accuracy, point_metrics["accuracy"], atol=1e-9):
        raise RuntimeError(
            "Point-estimate consistency check failed: accuracy computed from "
            "the confusion matrix does not match the directly computed accuracy."
        )
    return confusion


def run_bootstrap(df, num_classes, n_bootstrap, seed, resampling_unit, group_column):
    """Run the percentile bootstrap and return {metric_name: np.ndarray of length n_bootstrap}."""
    y_true_all = df["true_label"].to_numpy(dtype=int)
    y_pred_all = df["predicted_label"].to_numpy(dtype=int)
    prob_columns = [f"prob_class_{c}" for c in range(num_classes)]
    y_prob_all = df[prob_columns].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    metrics = metric_order(num_classes)
    boot_results = {name: np.full(n_bootstrap, np.nan) for name in metrics}

    if resampling_unit == "group":
        groups = df[group_column].to_numpy()
        unique_groups = pd.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_units = len(unique_groups)

        for b in range(n_bootstrap):
            sampled_groups = rng.choice(unique_groups, size=n_units, replace=True)
            idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
            m = compute_point_metrics(y_true_all[idx], y_pred_all[idx], y_prob_all[idx], num_classes)
            for name in metrics:
                boot_results[name][b] = m[name]
    else:
        n_units = len(df)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_units, size=n_units)
            m = compute_point_metrics(y_true_all[idx], y_pred_all[idx], y_prob_all[idx], num_classes)
            for name in metrics:
                boot_results[name][b] = m[name]

    return boot_results


def summarize_confidence_intervals(point_metrics, boot_results, n_bootstrap, confidence_level):
    """Compute percentile CIs and valid/invalid counts for every metric."""
    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    rows = []
    for name, values in boot_results.items():
        valid = values[~np.isnan(values)]
        n_valid = int(len(valid))
        n_invalid = int(n_bootstrap - n_valid)
        if n_valid > 0:
            ci_lower, ci_upper = np.percentile(valid, [lower_pct, upper_pct])
        else:
            ci_lower, ci_upper = float("nan"), float("nan")
        rows.append({
            "metric": name,
            "point_estimate": point_metrics[name],
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n_bootstrap_valid": n_valid,
            "n_bootstrap_invalid": n_invalid,
        })
    return rows


def _sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(output_dir, rows, config, input_file_path):
    """Write bootstrap_ci.csv and bootstrap_summary.json. No other files are created."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_df = pd.DataFrame([
        {
            "metric": r["metric"],
            "point_estimate": r["point_estimate"],
            "confidence_level": config["confidence_level"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
            "n_bootstrap_requested": config["n_bootstrap"],
            "n_bootstrap_valid": r["n_bootstrap_valid"],
            "n_bootstrap_invalid": r["n_bootstrap_invalid"],
            "seed": config["seed"],
            "resampling_unit": config["resampling_unit"],
        }
        for r in rows
    ])
    ci_csv_path = output_dir / "bootstrap_ci.csv"
    ci_df.to_csv(ci_csv_path, index=False)

    summary = {
        "config": config,
        "input_file_sha256": _sha256_of_file(input_file_path),
        "point_estimates": {r["metric"]: r["point_estimate"] for r in rows},
        "confidence_intervals": {
            r["metric"]: {
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "n_bootstrap_valid": r["n_bootstrap_valid"],
                "n_bootstrap_invalid": r["n_bootstrap_invalid"],
            }
            for r in rows
        },
    }
    summary_path = output_dir / "bootstrap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["output_files_sha256"] = {"bootstrap_ci.csv": _sha256_of_file(ci_csv_path)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not (0.0 < args.confidence_level < 1.0):
        parser.error("--confidence-level must be strictly between 0 and 1")
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be a positive integer")

    predictions_path = Path(args.predictions_file).resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(f"--predictions-file not found: {predictions_path}")

    df = load_predictions_table(predictions_path)
    df, num_classes = validate_predictions(df, args.group_column, args.resampling_unit)

    y_true_all = df["true_label"].to_numpy(dtype=int)
    y_pred_all = df["predicted_label"].to_numpy(dtype=int)
    prob_columns = [f"prob_class_{c}" for c in range(num_classes)]
    y_prob_all = df[prob_columns].to_numpy(dtype=float)

    point_metrics = compute_point_metrics(y_true_all, y_pred_all, y_prob_all, num_classes)
    check_point_estimate_self_consistency(y_true_all, y_pred_all, point_metrics, num_classes)

    boot_results = run_bootstrap(
        df, num_classes, args.n_bootstrap, args.seed, args.resampling_unit, args.group_column
    )
    rows = summarize_confidence_intervals(
        point_metrics, boot_results, args.n_bootstrap, args.confidence_level
    )

    config = {
        "n_bootstrap": args.n_bootstrap,
        "confidence_level": args.confidence_level,
        "seed": args.seed,
        "resampling_unit": args.resampling_unit,
        "group_column": args.group_column if args.resampling_unit == "group" else None,
        "num_classes": num_classes,
        "n_records": int(len(df)),
    }

    output_dir = resolve_output_dir(args.output_dir)
    write_outputs(output_dir, rows, config, predictions_path)

    print(f"Bootstrap CI complete. resampling_unit={args.resampling_unit}  "
          f"n_bootstrap={args.n_bootstrap}  confidence_level={args.confidence_level}")
    print(f"Outputs written to: {output_dir}")

    return {"config": config, "rows": rows}


if __name__ == "__main__":
    main()
