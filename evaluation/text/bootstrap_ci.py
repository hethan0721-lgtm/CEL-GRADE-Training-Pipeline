#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic clustered/record-level bootstrap confidence-interval tool for
binary classification predictions.

Consumes a predictions CSV (such as the one produced by
evaluation.text.external_validation) and computes percentile bootstrap
confidence intervals for a fixed set of binary classification metrics.
Performs no inference, training, or fitting -- it only reads already-saved
predictions and probabilities.

Usage:
    python -m evaluation.text.bootstrap_ci \\
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

POSITIVE_CLASS = 1
ALLOWED_LABELS = {0, 1}
REQUIRED_BASE_COLUMNS = ["sample_index", "true_label", "predicted_label", "prob_class_0", "prob_class_1"]
PROB_COLUMNS = ["prob_class_0", "prob_class_1"]
PROBABILITY_SUM_TOLERANCE = 1e-2

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def build_arg_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluation.text.bootstrap_ci",
        description=(
            "Compute percentile bootstrap confidence intervals for binary "
            "classification metrics from a predictions CSV. Reads only the "
            "given predictions file; performs no inference, training, or "
            "fitting."
        ),
    )
    parser.add_argument(
        "--predictions-file", type=str, required=True,
        help="Path to a predictions CSV containing sample_index, true_label, "
             "predicted_label, prob_class_0, and prob_class_1 columns (such "
             "as the output of evaluation.text.external_validation).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write bootstrap outputs to. Created only when "
             "the program runs (default: <evaluation package>/outputs/text_bootstrap_ci).",
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
    """Resolve --output-dir, defaulting to <evaluation package>/outputs/text_bootstrap_ci."""
    if output_dir_arg is not None:
        return Path(output_dir_arg).resolve()
    return (EVALUATION_DIR / "outputs" / "text_bootstrap_ci").resolve()


def load_predictions_table(path):
    """Read the predictions CSV into a DataFrame."""
    return pd.read_csv(path)


def validate_predictions(df, group_column, resampling_unit):
    """
    Validate the predictions table against the required contract and return
    the validated DataFrame. Raises ValueError with a specific, actionable
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

    if df["sample_index"].isna().any():
        raise ValueError("Predictions file has missing values in column 'sample_index'.")
    if df["sample_index"].duplicated().any():
        duplicates = sorted(set(df.loc[df["sample_index"].duplicated(), "sample_index"]))
        raise ValueError(f"Predictions file has duplicate 'sample_index' values: {duplicates}")

    for column in ("true_label", "predicted_label"):
        if df[column].isna().any():
            raise ValueError(f"Predictions file has missing values in column '{column}'.")
        invalid = sorted(set(df[column].unique()) - ALLOWED_LABELS)
        if invalid:
            raise ValueError(
                f"Predictions file column '{column}' contains value(s) outside "
                f"the allowed class range {sorted(ALLOWED_LABELS)}: {invalid}"
            )

    prob_values = df[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(prob_values).all():
        raise ValueError(
            f"Predictions file probability columns {PROB_COLUMNS} contain "
            f"non-finite (NaN/Inf) values."
        )
    if (prob_values < 0).any() or (prob_values > 1).any():
        raise ValueError(
            f"Predictions file probability columns {PROB_COLUMNS} contain "
            f"values outside the [0, 1] range."
        )
    row_sums = prob_values.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > PROBABILITY_SUM_TOLERANCE):
        raise ValueError(
            f"Predictions file rows exist where {PROB_COLUMNS} do not sum to "
            f"1 within tolerance {PROBABILITY_SUM_TOLERANCE}."
        )

    if resampling_unit == "group":
        group_series = df[group_column]
        if group_series.isna().any():
            raise ValueError(f"Predictions file has missing values in group column '{group_column}'.")
        if (group_series.astype(str).str.strip() == "").any():
            raise ValueError(f"Predictions file has empty-string values in group column '{group_column}'.")

    return df.reset_index(drop=True)


def compute_point_metrics(y_true, y_pred, y_prob_positive):
    """Compute the fixed binary metric set for one true/predicted/probability array triple.

    Returns a dict of metric_name -> value, using NaN for any metric that is
    mathematically undefined for the given sample (e.g. no positive
    predictions for precision, or a single true class for ROC-AUC/AP).
    """
    n = len(y_true)
    out = {"accuracy": float(np.mean(y_true == y_pred))}

    tp = int(np.sum((y_true == POSITIVE_CLASS) & (y_pred == POSITIVE_CLASS)))
    fp = int(np.sum((y_true != POSITIVE_CLASS) & (y_pred == POSITIVE_CLASS)))
    fn = int(np.sum((y_true == POSITIVE_CLASS) & (y_pred != POSITIVE_CLASS)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if not np.isnan(precision) and not np.isnan(recall):
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = f1

    n_pos = int(np.sum(y_true == POSITIVE_CLASS))
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob_positive))
        out["average_precision"] = float(average_precision_score(y_true, y_prob_positive))
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan

    return out


def check_point_estimate_self_consistency(y_true, y_pred, point_metrics):
    """
    Recompute the confusion matrix independently from y_true/y_pred and
    verify it is internally consistent with the point-estimate accuracy
    before any bootstrap resampling begins. Raises RuntimeError on failure.
    """
    n = len(y_true)
    confusion = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
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


def run_bootstrap(df, n_bootstrap, seed, resampling_unit, group_column):
    """Run the percentile bootstrap and return {metric_name: np.ndarray of length n_bootstrap}."""
    y_true_all = df["true_label"].to_numpy(dtype=int)
    y_pred_all = df["predicted_label"].to_numpy(dtype=int)
    y_prob_all = df["prob_class_1"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    boot_results = {name: np.full(n_bootstrap, np.nan) for name in METRIC_NAMES}

    if resampling_unit == "group":
        groups = df[group_column].to_numpy()
        unique_groups = pd.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_units = len(unique_groups)

        for b in range(n_bootstrap):
            sampled_groups = rng.choice(unique_groups, size=n_units, replace=True)
            idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
            m = compute_point_metrics(y_true_all[idx], y_pred_all[idx], y_prob_all[idx])
            for name in METRIC_NAMES:
                boot_results[name][b] = m[name]
    else:
        n_units = len(df)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_units, size=n_units)
            m = compute_point_metrics(y_true_all[idx], y_pred_all[idx], y_prob_all[idx])
            for name in METRIC_NAMES:
                boot_results[name][b] = m[name]

    return boot_results


def summarize_confidence_intervals(point_metrics, boot_results, n_bootstrap, confidence_level):
    """Compute percentile CIs and valid/invalid counts for every metric."""
    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    rows = []
    for name in METRIC_NAMES:
        values = boot_results[name]
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
    df = validate_predictions(df, args.group_column, args.resampling_unit)

    y_true_all = df["true_label"].to_numpy(dtype=int)
    y_pred_all = df["predicted_label"].to_numpy(dtype=int)
    y_prob_all = df["prob_class_1"].to_numpy(dtype=float)

    point_metrics = compute_point_metrics(y_true_all, y_pred_all, y_prob_all)
    check_point_estimate_self_consistency(y_true_all, y_pred_all, point_metrics)

    boot_results = run_bootstrap(
        df, args.n_bootstrap, args.seed, args.resampling_unit, args.group_column
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
        "positive_class": POSITIVE_CLASS,
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
