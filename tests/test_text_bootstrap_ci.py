"""
Tests for the generic binary-classification bootstrap confidence-interval
tool (evaluation.text.bootstrap_ci), using only programmatically generated
anonymous numeric predictions tables. No real evaluation data, real model
weights, real patient/group identifiers, or historical results are used
anywhere in this file.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import evaluation.text.bootstrap_ci as bc


def _build_predictions_df(n_per_class=6, with_groups=True, seed=0):
    """Build a small synthetic, fully anonymous binary predictions table."""
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for true_label in (0, 1):
        for i in range(n_per_class):
            prob_positive = 0.85 if true_label == 1 else 0.15
            prob_positive = min(max(prob_positive + rng.normal(scale=0.03), 0.01), 0.99)
            predicted_label = true_label if rng.random() > 0.15 else 1 - true_label
            row = {
                "sample_index": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "prob_class_0": 1.0 - prob_positive,
                "prob_class_1": prob_positive,
            }
            if with_groups:
                # Two records per group, groups kept separate per class so
                # a single synthetic "subject" never spans both classes --
                # this gives enough distinct clusters (n_per_class // 2 per
                # class) for the bootstrap resampling space to be large
                # enough that two different seeds reliably diverge.
                row["group_id"] = f"anon_group_{true_label}_{i // 2:04d}"
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


def _write_predictions_csv(tmp_path, df, name="predictions.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Import has no side effects
# ---------------------------------------------------------------------------

def test_module_import_has_no_side_effects():
    import importlib
    importlib.reload(bc)
    assert not (bc.EVALUATION_DIR / "outputs" / "text_bootstrap_ci").exists()


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

def test_help_lists_all_required_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        bc.main(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    for flag in (
        "--predictions-file", "--output-dir", "--n-bootstrap", "--confidence-level",
        "--seed", "--group-column", "--resampling-unit",
    ):
        assert flag in captured.out


# ---------------------------------------------------------------------------
# Formal defaults
# ---------------------------------------------------------------------------

def test_default_n_bootstrap_is_10000():
    args = bc.build_arg_parser().parse_args(["--predictions-file", "x.csv"])
    assert args.n_bootstrap == 10000


def test_default_confidence_level_is_0_95():
    args = bc.build_arg_parser().parse_args(["--predictions-file", "x.csv"])
    assert args.confidence_level == 0.95


def test_default_seed_is_20260909():
    args = bc.build_arg_parser().parse_args(["--predictions-file", "x.csv"])
    assert args.seed == 20260909


def test_default_resampling_unit_is_group():
    args = bc.build_arg_parser().parse_args(["--predictions-file", "x.csv"])
    assert args.resampling_unit == "group"
    assert args.group_column == "group_id"


# ---------------------------------------------------------------------------
# Group-mode validation: never silently degrades to sample-level
# ---------------------------------------------------------------------------

def test_group_mode_missing_group_id_column_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=False)
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="missing required column"):
        bc.main([
            "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
            "--n-bootstrap", "5",
        ])


def test_group_mode_group_id_with_missing_values_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[0, "group_id"] = None
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="missing values in group column"):
        bc.main([
            "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
            "--n-bootstrap", "5",
        ])


def test_group_mode_group_id_empty_string_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[0, "group_id"] = "   "
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="empty-string values"):
        bc.main([
            "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
            "--n-bootstrap", "5",
        ])


def test_group_resampling_keeps_full_group_together(monkeypatch):
    df = pd.DataFrame({
        "sample_index": range(6),
        "true_label": [0, 0, 0, 1, 1, 1],
        "predicted_label": [0, 0, 0, 1, 1, 1],
        "prob_class_0": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
        "prob_class_1": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
        "group_id": ["group_a", "group_a", "group_a", "group_b", "group_c", "group_c"],
    })
    df = bc.validate_predictions(df, "group_id", "group")

    observed_lengths = []
    original = bc.compute_point_metrics

    def spy(y_true, y_pred, y_prob_positive):
        observed_lengths.append(len(y_true))
        return original(y_true, y_pred, y_prob_positive)

    monkeypatch.setattr(bc, "compute_point_metrics", spy)
    bc.run_bootstrap(df, n_bootstrap=200, seed=1, resampling_unit="group", group_column="group_id")

    # group sizes are {3, 1, 2}; 3 draws with replacement from these sizes
    # can only ever sum to a value between 3 (all size-1) and 9 (all size-3).
    assert all(3 <= length <= 9 for length in observed_lengths)
    assert len(set(observed_lengths)) > 1


def test_sample_mode_requires_explicit_flag(tmp_path):
    df = _build_predictions_df(with_groups=False)
    path = _write_predictions_csv(tmp_path, df)
    result = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--resampling-unit", "sample", "--n-bootstrap", "5",
    ])
    assert result["config"]["resampling_unit"] == "sample"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_duplicate_sample_index_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[1, "sample_index"] = df.loc[0, "sample_index"]
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="duplicate"):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


def test_invalid_label_value_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[0, "true_label"] = 2
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="outside the allowed class range"):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.1, 1.5])
def test_invalid_probability_value_fails_clearly(tmp_path, bad_value):
    df = _build_predictions_df(with_groups=True)
    df.loc[0, "prob_class_0"] = bad_value
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


def test_probabilities_not_summing_to_one_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[0, "prob_class_0"] = 0.9
    df.loc[0, "prob_class_1"] = 0.9
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="do not sum to 1"):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


def test_empty_predictions_file_fails_clearly(tmp_path):
    empty_df = pd.DataFrame(columns=["sample_index", "true_label", "predicted_label", "prob_class_0", "prob_class_1", "group_id"])
    path = _write_predictions_csv(tmp_path, empty_df)
    with pytest.raises(ValueError, match="no records"):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


def test_missing_predictions_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        bc.main([
            "--predictions-file", str(tmp_path / "does_not_exist.csv"),
            "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5",
        ])


# ---------------------------------------------------------------------------
# Binary metrics: positive class fixed to 1, values match a hand-checkable case
# ---------------------------------------------------------------------------

def test_metrics_match_hand_checkable_perfect_case(tmp_path):
    df = pd.DataFrame({
        "sample_index": range(4),
        "true_label": [0, 0, 1, 1],
        "predicted_label": [0, 0, 1, 1],
        "prob_class_0": [0.9, 0.9, 0.1, 0.1],
        "prob_class_1": [0.1, 0.1, 0.9, 0.9],
        "group_id": ["g0", "g1", "g2", "g3"],
    })
    path = _write_predictions_csv(tmp_path, df)

    result = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--n-bootstrap", "5",
    ])
    point_estimates = {r["metric"]: r["point_estimate"] for r in result["rows"]}

    assert point_estimates["accuracy"] == pytest.approx(1.0)
    assert point_estimates["precision"] == pytest.approx(1.0)
    assert point_estimates["recall"] == pytest.approx(1.0)
    assert point_estimates["f1"] == pytest.approx(1.0)
    assert point_estimates["roc_auc"] == pytest.approx(1.0)
    assert point_estimates["average_precision"] == pytest.approx(1.0)


def test_metrics_match_sklearn_on_one_misclassification(tmp_path):
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])  # one false positive, one false negative
    prob_1 = np.array([0.2, 0.6, 0.3, 0.8, 0.9, 0.4])

    df = pd.DataFrame({
        "sample_index": range(6),
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_class_0": 1 - prob_1,
        "prob_class_1": prob_1,
        "group_id": [f"g{i}" for i in range(6)],
    })
    path = _write_predictions_csv(tmp_path, df)

    result = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--n-bootstrap", "5",
    ])
    point_estimates = {r["metric"]: r["point_estimate"] for r in result["rows"]}

    expected_f1 = f1_score(y_true, y_pred, average="binary")
    assert point_estimates["accuracy"] == pytest.approx(4 / 6)
    assert point_estimates["f1"] == pytest.approx(expected_f1)


# ---------------------------------------------------------------------------
# NaN handling for ROC-AUC / AP when a resample lacks both classes
# ---------------------------------------------------------------------------

def test_roc_auc_and_ap_are_nan_when_only_one_true_class_present():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.4, 0.7])

    metrics = bc.compute_point_metrics(y_true, y_pred, y_prob)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["average_precision"])
    assert not np.isnan(metrics["accuracy"])


def test_bootstrap_reports_valid_and_invalid_counts_for_undefined_metrics(tmp_path):
    # All records share one group and one true class, so every group-level
    # resample necessarily reproduces the same single-class sample --
    # ROC-AUC/AP must be NaN (and reported as invalid) every time.
    df = pd.DataFrame({
        "sample_index": range(4),
        "true_label": [1, 1, 1, 1],
        "predicted_label": [1, 1, 0, 1],
        "prob_class_0": [0.1, 0.2, 0.6, 0.3],
        "prob_class_1": [0.9, 0.8, 0.4, 0.7],
        "group_id": ["only_group"] * 4,
    })
    path = _write_predictions_csv(tmp_path, df)

    n_bootstrap = 30
    result = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--n-bootstrap", str(n_bootstrap), "--seed", "1",
    ])
    rows_by_metric = {r["metric"]: r for r in result["rows"]}

    assert rows_by_metric["roc_auc"]["n_bootstrap_valid"] == 0
    assert rows_by_metric["roc_auc"]["n_bootstrap_invalid"] == n_bootstrap
    assert np.isnan(rows_by_metric["roc_auc"]["ci_lower"])
    assert np.isnan(rows_by_metric["roc_auc"]["ci_upper"])
    assert rows_by_metric["accuracy"]["n_bootstrap_valid"] == n_bootstrap


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_results(tmp_path):
    df = _build_predictions_df(n_per_class=10, with_groups=True)
    path = _write_predictions_csv(tmp_path, df)

    result_a = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out_a"),
        "--n-bootstrap", "200", "--seed", "7",
    ])
    result_b = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out_b"),
        "--n-bootstrap", "200", "--seed", "7",
    ])

    df_a = pd.read_csv(tmp_path / "out_a" / "bootstrap_ci.csv")
    df_b = pd.read_csv(tmp_path / "out_b" / "bootstrap_ci.csv")
    pd.testing.assert_frame_equal(df_a, df_b)
    assert result_a["rows"] == result_b["rows"]


def test_different_seed_gives_different_bootstrap_draws(tmp_path):
    df = _build_predictions_df(n_per_class=10, with_groups=True)
    path = _write_predictions_csv(tmp_path, df)

    bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out_a"),
        "--n-bootstrap", "300", "--seed", "1",
    ])
    bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out_b"),
        "--n-bootstrap", "300", "--seed", "2",
    ])

    df_a = pd.read_csv(tmp_path / "out_a" / "bootstrap_ci.csv")
    df_b = pd.read_csv(tmp_path / "out_b" / "bootstrap_ci.csv")
    assert not df_a["ci_lower"].equals(df_b["ci_lower"]) or not df_a["ci_upper"].equals(df_b["ci_upper"])


# ---------------------------------------------------------------------------
# Output sanity: CI ordering, valid ranges, requested/valid/invalid counts
# ---------------------------------------------------------------------------

def test_ci_bounds_ordered_within_range_and_counts_reported(tmp_path):
    df = _build_predictions_df(n_per_class=8, with_groups=True)
    path = _write_predictions_csv(tmp_path, df)

    n_bootstrap = 150
    bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--n-bootstrap", str(n_bootstrap), "--seed", "3",
    ])
    ci_df = pd.read_csv(tmp_path / "out" / "bootstrap_ci.csv")

    assert set(ci_df["metric"]) == set(bc.METRIC_NAMES)

    for column in ("point_estimate", "ci_lower", "ci_upper"):
        values = ci_df[column].dropna()
        assert (values >= -1e-9).all()
        assert (values <= 1 + 1e-9).all()

    valid_rows = ci_df.dropna(subset=["ci_lower", "ci_upper"])
    assert (valid_rows["ci_lower"] <= valid_rows["ci_upper"] + 1e-12).all()

    assert (ci_df["n_bootstrap_requested"] == n_bootstrap).all()
    assert (ci_df["n_bootstrap_valid"] + ci_df["n_bootstrap_invalid"] == n_bootstrap).all()


# ---------------------------------------------------------------------------
# Privacy: outputs exclude raw group values and paths
# ---------------------------------------------------------------------------

def test_outputs_exclude_raw_group_values_and_paths(tmp_path):
    df = _build_predictions_df(n_per_class=4, with_groups=True)
    distinctive_group_value = "definitely_not_a_real_identifier_98765"
    df.loc[0, "group_id"] = distinctive_group_value
    path = _write_predictions_csv(tmp_path, df)

    bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--n-bootstrap", "5",
    ])

    for output_name in ("bootstrap_ci.csv", "bootstrap_summary.json"):
        text = (tmp_path / "out" / output_name).read_text(encoding="utf-8")
        assert distinctive_group_value not in text
        assert str(tmp_path) not in text


# ---------------------------------------------------------------------------
# No plotting, no image artifacts, no DeLong/McNemar/DCA/calibration
# ---------------------------------------------------------------------------

def test_source_does_not_import_plotting_or_other_statistics_libraries():
    source = inspect.getsource(bc)
    for forbidden in (
        "matplotlib", "seaborn", "plotly", "bokeh", "altair",
        "delong", "DeLong", "mcnemar", "McNemar", "calibration",
        "decision_curve", "DCA",
    ):
        assert forbidden not in source


def test_no_image_files_created_after_run(tmp_path):
    df = _build_predictions_df(with_groups=True)
    path = _write_predictions_csv(tmp_path, df)
    output_dir = tmp_path / "out"
    bc.main(["--predictions-file", str(path), "--output-dir", str(output_dir), "--n-bootstrap", "5"])

    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".bmp", ".tiff"}
    created_images = [p for p in output_dir.rglob("*") if p.suffix.lower() in image_extensions]
    assert created_images == []


def test_only_two_output_files_are_created(tmp_path):
    df = _build_predictions_df(with_groups=True)
    path = _write_predictions_csv(tmp_path, df)
    output_dir = tmp_path / "out"
    bc.main(["--predictions-file", str(path), "--output-dir", str(output_dir), "--n-bootstrap", "5"])

    created_files = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    assert created_files == ["bootstrap_ci.csv", "bootstrap_summary.json"]


# ---------------------------------------------------------------------------
# No historical dataset names, sizes, results, or forbidden content
# ---------------------------------------------------------------------------

NEW_FILES = [
    "evaluation/text/bootstrap_ci.py",
]


def test_new_files_contain_no_forbidden_content():
    import re as _re
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    # Built via concatenation (not written as contiguous literals) so this
    # test file's own pattern list does not itself trip the repo-wide
    # drive-letter-path scanner in test_text_pipeline_path_scan.py.
    colon = ":"
    forbidden_substrings = [
        "E" + colon + "\\eye", "F" + colon + "\\eye",
        "E" + colon + "/eye", "F" + colon + "/eye",
        "C" + colon + "\\Users",
        "ZOC", "Zhongshan",
    ]
    forbidden_pattern = _re.compile(r"password|api[_-]?key|secret[_-]?key", _re.IGNORECASE)
    cjk_pattern = _re.compile("[" + chr(0x4E00) + "-" + chr(0x9FFF) + "]")

    for relative_path in NEW_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for needle in forbidden_substrings:
            assert needle not in text, f"forbidden substring {needle!r} found in {relative_path}"
        assert not forbidden_pattern.search(text), f"forbidden secret-like pattern found in {relative_path}"
        assert not cjk_pattern.search(text), f"CJK character found in {relative_path}"


def test_new_files_contain_no_id_or_phone_shaped_literals():
    import re as _re
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    id_or_phone_pattern = _re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{11})(?!\d)")
    for relative_path in NEW_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert not id_or_phone_pattern.search(text), f"ID/phone-shaped literal found in {relative_path}"
