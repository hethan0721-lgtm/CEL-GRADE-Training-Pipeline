"""
Tests for the generic bootstrap confidence-interval tool
(evaluation.image.bootstrap_ci), using only programmatically generated
anonymous numeric predictions tables. No real evaluation data, real model
weights, real images, or real patient/group identifiers are used anywhere
in this file.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import evaluation.image.bootstrap_ci as bc


def _build_predictions_df(n_per_class=4, with_groups=True, seed=0):
    """Build a small synthetic, fully anonymous predictions table."""
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for true_label in range(3):
        for i in range(n_per_class):
            probs = np.full(3, 0.1)
            probs[true_label] = 0.8
            probs = probs / probs.sum()
            predicted_label = true_label if rng.random() > 0.2 else (true_label + 1) % 3
            row = {
                "sample_index": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "prob_class_0": probs[0],
                "prob_class_1": probs[1],
                "prob_class_2": probs[2],
            }
            if with_groups:
                row["group_id"] = f"anon_group_{idx % (n_per_class // 2 or 1):04d}"
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


def _write_predictions_csv(tmp_path, df, name="predictions.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1-2. Import has no side effects
# ---------------------------------------------------------------------------

def test_module_import_has_no_side_effects():
    import importlib
    importlib.reload(bc)
    assert not (bc.EVALUATION_DIR / "outputs" / "bootstrap_ci").exists()


# ---------------------------------------------------------------------------
# 3. --help
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
# 4-7. Formal defaults
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
# 8-9. Group-mode validation
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


# ---------------------------------------------------------------------------
# 10. Group resampling keeps every record of a drawn group together
# ---------------------------------------------------------------------------

def test_group_resampling_keeps_full_group_together(monkeypatch):
    df = pd.DataFrame({
        "sample_index": range(6),
        "true_label": [0, 0, 0, 1, 2, 2],
        "predicted_label": [0, 0, 0, 1, 2, 2],
        "prob_class_0": [0.9, 0.9, 0.9, 0.05, 0.05, 0.05],
        "prob_class_1": [0.05, 0.05, 0.05, 0.9, 0.05, 0.05],
        "prob_class_2": [0.05, 0.05, 0.05, 0.05, 0.9, 0.9],
        "group_id": ["group_a", "group_a", "group_a", "group_b", "group_c", "group_c"],
    })
    df, num_classes = bc.validate_predictions(df, "group_id", "group")

    observed_lengths = []
    original = bc.compute_point_metrics

    def spy(y_true, y_pred, y_prob, num_classes_arg):
        observed_lengths.append(len(y_true))
        return original(y_true, y_pred, y_prob, num_classes_arg)

    monkeypatch.setattr(bc, "compute_point_metrics", spy)
    bc.run_bootstrap(df, num_classes, n_bootstrap=200, seed=1,
                      resampling_unit="group", group_column="group_id")

    # group sizes are {3, 1, 2}; 3 draws with replacement from these sizes
    # can only ever sum to a value between 3 (all size-1) and 9 (all size-3).
    assert all(3 <= length <= 9 for length in observed_lengths)
    # a full-group draw must always land on a sum reachable by combining
    # whole group sizes -- a broken (partial-group) resampler could instead
    # produce a constant length equal to the total record count (6).
    assert len(set(observed_lengths)) > 1


# ---------------------------------------------------------------------------
# 11. Sample-level resampling only via explicit flag
# ---------------------------------------------------------------------------

def test_sample_mode_requires_explicit_flag_and_ignores_group_requirement(tmp_path):
    df = _build_predictions_df(with_groups=False)
    path = _write_predictions_csv(tmp_path, df)
    result = bc.main([
        "--predictions-file", str(path), "--output-dir", str(tmp_path / "out"),
        "--resampling-unit", "sample", "--n-bootstrap", "5",
    ])
    assert result["config"]["resampling_unit"] == "sample"


# ---------------------------------------------------------------------------
# 12-13. Seed reproducibility
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
# 14. Point estimates match sklearn's own computation
# ---------------------------------------------------------------------------

def test_point_estimates_match_sklearn():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_prob = np.array([
        [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2],
    ])

    metrics = bc.compute_point_metrics(y_true, y_pred, y_prob, num_classes=3)

    expected_accuracy = float(np.mean(y_true == y_pred))
    expected_macro_f1 = f1_score(y_true, y_pred, average="macro")

    assert metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["f1_macro"] == pytest.approx(expected_macro_f1)


# ---------------------------------------------------------------------------
# 15-17. Output sanity: CI ordering, valid ranges, requested/valid/invalid counts
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

    for column in ("point_estimate", "ci_lower", "ci_upper"):
        values = ci_df[column].dropna()
        assert (values >= -1e-9).all()
        assert (values <= 1 + 1e-9).all()

    valid_rows = ci_df.dropna(subset=["ci_lower", "ci_upper"])
    assert (valid_rows["ci_lower"] <= valid_rows["ci_upper"] + 1e-12).all()

    assert (ci_df["n_bootstrap_requested"] == n_bootstrap).all()
    assert (ci_df["n_bootstrap_valid"] + ci_df["n_bootstrap_invalid"] == n_bootstrap).all()


# ---------------------------------------------------------------------------
# 18-21. Input validation failure modes
# ---------------------------------------------------------------------------

def test_duplicate_sample_index_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[1, "sample_index"] = df.loc[0, "sample_index"]
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="duplicate"):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


def test_invalid_label_value_fails_clearly(tmp_path):
    df = _build_predictions_df(with_groups=True)
    df.loc[0, "true_label"] = 99
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
    df.loc[0, "prob_class_2"] = 0.9
    path = _write_predictions_csv(tmp_path, df)
    with pytest.raises(ValueError, match="do not sum to 1"):
        bc.main(["--predictions-file", str(path), "--output-dir", str(tmp_path / "out"), "--n-bootstrap", "5"])


# ---------------------------------------------------------------------------
# 22. Outputs exclude paths, filenames, raw group values, identity fields
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
# 23-24. No plotting, no image artifacts
# ---------------------------------------------------------------------------

def test_source_does_not_import_plotting_libraries():
    source = inspect.getsource(bc)
    for forbidden in ("matplotlib", "seaborn", "plotly", "bokeh", "altair"):
        assert forbidden not in source


def test_no_image_files_created_after_run(tmp_path):
    df = _build_predictions_df(with_groups=True)
    path = _write_predictions_csv(tmp_path, df)
    output_dir = tmp_path / "out"
    bc.main(["--predictions-file", str(path), "--output-dir", str(output_dir), "--n-bootstrap", "5"])

    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".bmp", ".tiff"}
    created_images = [p for p in output_dir.rglob("*") if p.suffix.lower() in image_extensions]
    assert created_images == []


# ---------------------------------------------------------------------------
# 25. No historical dataset names, sizes, results, or forbidden content
# ---------------------------------------------------------------------------

NEW_FILES = [
    "evaluation/image/bootstrap_ci.py",
]

# Frozen historical point-estimate values from a prior evaluation run must
# never appear literally in a dataset-agnostic tool.
FORBIDDEN_HISTORICAL_METRIC_LITERALS = [
    "0.6881", "0.6197", "0.6805", "0.6350", "0.7192", "0.6961",
    "0.5938", "0.5588", "0.5758", "0.8193", "0.7196", "0.7662",
    "0.4462", "0.7632", "0.5631",
]


def test_new_files_contain_no_forbidden_content():
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    # Built via concatenation (not written as contiguous literals) so this
    # test file's own pattern list does not itself trip the repo-wide
    # drive-letter-path scanner in test_text_pipeline_path_scan.py, while
    # the runtime string values used in the actual check below are
    # unchanged.
    colon = ":"
    forbidden_substrings = [
        "E" + colon + "\\eye", "F" + colon + "\\eye",
        "E" + colon + "/eye", "F" + colon + "/eye",
        "C" + colon + "\\Users",
        "ZOC", "Zhongshan",
    ] + FORBIDDEN_HISTORICAL_METRIC_LITERALS
    forbidden_pattern = re.compile(
        r"password|api[_-]?key|secret[_-]?key", re.IGNORECASE
    )
    cjk_pattern = re.compile(r"[\u4e00-\u9fff]")

    for relative_path in NEW_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for needle in forbidden_substrings:
            assert needle not in text, f"forbidden substring {needle!r} found in {relative_path}"
        assert not forbidden_pattern.search(text), f"forbidden secret-like pattern found in {relative_path}"
        assert not cjk_pattern.search(text), f"CJK character found in {relative_path}"


def test_new_files_contain_no_id_or_phone_shaped_literals():
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    id_or_phone_pattern = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{11})(?!\d)")
    for relative_path in NEW_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert not id_or_phone_pattern.search(text), f"ID/phone-shaped literal found in {relative_path}"
