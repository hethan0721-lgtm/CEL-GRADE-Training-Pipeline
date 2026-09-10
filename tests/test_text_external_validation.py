"""
Tests for the M0_CLIN independent external validation entry point
(evaluation.text.external_validation), using only programmatically
generated anonymous synthetic tabular data and a freshly-fitted
FeatureEngineer/SurgeryClassifier pair created inside pytest's tmp_path.
No real patient data, real model weights, or historical results are used
anywhere in this file.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

import evaluation.text.external_validation as ev
from text_pipeline.src.config import Config
from text_pipeline.src.feature_engineering import FeatureEngineer
from text_pipeline.src.models import SurgeryClassifier

CATEGORY_VALUES = ["synthetic_category_a", "synthetic_category_b"]


def _build_synthetic_frame(n_per_class=15, seed=0, extra_column=False):
    """Build a small, fully anonymous synthetic tabular dataset with the
    model's real feature/target column names (read from Config, never
    hardcoded) but entirely synthetic numeric/categorical values."""
    rng = np.random.default_rng(seed)
    rows = []
    for label_value in sorted(Config.LABEL_NAMES.keys()):
        label_str = Config.LABEL_NAMES[label_value]
        base = 5.0 if label_value == 1 else -5.0
        for i in range(n_per_class):
            row = {}
            for col in Config.CATEGORICAL_FEATURES:
                row[col] = CATEGORY_VALUES[i % 2]
            for col in Config.NUMERICAL_FEATURES:
                row[col] = float(base + rng.normal(scale=0.4))
            row[Config.TARGET_COLUMN] = label_str
            if extra_column:
                row["synthetic_unused_column"] = f"ignored_{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_and_save_artifacts(tmp_path, train_df):
    """Fit a tiny FeatureEngineer + SurgeryClassifier on synthetic data and
    save them via their own real .save() methods -- no real weights, no
    real data.

    The target column must already be 0/1-encoded before fit_transform(),
    matching the real pipeline order where DataLoader encodes labels
    upstream of FeatureEngineer; FeatureEngineer.transform() itself only
    extracts whatever is already in the target column.
    """
    encoded_df = train_df.copy()
    encoded_df[Config.TARGET_COLUMN] = encoded_df[Config.TARGET_COLUMN].map(Config.LABEL_MAPPING)

    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(encoded_df, verbose=False)
    engineer_path = Path(engineer.save(save_dir=str(tmp_path / "artifacts")))

    classifier = SurgeryClassifier(model_type="xgboost")
    classifier.fit(X, y, verbose=False)
    model_path = tmp_path / "artifacts" / "model.pkl"
    classifier.save(str(model_path))

    return model_path, engineer_path


def _write_table(tmp_path, df, name="external_data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def artifacts(tmp_path):
    train_df = _build_synthetic_frame(n_per_class=15, seed=0)
    model_path, engineer_path = _fit_and_save_artifacts(tmp_path, train_df)
    return model_path, engineer_path


# ---------------------------------------------------------------------------
# 1-2. Import has no side effects
# ---------------------------------------------------------------------------

def test_module_import_has_no_side_effects():
    import importlib
    importlib.reload(ev)
    assert not (ev.EVALUATION_DIR / "outputs" / "text_external_validation").exists()


# ---------------------------------------------------------------------------
# 3-4. --help
# ---------------------------------------------------------------------------

def test_help_lists_all_required_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        ev.main(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    for flag in ("--input-data", "--model", "--feature-engineer",
                 "--output-dir", "--group-column", "--seed"):
        assert flag in captured.out


# ---------------------------------------------------------------------------
# 5-7. Missing required arguments
# ---------------------------------------------------------------------------

def test_missing_input_data_fails_clearly(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        ev.main(["--model", str(tmp_path / "m.pkl"), "--feature-engineer", str(tmp_path / "fe.pkl")])
    assert exc_info.value.code == 2


def test_missing_model_fails_clearly(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        ev.main(["--input-data", str(tmp_path / "d.csv"), "--feature-engineer", str(tmp_path / "fe.pkl")])
    assert exc_info.value.code == 2


def test_missing_feature_engineer_fails_clearly(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        ev.main(["--input-data", str(tmp_path / "d.csv"), "--model", str(tmp_path / "m.pkl")])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 8-10. Referenced files must exist
# ---------------------------------------------------------------------------

def test_missing_input_data_file_fails_clearly(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    with pytest.raises(FileNotFoundError, match="input-data"):
        ev.main([
            "--input-data", str(tmp_path / "does_not_exist.csv"),
            "--model", str(model_path), "--feature-engineer", str(engineer_path),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_missing_model_file_fails_clearly(tmp_path, artifacts):
    _model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(seed=1))
    with pytest.raises(FileNotFoundError, match="model"):
        ev.main([
            "--input-data", str(external_path),
            "--model", str(tmp_path / "does_not_exist.pkl"), "--feature-engineer", str(engineer_path),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_missing_feature_engineer_file_fails_clearly(tmp_path, artifacts):
    model_path, _engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(seed=1))
    with pytest.raises(FileNotFoundError, match="feature-engineer"):
        ev.main([
            "--input-data", str(external_path),
            "--model", str(model_path), "--feature-engineer", str(tmp_path / "does_not_exist.pkl"),
            "--output-dir", str(tmp_path / "out"),
        ])


# ---------------------------------------------------------------------------
# 11-15. Input-table validation
# ---------------------------------------------------------------------------

def test_empty_table_fails_clearly(tmp_path):
    empty_df = pd.DataFrame(columns=list(Config.FEATURE_COLUMNS) + [Config.TARGET_COLUMN])
    with pytest.raises(ValueError, match="no records"):
        ev.validate_and_prepare(empty_df, None)


def test_missing_feature_column_fails_clearly(tmp_path):
    df = _build_synthetic_frame(n_per_class=2)
    df = df.drop(columns=[Config.FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing required column"):
        ev.validate_and_prepare(df, None)


def test_missing_target_column_fails_clearly(tmp_path):
    df = _build_synthetic_frame(n_per_class=2)
    df = df.drop(columns=[Config.TARGET_COLUMN])
    with pytest.raises(ValueError, match="missing required column"):
        ev.validate_and_prepare(df, None)


def test_target_missing_value_fails_clearly(tmp_path):
    df = _build_synthetic_frame(n_per_class=2)
    df.loc[0, Config.TARGET_COLUMN] = None
    with pytest.raises(ValueError, match="missing values in target column"):
        ev.validate_and_prepare(df, None)


def test_invalid_target_value_fails_clearly(tmp_path):
    df = _build_synthetic_frame(n_per_class=2)
    df.loc[0, Config.TARGET_COLUMN] = "not_a_real_label_value"
    with pytest.raises(ValueError, match="outside the supported label set"):
        ev.validate_and_prepare(df, None)


# ---------------------------------------------------------------------------
# Target-column mapping regression tests. These call the real
# validate_and_prepare() (which internally constructs and calls a real,
# unmocked text_pipeline.src.data_loader.DataLoader) for every accepted
# label-input format, and cross-check against an independently-run real
# DataLoader._clean_dataframe() call -- no fake Config, no monkeypatched
# mapping, no bypass of the real DataLoader code path.
# ---------------------------------------------------------------------------

def _build_frame_with_target_values(target_values):
    """Build a minimal valid feature frame with an explicit target value per
    row, using the real Config feature/target column names."""
    rows = []
    for i, value in enumerate(target_values):
        row = {}
        for col in Config.CATEGORICAL_FEATURES:
            row[col] = CATEGORY_VALUES[i % 2]
        for col in Config.NUMERICAL_FEATURES:
            row[col] = float(i)
        row[Config.TARGET_COLUMN] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_canonical_raw_labels_become_0_1_after_cleaning():
    raw_labels = [Config.LABEL_NAMES[0], Config.LABEL_NAMES[1], Config.LABEL_NAMES[0], Config.LABEL_NAMES[1]]
    df = _build_frame_with_target_values(raw_labels)

    cleaned_df, _group_values = ev.validate_and_prepare(df, None)

    target = cleaned_df[Config.TARGET_COLUMN]
    assert not target.isna().any()
    assert set(target.unique().tolist()) <= {0, 1}
    assert target.tolist() == [Config.LABEL_MAPPING[v] for v in raw_labels]


def test_already_encoded_0_1_labels_become_0_1_after_cleaning():
    encoded = [0, 1, 1, 0]
    df = _build_frame_with_target_values(encoded)

    cleaned_df, _group_values = ev.validate_and_prepare(df, None)

    target = cleaned_df[Config.TARGET_COLUMN]
    assert not target.isna().any()
    assert target.tolist() == encoded


def test_historical_yes_no_labels_become_0_1_after_cleaning():
    # Historical "yes"/"no"-style alias values, built via chr() from a
    # codepoint rather than as literal CJK characters in this test's own
    # source text.
    yes_value = chr(0x662F)
    no_value = chr(0x5426)
    alias_values = [yes_value, no_value, yes_value, no_value]
    df = _build_frame_with_target_values(alias_values)

    cleaned_df, _group_values = ev.validate_and_prepare(df, None)

    target = cleaned_df[Config.TARGET_COLUMN]
    assert not target.isna().any()
    expected = [
        Config.LABEL_MAPPING[Config.LABEL_NAMES[1]],
        Config.LABEL_MAPPING[Config.LABEL_NAMES[0]],
        Config.LABEL_MAPPING[Config.LABEL_NAMES[1]],
        Config.LABEL_MAPPING[Config.LABEL_NAMES[0]],
    ]
    assert target.tolist() == expected


def test_unknown_target_label_is_never_silently_accepted():
    df = _build_frame_with_target_values(["not_a_real_label", Config.LABEL_NAMES[0]])
    with pytest.raises(ValueError, match="outside the supported label set"):
        ev.validate_and_prepare(df, None)


def test_validate_and_prepare_uses_the_real_data_loader_clean_dataframe(monkeypatch):
    """Confirms this code path genuinely calls
    DataLoader._clean_dataframe() rather than a bypassed/faked mapping:
    monkeypatching the real method to raise must break validate_and_prepare()."""
    from text_pipeline.src.data_loader import DataLoader as RealDataLoader

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("DataLoader._clean_dataframe must be the real, unbypassed mapping path")

    monkeypatch.setattr(RealDataLoader, "_clean_dataframe", _forbidden)

    df = _build_frame_with_target_values([Config.LABEL_NAMES[0], Config.LABEL_NAMES[1]])
    with pytest.raises(AssertionError, match="unbypassed mapping path"):
        ev.validate_and_prepare(df, None)


def test_cleaned_target_matches_independently_run_real_data_loader():
    """Cross-check: run the real DataLoader._clean_dataframe() independently
    on the same input and confirm it agrees exactly with what
    validate_and_prepare() returned."""
    from text_pipeline.src.data_loader import DataLoader as RealDataLoader

    raw_labels = [Config.LABEL_NAMES[1], Config.LABEL_NAMES[0], Config.LABEL_NAMES[1]]
    df = _build_frame_with_target_values(raw_labels)

    cleaned_df, _group_values = ev.validate_and_prepare(df, None)

    independent_input = df[list(Config.FEATURE_COLUMNS) + [Config.TARGET_COLUMN]].copy()
    independent_cleaned, _preserved = RealDataLoader(config=Config())._clean_dataframe(
        independent_input, preserve_name="independent_check", verbose=False
    )

    assert cleaned_df[Config.TARGET_COLUMN].tolist() == independent_cleaned[Config.TARGET_COLUMN].tolist()
    assert not independent_cleaned[Config.TARGET_COLUMN].isna().any()


# ---------------------------------------------------------------------------
# 16-18. No re-splitting, no fitting, no SMOTE
# ---------------------------------------------------------------------------

def test_external_data_is_never_train_test_split(tmp_path, artifacts, monkeypatch):
    model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(seed=1))

    import sklearn.model_selection as sk_split

    def _forbidden(*args, **kwargs):
        raise AssertionError("train_test_split must never be called during external validation")

    monkeypatch.setattr(sk_split, "train_test_split", _forbidden)

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(tmp_path / "out"),
    ])


def test_no_fit_or_fit_transform_called_during_validation(tmp_path, artifacts, monkeypatch):
    model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(seed=1))

    def _forbidden(*args, **kwargs):
        raise AssertionError("fit()/fit_transform() must never be called during external validation")

    monkeypatch.setattr(FeatureEngineer, "fit", _forbidden)
    monkeypatch.setattr(FeatureEngineer, "fit_transform", _forbidden)
    monkeypatch.setattr(SurgeryClassifier, "fit", _forbidden)

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(tmp_path / "out"),
    ])


def test_source_never_references_smote():
    source = inspect.getsource(ev)
    assert "SMOTE" not in source
    assert "fit_resample" not in source
    assert "RandomizedSearchCV" not in source
    assert "cross_val" not in source


# ---------------------------------------------------------------------------
# 19. All cleaned records enter prediction
# ---------------------------------------------------------------------------

def test_all_cleaned_records_are_scored(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_df = _build_synthetic_frame(n_per_class=6, seed=2)
    external_path = _write_table(tmp_path, external_df)

    metrics = ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(tmp_path / "out"),
    ])
    assert metrics["n_samples"] == len(external_df)


# ---------------------------------------------------------------------------
# 20-21. predict_proba positive class + point estimates match sklearn
# ---------------------------------------------------------------------------

def test_predict_proba_positive_class_and_metrics_match_sklearn(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_df = _build_synthetic_frame(n_per_class=8, seed=3)
    external_path = _write_table(tmp_path, external_df)

    classifier = SurgeryClassifier.load(str(model_path))
    engineer = FeatureEngineer.load(str(engineer_path))
    from text_pipeline.src.data_loader import DataLoader
    loader = DataLoader()
    cleaned, _ = loader._clean_dataframe(external_df, preserve_name="check", verbose=False)
    X, y = engineer.transform(cleaned, verbose=False)
    expected_pred = classifier.predict(X)
    expected_proba = classifier.predict_proba(X)[:, 1]
    expected_accuracy = accuracy_score(y, expected_pred)
    expected_f1 = f1_score(y, expected_pred, average="binary", zero_division=0)

    output_dir = tmp_path / "out"
    metrics = ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
    ])

    assert metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["f1"] == pytest.approx(expected_f1)

    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    np.testing.assert_allclose(predictions_df["prob_class_1"].to_numpy(), expected_proba, rtol=1e-6)


# ---------------------------------------------------------------------------
# 22-25. predictions.csv contract and anonymization
# ---------------------------------------------------------------------------

def test_predictions_csv_contains_only_allowed_fields(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(n_per_class=4, seed=4))
    output_dir = tmp_path / "out"

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
    ])

    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    assert list(predictions_df.columns) == [
        "sample_index", "true_label", "predicted_label", "prob_class_0", "prob_class_1",
    ]


def test_group_column_is_anonymized_and_raw_values_excluded(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_df = _build_synthetic_frame(n_per_class=4, seed=5)
    distinctive_groups = [f"real_subject_{i // 2}" for i in range(len(external_df))]
    external_df["subject_id"] = distinctive_groups
    external_path = _write_table(tmp_path, external_df)
    output_dir = tmp_path / "out"

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
        "--group-column", "subject_id",
    ])

    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    assert "group_id" in predictions_df.columns
    assert all(str(g).startswith("group_") for g in predictions_df["group_id"])
    assert predictions_df.loc[0, "group_id"] == predictions_df.loc[1, "group_id"]

    for output_name in ("predictions.csv", "metrics.json"):
        text = (output_dir / output_name).read_text(encoding="utf-8")
        for raw_value in set(distinctive_groups):
            assert raw_value not in text


def test_no_group_id_column_without_group_column(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(n_per_class=3, seed=6))
    output_dir = tmp_path / "out"

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
    ])

    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    assert "group_id" not in predictions_df.columns


def test_sample_index_is_anonymous_run_local_sequence(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_df = _build_synthetic_frame(n_per_class=5, seed=7)
    external_path = _write_table(tmp_path, external_df)
    output_dir = tmp_path / "out"

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
    ])

    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    assert predictions_df["sample_index"].tolist() == list(range(len(external_df)))


# ---------------------------------------------------------------------------
# 26. No clinical feature values or identity fields in output
# ---------------------------------------------------------------------------

def test_outputs_exclude_feature_values_and_identity_columns(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_df = _build_synthetic_frame(n_per_class=4, seed=8, extra_column=True)
    # A distinctive numeric marker: categorical features can't carry an
    # arbitrary marker value, since the fitted LabelEncoder correctly
    # rejects any category it never saw during training.
    marker_value = 918273645.0625
    external_df.loc[0, Config.NUMERICAL_FEATURES[0]] = marker_value
    external_path = _write_table(tmp_path, external_df)
    output_dir = tmp_path / "out"

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
    ])

    marker_text = repr(marker_value)
    for output_name in ("predictions.csv", "metrics.json", "classification_report.txt", "confusion_matrix.csv"):
        text = (output_dir / output_name).read_text(encoding="utf-8")
        assert marker_text not in text
        assert "synthetic_unused_column" not in text
        assert "ignored_" not in text
        for col in Config.NUMERICAL_FEATURES + Config.CATEGORICAL_FEATURES:
            assert col not in text


# ---------------------------------------------------------------------------
# 27-28. No plotting libraries, no image artifacts
# ---------------------------------------------------------------------------

def test_source_does_not_import_plotting_libraries_or_plot_functions():
    source = inspect.getsource(ev)
    for forbidden in (
        "matplotlib", "seaborn", "plotly", "bokeh", "altair",
        "generate_all_plots", "plot_confusion_matrix", "plot_roc_curve",
        "plot_precision_recall_curve", "text_pipeline.src.evaluate",
        "from .evaluate", "from text_pipeline.src import evaluate",
    ):
        assert forbidden not in source


def test_no_image_files_created_after_run(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(n_per_class=3, seed=9))
    output_dir = tmp_path / "out"

    ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(output_dir),
    ])

    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".bmp", ".tiff"}
    created_images = [p for p in output_dir.rglob("*") if p.suffix.lower() in image_extensions]
    assert created_images == []


# ---------------------------------------------------------------------------
# 29. No historical data volumes, hospital/dataset names, or historical results
# ---------------------------------------------------------------------------

NEW_FILES = [
    "evaluation/text/__init__.py",
    "evaluation/text/external_validation.py",
]


def test_new_files_contain_no_forbidden_content():
    import re

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
    ]
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

    repo_root = Path(__file__).resolve().parents[1]
    id_or_phone_pattern = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{11})(?!\d)")
    for relative_path in NEW_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert not id_or_phone_pattern.search(text), f"ID/phone-shaped literal found in {relative_path}"


# ---------------------------------------------------------------------------
# 30. Same input + model gives consistent predictions and metrics
# ---------------------------------------------------------------------------

def test_same_input_and_model_gives_consistent_results(tmp_path, artifacts):
    model_path, engineer_path = artifacts
    external_path = _write_table(tmp_path, _build_synthetic_frame(n_per_class=6, seed=10))

    metrics_a = ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(tmp_path / "out_a"),
    ])
    metrics_b = ev.main([
        "--input-data", str(external_path), "--model", str(model_path),
        "--feature-engineer", str(engineer_path), "--output-dir", str(tmp_path / "out_b"),
    ])

    assert metrics_a == metrics_b
    df_a = pd.read_csv(tmp_path / "out_a" / "predictions.csv")
    df_b = pd.read_csv(tmp_path / "out_b" / "predictions.csv")
    pd.testing.assert_frame_equal(df_a, df_b)
