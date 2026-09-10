"""
Tests for the M0_IMG independent external validation entry point
(evaluation.image.external_validation), using only programmatically
generated synthetic images -- solid-color small PNGs created with
Pillow's Image.new(), never real patient photos, and a stand-in model
object instead of any real checkpoint. No training or real inference
happens in this file.
"""

import inspect
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import evaluation.image.external_validation as ev
from image_pipeline.src.config import Config


CLASS_VALUES = [Config.LABEL_MAPPING[i] for i in sorted(Config.LABEL_MAPPING.keys())]


def _build_dataset(tmp_path, per_class=2, group_values=None):
    """Create tmp_path/images/*.png plus a matching labels.csv."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    idx = 0
    for class_idx in sorted(Config.LABEL_MAPPING.keys()):
        label_value = Config.LABEL_MAPPING[class_idx]
        for _ in range(per_class):
            filename = f"sample_{idx:03d}.png"
            Image.new("RGB", (8, 8), (idx % 256, 100, 150)).save(image_dir / filename)
            group_value = group_values[idx] if group_values is not None else f"subject_{idx}"
            rows.append({"image": filename, "label": label_value, "subject": group_value})
            idx += 1
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    return labels_path, image_dir


def _touch_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"")
    return checkpoint_path


class _ConstantLogitsModel:
    """Minimal stand-in for a trained model: no parameters, no checkpoint."""

    def __init__(self, logits_by_call=None, default_logits=(1.0, 0.0, 0.0)):
        self._logits_by_call = list(logits_by_call) if logits_by_call is not None else None
        self._default_logits = default_logits
        self.call_count = 0
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, x):
        assert not torch.is_grad_enabled(), "inference must run under torch.no_grad()"
        if self._logits_by_call is not None:
            logits = self._logits_by_call[self.call_count]
        else:
            logits = self._default_logits
        self.call_count += 1
        return torch.tensor([logits], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Import has no side effects
# ---------------------------------------------------------------------------

def test_module_import_has_no_side_effects():
    import importlib
    importlib.reload(ev)
    assert not (ev.THIS_DIR / "outputs").exists()


# ---------------------------------------------------------------------------
# 2. --help
# ---------------------------------------------------------------------------

def test_help_lists_all_required_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        ev.main(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    for flag in (
        "--labels-file", "--image-dir", "--checkpoint", "--output-dir",
        "--image-column", "--label-column", "--group-column", "--device", "--seed",
    ):
        assert flag in captured.out


# ---------------------------------------------------------------------------
# 3. Missing required arguments
# ---------------------------------------------------------------------------

def test_missing_required_arguments_fails_clearly():
    with pytest.raises(SystemExit) as exc_info:
        ev.main([])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 4. labels-file / image-dir / checkpoint must exist
# ---------------------------------------------------------------------------

def test_missing_labels_file_fails_clearly(tmp_path):
    _labels_path, image_dir = _build_dataset(tmp_path)
    checkpoint_path = _touch_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="labels-file"):
        ev.main([
            "--labels-file", str(tmp_path / "does_not_exist.csv"),
            "--image-dir", str(image_dir),
            "--checkpoint", str(checkpoint_path),
            "--image-column", "image", "--label-column", "label",
            "--output-dir", str(tmp_path / "out"),
        ])


def test_missing_image_dir_fails_clearly(tmp_path):
    labels_path, _image_dir = _build_dataset(tmp_path)
    checkpoint_path = _touch_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="image-dir"):
        ev.main([
            "--labels-file", str(labels_path),
            "--image-dir", str(tmp_path / "does_not_exist_dir"),
            "--checkpoint", str(checkpoint_path),
            "--image-column", "image", "--label-column", "label",
            "--output-dir", str(tmp_path / "out"),
        ])


def test_missing_checkpoint_fails_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        ev.main([
            "--labels-file", str(labels_path),
            "--image-dir", str(image_dir),
            "--checkpoint", str(tmp_path / "does_not_exist.pth"),
            "--image-column", "image", "--label-column", "label",
            "--output-dir", str(tmp_path / "out"),
        ])


# ---------------------------------------------------------------------------
# 5. Missing required columns
# ---------------------------------------------------------------------------

def test_missing_required_column_fails_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path)
    df = pd.read_csv(labels_path)
    with pytest.raises(ValueError, match="missing required column"):
        ev.validate_and_build_records(df, image_dir, "not_a_column", "label", None)


# ---------------------------------------------------------------------------
# 6. Unsupported label values
# ---------------------------------------------------------------------------

def test_unsupported_label_value_fails_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    df = pd.read_csv(labels_path)
    df.loc[0, "label"] = "not_a_real_class"
    with pytest.raises(ValueError, match="unsupported label value"):
        ev.validate_and_build_records(df, image_dir, "image", "label", None)


# ---------------------------------------------------------------------------
# 7. Missing image file
# ---------------------------------------------------------------------------

def test_missing_image_file_fails_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    df = pd.read_csv(labels_path)
    (image_dir / df.loc[0, "image"]).unlink()
    with pytest.raises(FileNotFoundError, match="not found under"):
        ev.validate_and_build_records(df, image_dir, "image", "label", None)


# ---------------------------------------------------------------------------
# 8. Unsupported image extension
# ---------------------------------------------------------------------------

def test_unsupported_extension_fails_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    df = pd.read_csv(labels_path)
    df.loc[0, "image"] = "sample.gif"
    with pytest.raises(ValueError, match="unsupported extension"):
        ev.validate_and_build_records(df, image_dir, "image", "label", None)


# ---------------------------------------------------------------------------
# Additional input-validation coverage: empty table and duplicate records
# ---------------------------------------------------------------------------

def test_empty_labels_table_fails_clearly(tmp_path):
    _labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    empty_df = pd.DataFrame(columns=["image", "label"])
    with pytest.raises(ValueError, match="no records"):
        ev.validate_and_build_records(empty_df, image_dir, "image", "label", None)


def test_duplicate_image_records_fail_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    df = pd.read_csv(labels_path)
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ev.validate_and_build_records(duplicated, image_dir, "image", "label", None)


def test_missing_label_value_fails_clearly(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    df = pd.read_csv(labels_path)
    df.loc[0, "label"] = None
    with pytest.raises(ValueError, match="missing values"):
        ev.validate_and_build_records(df, image_dir, "image", "label", None)


# ---------------------------------------------------------------------------
# 9. Validation transform has no random augmentation
# ---------------------------------------------------------------------------

def test_validation_transform_has_no_random_augmentation():
    from image_pipeline.src.data_utils import create_data_transforms
    _train_transform, val_transform = create_data_transforms(Config.IMAGE_SIZE)
    for step in val_transform.transforms:
        assert "Random" not in type(step).__name__


# ---------------------------------------------------------------------------
# 10. Inference uses model.eval() and torch.no_grad()
# ---------------------------------------------------------------------------

def test_inference_calls_eval_and_runs_under_no_grad(tmp_path):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    df = pd.read_csv(labels_path)
    records = ev.validate_and_build_records(df, image_dir, "image", "label", None)

    from image_pipeline.src.data_utils import create_data_transforms
    _train_transform, val_transform = create_data_transforms(Config.IMAGE_SIZE)

    model = _ConstantLogitsModel()
    ev.run_inference(model, torch.device("cpu"), records, val_transform)

    assert model.eval_called is True
    assert model.call_count == len(records)


# ---------------------------------------------------------------------------
# 11. Checkpoint loading uses strict=True (real load_model contract)
# ---------------------------------------------------------------------------

def test_checkpoint_loading_uses_strict_state_dict_matching(tmp_path):
    from image_pipeline.src.models import create_model
    from image_pipeline.src.train import load_model

    model = create_model(num_classes=3, input_size=224, pretrained=False)
    good_checkpoint_path = tmp_path / "good.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 0,
        "best_val_acc": 0.5,
    }, good_checkpoint_path)

    reloaded_model = create_model(num_classes=3, input_size=224, pretrained=False)
    reloaded_model, _checkpoint = load_model(reloaded_model, str(good_checkpoint_path), torch.device("cpu"))
    assert reloaded_model is not None

    bad_state_dict = dict(model.state_dict())
    bad_state_dict["unexpected_extra_key"] = torch.zeros(1)
    del bad_state_dict[next(iter(model.state_dict().keys()))]
    bad_checkpoint_path = tmp_path / "bad.pth"
    torch.save({
        "model_state_dict": bad_state_dict,
        "optimizer_state_dict": {},
        "epoch": 0,
        "best_val_acc": 0.5,
    }, bad_checkpoint_path)

    mismatched_model = create_model(num_classes=3, input_size=224, pretrained=False)
    with pytest.raises(RuntimeError):
        load_model(mismatched_model, str(bad_checkpoint_path), torch.device("cpu"))


# ---------------------------------------------------------------------------
# 12. pretrained=False; no ImageNet download during evaluation
# ---------------------------------------------------------------------------

def test_model_is_created_with_pretrained_false():
    source = inspect.getsource(ev.main)
    assert "pretrained=False" in source


def test_create_model_pretrained_false_requests_no_weights(monkeypatch):
    from image_pipeline.src import models as models_module

    captured = {}

    def fake_resnet50(*, weights=None, **kwargs):
        import torch.nn as nn
        captured["weights"] = weights
        return nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
            nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
            nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
            nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(4, 1000),
        )

    monkeypatch.setattr(models_module.models, "resnet50", fake_resnet50)
    models_module.create_model(num_classes=3, input_size=224, pretrained=False)
    assert captured["weights"] is None


# ---------------------------------------------------------------------------
# 13 & 14. predictions.csv contains only anonymous fields
# ---------------------------------------------------------------------------

def _run_end_to_end(tmp_path, monkeypatch, per_class=2, group_column=None, group_values=None, seed=42):
    labels_path, image_dir = _build_dataset(tmp_path, per_class=per_class, group_values=group_values)
    checkpoint_path = _touch_checkpoint(tmp_path)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(ev, "create_model", lambda **kwargs: _ConstantLogitsModel())
    monkeypatch.setattr(ev, "load_model", lambda model, path, device: (model, {}))

    args = [
        "--labels-file", str(labels_path),
        "--image-dir", str(image_dir),
        "--checkpoint", str(checkpoint_path),
        "--image-column", "image", "--label-column", "label",
        "--output-dir", str(output_dir),
        "--device", "cpu",
        "--seed", str(seed),
    ]
    if group_column is not None:
        args += ["--group-column", group_column]

    metrics = ev.main(args)
    return output_dir, metrics


def test_predictions_csv_excludes_raw_filenames_and_paths(tmp_path, monkeypatch):
    output_dir, _metrics = _run_end_to_end(tmp_path, monkeypatch)

    predictions_path = output_dir / "predictions.csv"
    assert predictions_path.is_file()
    predictions_text = predictions_path.read_text(encoding="utf-8")

    assert list(pd.read_csv(predictions_path).columns[:3]) == [
        "sample_index", "true_label", "predicted_label",
    ]
    assert ".png" not in predictions_text
    assert "sample_000.png" not in predictions_text
    assert str(tmp_path) not in predictions_text


def test_predictions_csv_group_id_is_anonymous(tmp_path, monkeypatch):
    group_values = ["real_subject_alpha", "real_subject_alpha", "real_subject_beta",
                     "real_subject_beta", "real_subject_gamma", "real_subject_gamma"]
    output_dir, _metrics = _run_end_to_end(
        tmp_path, monkeypatch, per_class=2, group_column="subject", group_values=group_values
    )

    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    assert "group_id" in predictions_df.columns
    assert all(str(g).startswith("group_") for g in predictions_df["group_id"])

    predictions_text = (output_dir / "predictions.csv").read_text(encoding="utf-8")
    for raw_value in set(group_values):
        assert raw_value not in predictions_text

    # Two rows sharing a real subject must share the same anonymous group_id.
    assert predictions_df.loc[0, "group_id"] == predictions_df.loc[1, "group_id"]
    assert predictions_df.loc[0, "group_id"] != predictions_df.loc[2, "group_id"]


def test_predictions_csv_has_no_group_id_column_without_group_column(tmp_path, monkeypatch):
    output_dir, _metrics = _run_end_to_end(tmp_path, monkeypatch, group_column=None)
    predictions_df = pd.read_csv(output_dir / "predictions.csv")
    assert "group_id" not in predictions_df.columns


# ---------------------------------------------------------------------------
# 15. Metrics match a hand-checkable small prediction case
# ---------------------------------------------------------------------------

def test_metrics_match_hand_checkable_perfect_case(tmp_path, monkeypatch):
    logits_for_class = {0: (10.0, 0.0, 0.0), 1: (0.0, 10.0, 0.0), 2: (0.0, 0.0, 10.0)}
    logits_sequence = [logits_for_class[c] for c in sorted(Config.LABEL_MAPPING.keys())]

    labels_path, image_dir = _build_dataset(tmp_path, per_class=1)
    checkpoint_path = _touch_checkpoint(tmp_path)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(ev, "create_model", lambda **kwargs: _ConstantLogitsModel(logits_by_call=logits_sequence))
    monkeypatch.setattr(ev, "load_model", lambda model, path, device: (model, {}))

    metrics = ev.main([
        "--labels-file", str(labels_path), "--image-dir", str(image_dir),
        "--checkpoint", str(checkpoint_path),
        "--image-column", "image", "--label-column", "label",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_metrics_match_hand_checkable_one_misclassification(tmp_path, monkeypatch):
    # True labels in file order: class0, class0, class1, class1, class2, class2.
    # Force the first class0 sample to be predicted as class1.
    logits_sequence = [
        (0.0, 10.0, 0.0),   # true 0 -> predicted 1 (misclassified)
        (10.0, 0.0, 0.0),   # true 0 -> predicted 0
        (0.0, 10.0, 0.0),   # true 1 -> predicted 1
        (0.0, 10.0, 0.0),   # true 1 -> predicted 1
        (0.0, 0.0, 10.0),   # true 2 -> predicted 2
        (0.0, 0.0, 10.0),   # true 2 -> predicted 2
    ]
    labels_path, image_dir = _build_dataset(tmp_path, per_class=2)
    checkpoint_path = _touch_checkpoint(tmp_path)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(ev, "create_model", lambda **kwargs: _ConstantLogitsModel(logits_by_call=logits_sequence))
    monkeypatch.setattr(ev, "load_model", lambda model, path, device: (model, {}))

    metrics = ev.main([
        "--labels-file", str(labels_path), "--image-dir", str(image_dir),
        "--checkpoint", str(checkpoint_path),
        "--image-column", "image", "--label-column", "label",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert metrics["n_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]


# ---------------------------------------------------------------------------
# 16. Fixed seed gives reproducible output order and anonymous group_id
# ---------------------------------------------------------------------------

def test_fixed_seed_gives_reproducible_output(tmp_path, monkeypatch):
    group_values = ["subj_a", "subj_a", "subj_b", "subj_b", "subj_c", "subj_c"]
    labels_path, image_dir = _build_dataset(tmp_path, per_class=2, group_values=group_values)
    checkpoint_path = _touch_checkpoint(tmp_path)

    monkeypatch.setattr(ev, "create_model", lambda **kwargs: _ConstantLogitsModel())
    monkeypatch.setattr(ev, "load_model", lambda model, path, device: (model, {}))

    def _run(output_dir):
        return ev.main([
            "--labels-file", str(labels_path), "--image-dir", str(image_dir),
            "--checkpoint", str(checkpoint_path),
            "--image-column", "image", "--label-column", "label",
            "--group-column", "subject",
            "--output-dir", str(output_dir), "--device", "cpu", "--seed", "123",
        ])

    _run(tmp_path / "out_a")
    _run(tmp_path / "out_b")

    df_a = pd.read_csv(tmp_path / "out_a" / "predictions.csv")
    df_b = pd.read_csv(tmp_path / "out_b" / "predictions.csv")
    pd.testing.assert_frame_equal(df_a, df_b)


# ---------------------------------------------------------------------------
# 17. No personal paths, hospital names, secrets, or CJK characters
# ---------------------------------------------------------------------------

NEW_FILES = [
    "evaluation/__init__.py",
    "evaluation/image/__init__.py",
    "evaluation/image/external_validation.py",
    "evaluation/requirements.txt",
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
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    id_or_phone_pattern = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{11})(?!\d)")
    for relative_path in NEW_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert not id_or_phone_pattern.search(text), f"ID/phone-shaped literal found in {relative_path}"
