# Evaluation Module

Standalone command-line tools for scoring already-trained M0_IMG (image)
and M0_CLIN (text/tabular) models against external data, and for computing
bootstrap confidence intervals from their prediction outputs.

Every tool in this module is read-only with respect to model artifacts: it
loads an already-trained checkpoint/model and an already-fitted
preprocessor, and only ever calls their inference/transform interfaces
(`model.eval()` + no-gradient inference for the image model;
`FeatureEngineer.transform()` and `SurgeryClassifier.predict()` /
`predict_proba()` for the text model). No tool in this module trains,
fits, fine-tunes, resamples, or re-splits any dataset, and none of them
generate plots or figures.

This module ships no real data, no model weights, and no prediction
results. Every input path is supplied by the caller via command-line
arguments.

## Contents

```
evaluation/
├── requirements.txt
├── image/
│   ├── external_validation.py
│   └── bootstrap_ci.py
└── text/
    ├── external_validation.py
    └── bootstrap_ci.py
```

## Installation

```bash
pip install -r evaluation/requirements.txt
```

This inherits both `image_pipeline/requirements.txt` and
`text_pipeline/requirements.txt`, since the tools below import and reuse
each pipeline's own model, config, and data-loading code directly.

---

## `image/external_validation.py`

Independent external validation for the M0_IMG image classification
model. Evaluates every valid record in a labels file against a directory
of images, using an already-trained checkpoint. No training, fitting, or
re-splitting of the external data is performed.

### Input requirements

- A labels table (`.csv`, `.xlsx`, or `.xls`) with one row per image, an
  image-path column, and a label column.
- A directory of images. Image-column values are resolved as paths
  relative to this directory.
- A model checkpoint (`.pth`) saved by the M0_IMG training pipeline (a
  dict containing `model_state_dict`).

### CLI usage

```bash
python -m evaluation.image.external_validation \
    --labels-file /path/to/labels.csv \
    --image-dir /path/to/images \
    --checkpoint /path/to/best_model.pth \
    --image-column image_path \
    --label-column severity \
    --output-dir /path/to/output \
    [--group-column subject_id] \
    [--device auto|cpu|cuda] \
    [--seed SEED]
```

| Argument | Required | Description |
|---|---|---|
| `--labels-file` | yes | Path to a labels table (`.csv`, `.xlsx`, or `.xls`) listing one row per image. |
| `--image-dir` | yes | Directory containing the external images. Image-column values are resolved as paths relative to this directory. |
| `--checkpoint` | yes | Path to a model checkpoint (`.pth`) saved by the M0_IMG training pipeline (a dict containing `model_state_dict`). |
| `--output-dir` | no | Directory to write validation outputs to. Created only when the program runs (default: `<this package>/outputs`). |
| `--image-column` | yes | Name of the labels-file column containing each image's path relative to `--image-dir`. |
| `--label-column` | yes | Name of the labels-file column containing the ground-truth class label for each image. |
| `--group-column` | no | Optional name of a labels-file column identifying which records belong to the same subject/cluster (e.g. for later grouped resampling). If given, `predictions.csv` includes an anonymous `group_id` column (sequential `group_NNNNNN` labels assigned in first-seen order; original values are never written out and no mapping file is produced). If omitted, `predictions.csv` has no `group_id` column at all. |
| `--device` | no | Device selection: `auto` picks `cuda` if available else `cpu`; `cpu` forces CPU; `cuda` fails loudly if unavailable (default: `auto`). |
| `--seed` | no | Random seed for reproducibility (default: the image pipeline's configured `RANDOM_STATE`). |

### Output files

Written to `--output-dir` only when the program runs:

- `metrics.json` — accuracy, macro/weighted precision/recall/F1, per-class
  precision/recall/F1/support, confusion matrix, class names.
- `classification_report.txt` — text classification report.
- `confusion_matrix.csv`.
- `predictions.csv` — `sample_index`, `true_label`, `predicted_label`, one
  `prob_class_<i>` column per class, and an optional anonymous `group_id`.

---

## `image/bootstrap_ci.py`

Computes percentile bootstrap confidence intervals for multiclass
classification metrics from a predictions CSV, such as the one produced
by `image/external_validation.py`. Reads only the given predictions file;
performs no inference, training, or fitting.

### Input requirements

A predictions CSV containing `sample_index`, `true_label`,
`predicted_label`, and `prob_class_0`, `prob_class_1`, ... probability
columns (at least 3 contiguous classes starting at 0). A `group_id`-style
column is required when using the default group-level resampling (see
below).

### CLI usage

```bash
python -m evaluation.image.bootstrap_ci \
    --predictions-file /path/to/predictions.csv \
    --output-dir /path/to/output \
    [--n-bootstrap 10000] \
    [--confidence-level 0.95] \
    [--seed 20260909] \
    [--group-column group_id] \
    [--resampling-unit group|sample]
```

| Argument | Required | Description |
|---|---|---|
| `--predictions-file` | yes | Path to a predictions CSV containing `sample_index`, `true_label`, `predicted_label`, and `prob_class_0..` probability columns (such as the output of `evaluation.image.external_validation`). |
| `--output-dir` | no | Directory to write bootstrap outputs to. Created only when the program runs (default: `<evaluation package>/outputs/bootstrap_ci`). |
| `--n-bootstrap` | no | Number of bootstrap resamples (default: `10000`). |
| `--confidence-level` | no | Confidence level for the percentile interval, in `(0, 1)` (default: `0.95`). |
| `--seed` | no | Seed for the NumPy random generator used for resampling (default: `20260909`). |
| `--group-column` | no | Name of the predictions-file column identifying which records belong to the same resampling cluster. Only used when `--resampling-unit` is `group` (default: `group_id`). |
| `--resampling-unit` | no | `group` (default) or `sample`. See below. |

### Group-level cluster bootstrap (default)

By default, resampling draws whole clusters (identified by
`--group-column`) with replacement — every record belonging to a drawn
cluster is kept together, and a cluster drawn more than once contributes
its full record set again for each draw. If `--group-column` is missing
or contains missing/empty values, the tool raises an error rather than
falling back to sample-level resampling.

### Sample-level bootstrap (explicit opt-in only)

Record-level (rather than cluster-level) resampling is only performed
when `--resampling-unit sample` is passed explicitly. It is never
selected automatically.

### Metrics and NaN handling

Accuracy, macro/weighted precision/recall/F1, per-class precision/recall/
F1, and per-class one-vs-rest AUC/AP. If a metric cannot be computed for a
given resample (e.g. a class is absent from that resample), it is
recorded as NaN and excluded from that metric's confidence interval; the
number of valid and invalid resamples is reported per metric.

### Output files

Only these two files are written, to `--output-dir`:

- `bootstrap_ci.csv` — one row per metric: `metric`, `point_estimate`,
  `confidence_level`, `ci_lower`, `ci_upper`, `n_bootstrap_requested`,
  `n_bootstrap_valid`, `n_bootstrap_invalid`, `seed`, `resampling_unit`.
- `bootstrap_summary.json` — run configuration, point estimates,
  confidence intervals, valid/invalid counts, and the input/output file
  SHA-256 hashes.

No plots, figures, or other files are produced.

---

## `text/external_validation.py`

Independent external validation for the M0_CLIN tabular classification
model. Evaluates every valid record in an input data table using an
already-fitted `FeatureEngineer` and an already-trained model. No
fitting, training, resampling, or re-splitting of the external data is
performed.

### Input requirements

- An input data table (`.csv`, `.xlsx`, or `.xls`) containing the model's
  required feature columns and target column.
- An already-trained model file, saved by `SurgeryClassifier.save()`.
- An already-fitted `FeatureEngineer` file, saved by
  `FeatureEngineer.save()`.

### CLI usage

```bash
python -m evaluation.text.external_validation \
    --input-data /path/to/external_data.csv \
    --model /path/to/model.pkl \
    --feature-engineer /path/to/feature_engineer.pkl \
    --output-dir /path/to/output \
    [--group-column subject_id] \
    [--seed SEED]
```

| Argument | Required | Description |
|---|---|---|
| `--input-data` | yes | Path to an external data table (`.csv`, `.xlsx`, or `.xls`) containing the model's required feature columns and target column. |
| `--model` | yes | Path to an already-trained model file, saved by `SurgeryClassifier.save()`. |
| `--feature-engineer` | yes | Path to an already-fitted `FeatureEngineer` file, saved by `FeatureEngineer.save()`. |
| `--output-dir` | no | Directory to write validation outputs to. Created only when the program runs (default: `<evaluation package>/outputs/text_external_validation`). |
| `--group-column` | no | Optional name of an `--input-data` column identifying which records belong to the same subject/cluster. If given, `predictions.csv` includes an anonymous `group_id` column (sequential `group_NNNNNN` labels assigned in first-seen order; original values are never written out and no mapping file is produced). If omitted, `predictions.csv` has no `group_id` column. |
| `--seed` | no | Random seed for reproducibility (default: `42`). |

### Output files

Written to `--output-dir` only when the program runs:

- `metrics.json` — accuracy, precision, recall, F1, ROC-AUC, average
  precision, confusion matrix, label names, seed, and the input/model/
  feature-engineer file SHA-256 hashes.
- `classification_report.txt`.
- `confusion_matrix.csv`.
- `predictions.csv` — `sample_index`, `true_label`, `predicted_label`,
  `prob_class_0`, `prob_class_1`, and an optional anonymous `group_id`.

---

## `text/bootstrap_ci.py`

Computes percentile bootstrap confidence intervals for binary
classification metrics from a predictions CSV, such as the one produced
by `text/external_validation.py`. Reads only the given predictions file;
performs no inference, training, or fitting.

### Input requirements

A predictions CSV containing `sample_index`, `true_label`,
`predicted_label`, `prob_class_0`, and `prob_class_1` columns. A
`group_id`-style column is required when using the default group-level
resampling (see below).

### CLI usage

```bash
python -m evaluation.text.bootstrap_ci \
    --predictions-file /path/to/predictions.csv \
    --output-dir /path/to/output \
    [--n-bootstrap 10000] \
    [--confidence-level 0.95] \
    [--seed 20260909] \
    [--group-column group_id] \
    [--resampling-unit group|sample]
```

| Argument | Required | Description |
|---|---|---|
| `--predictions-file` | yes | Path to a predictions CSV containing `sample_index`, `true_label`, `predicted_label`, `prob_class_0`, and `prob_class_1` columns (such as the output of `evaluation.text.external_validation`). |
| `--output-dir` | no | Directory to write bootstrap outputs to. Created only when the program runs (default: `<evaluation package>/outputs/text_bootstrap_ci`). |
| `--n-bootstrap` | no | Number of bootstrap resamples (default: `10000`). |
| `--confidence-level` | no | Confidence level for the percentile interval, in `(0, 1)` (default: `0.95`). |
| `--seed` | no | Seed for the NumPy random generator used for resampling (default: `20260909`). |
| `--group-column` | no | Name of the predictions-file column identifying which records belong to the same resampling cluster. Only used when `--resampling-unit` is `group` (default: `group_id`). |
| `--resampling-unit` | no | `group` (default) or `sample`. See below. |

### Group-level cluster bootstrap (default)

Identical resampling contract to `image/bootstrap_ci.py`: whole clusters
(identified by `--group-column`) are drawn with replacement, keeping
every record of a drawn cluster together. A missing or incomplete
`--group-column` raises an error rather than silently falling back to
sample-level resampling.

### Sample-level bootstrap (explicit opt-in only)

Only performed when `--resampling-unit sample` is passed explicitly;
never selected automatically.

### Metrics and NaN handling

Accuracy, precision, recall, F1, ROC-AUC, and average precision, with the
positive class fixed to class `1`. If ROC-AUC or average precision cannot
be computed for a given resample (e.g. only one true class is present in
that resample), it is recorded as NaN and excluded from that metric's
confidence interval; the number of valid and invalid resamples is
reported per metric.

### Output files

Only these two files are written, to `--output-dir`:

- `bootstrap_ci.csv` — one row per metric, with the same columns as
  `image/bootstrap_ci.py`'s output.
- `bootstrap_summary.json` — run configuration, point estimates,
  confidence intervals, valid/invalid counts, and the input/output file
  SHA-256 hashes.

No plots, figures, or other files are produced.

---

## Scope

This module contains only the four tools described above. It does not
include statistical comparison tests (e.g. DeLong, McNemar), calibration
analysis, decision curve analysis, stratified or sensitivity analyses, or
any plotting/figure-generation code.
