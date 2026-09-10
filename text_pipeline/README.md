# M0_CLIN: XGBoost Tabular Text-Pipeline (Surgery-Need Classifier)

## 1. Purpose

This module trains and evaluates a binary XGBoost classifier that predicts,
from a small set of tabular clinical fields, whether a patient's record
indicates a need for surgery. It is the canonical M0_CLIN training pipeline
in this repository: data loading → train/test split → preprocessing →
training → prediction → evaluation → artifact saving, driven entirely by a
command-line interface.

No real patient data ships in this repository (see [Section 15](#15-real-patient-data-is-not-included)).
This README describes only what the code in `text_pipeline/src` actually
does — it does not claim any clinical performance, validation status, or
regulatory standing.

## 2. File Structure

```
text_pipeline/
  README.md                  This file
  requirements.txt            Runtime dependencies actually imported by src/
  requirements-dev.txt        Adds test-only dependencies (pytest) on top of requirements.txt
  src/
    __init__.py               Package init; UTF-8 stdout/stderr fix; re-exports
    config.py                 Config: paths, canonical feature set, hyperparameters, split/seed
    data_loader.py             DataLoader: .csv/.xlsx/.xls ingestion, whitelist, cleaning, target encoding
    feature_engineering.py     FeatureEngineer: imputers, StandardScaler, LabelEncoders
    models.py                  SurgeryClassifier: XGBoost (+ RandomForest/LogisticRegression) wrapper
    train.py                   ModelTrainer: split, SMOTE, class weights, fit, leakage-free CV
    evaluate.py                 ModelEvaluator: metrics, confusion matrix, ROC/PR, plots, reports
    main.py                     CLI entry point (train / internal-eval / external-eval)
  configs/, scripts/          Reserved for future use (currently empty)
  outputs/                    Default output directory (created at runtime; gitignored)
```

## 3. Required Input Table Fields

The loader whitelists exactly `Config.FEATURE_COLUMNS + [Config.TARGET_COLUMN]`
from each input file (`.csv`, `.xlsx`, or `.xls` — dispatched by extension,
see [Section 13](#13-supported-file-formats)); every other column is
dropped. `Config.FEATURE_COLUMNS` is, by design, exactly the 5 features
that reach the model (there is no separate "nominal vs. actually-used"
feature list anymore — see [Section 4](#4-model-features-used)). Your input
table should contain:

| Column | Type | Role |
|---|---|---|
| `脱位程度` | categorical (text) | used feature |
| `矫正视力` | numeric | used feature |
| `矫正球镜度数(D)` | numeric | used feature |
| `矫正柱镜度数(D)` | numeric | used feature |
| `IOLMaster-Cyl(D)` | numeric | used feature |
| `是否需要手术` | text, one of `手术` / `不手术` | target column |

**Backward-compatible extra columns.** Older raw exports may still contain
`是否配合` (cooperation) or `年龄` (age) columns, or other metadata. These
never entered the historical canonical model and must not be added back;
an input file MAY still contain them, but `DataLoader`'s whitelist silently
drops any column that is not in `FEATURE_COLUMNS`/`PRESERVE_COLUMNS`, so
they are guaranteed to never reach the model. This tolerance is covered by
a synthetic, in-test-only DataFrame in
`tests/test_text_pipeline_input_validation.py::test_legacy_columns_tolerated_but_never_reach_the_model`;
the public example files under `examples/synthetic_tabular_data/` contain
only the columns actually needed (`sample_id` + the 5 canonical features +
target) and do not include these legacy columns.

Optionally, `姓名` / `身份证号` / `省份` (`Config.PRESERVE_COLUMNS`) may be
present in the raw file — `DataLoader` splits these off into a separate,
in-memory-only `preserved_data` frame and they are **never** written to any
model input, saved artifact, or evaluation output. Do not add any other
identifying column (phone, address, medical record number, patient ID,
etc.); the whitelist already drops anything not listed above, but avoid
including identifying data in source files at all.

Numeric columns tolerate the raw-text artifacts real optometry exports
sometimes contain (a trailing `+`/`-`, or a full-width minus `−`); these are
stripped and the value coerced to `float`, with unparseable values becoming
`NaN` (`DataLoader.clean_numeric_columns`).

## 4. Model Features Used

`Config.FEATURE_COLUMNS` = `Config.CATEGORICAL_FEATURES` + `Config.NUMERICAL_FEATURES`,
exactly the 5 columns that reach the trained model — this is enforced by a
regression test (`tests/test_text_pipeline_input_validation.py::test_canonical_feature_columns_are_exactly_the_five_used_features`):

- Categorical (see [Section 5](#5-categorical-missing-value-handling-historical-behaviour) for how missingness is actually handled, then label-encoded): `脱位程度`
- Numerical (median-imputed, then standardized): `矫正视力`, `矫正球镜度数(D)`, `矫正柱镜度数(D)`, `IOLMaster-Cyl(D)`

`是否配合` and `年龄` are **not** in `FEATURE_COLUMNS` and must not be added
back — they never entered the historical canonical model (see
[Section 3](#3-required-input-table-fields)).

## 5. Categorical Missing-Value Handling (Historical Behaviour)

**This is intentionally preserved, non-"best-practice" behaviour — read
this before assuming `most_frequent` imputation is happening.**

`DataLoader.clean_categorical_columns()` / `_clean_dataframe()` casts the
categorical column to `str` (to standardize/strip values) *before*
`FeatureEngineer`'s categorical `SimpleImputer(strategy='most_frequent')`
ever runs. That cast turns a real missing value (`NaN`) into the literal
string `"nan"`. By the time the categorical imputer sees the data, there is
no actual `NaN` left to impute — `"nan"` is simply treated as its own
explicit category and label-encoded as such by `LabelEncoder`.

In other words: **categorical missing values are not most-frequent-imputed
in practice; they become an explicit "missing" category.** The
`SimpleImputer` object (`Config.FILL_STRATEGY['categorical'] = 'most_frequent'`)
is retained in the code for structural/pickle compatibility with the
existing saved pipeline shape — not because it is doing effective
imputation. This pass does not reorder the cleaning steps to make the
imputer effective, because that would change what the model is trained on
(a real, if subtle, change in categorical missing-value semantics).
`tests/test_text_pipeline_categorical_missing_regression.py` pins this
behaviour as a regression test.

## 6. Target Variable

`是否需要手术`, encoded via `Config.LABEL_MAPPING`: `不手术` → `0`, `手术` → `1`.
Rows with a missing/unmapped target are dropped before training.

## 7. Preprocessing Order (as executed)

1. **Whitelist columns** — keep only `FEATURE_COLUMNS + [TARGET_COLUMN]` (plus, separately, any `PRESERVE_COLUMNS` present). Any other column (legacy `是否配合`/`年龄`, identifiers, free-form metadata) is dropped here and never seen again.
2. **Clean numerical columns** — strip trailing `+`/`-`/full-width minus, coerce to `float`; rows where *every* numerical column is missing are dropped.
3. **Clean categorical columns** — cast to `str` and strip whitespace (this is what turns real missingness into the literal string `"nan"` — see [Section 5](#5-categorical-missing-value-handling-historical-behaviour)).
4. **Drop rows with a missing target**, then **encode the target** via `LABEL_MAPPING`.
5. **Split 80/20** (`train_test_split`, stratified on the target, `Config.TEST_SIZE=0.2`, `Config.RANDOM_STATE=42` by default) — this happens on the *cleaned, pre-feature-engineering* data.
6. **Fit `FeatureEngineer` on the 80% training partition only**: `SimpleImputer(strategy='median')` for numeric columns, `StandardScaler` fit on the imputed numeric values, the (in-practice-inert, see Section 5) categorical `SimpleImputer` + `LabelEncoder` per categorical column.
7. **Transform-only** the internal 20% test set and the external validation set with that same fitted `FeatureEngineer` — it is never refit on them.
8. **SMOTE** (`imblearn.over_sampling.SMOTE`, `sampling_strategy='auto'`, `k_neighbors=5`) is fit and applied to the 80% training partition only.
9. **Class weights** (`sklearn.utils.class_weight.compute_class_weight('balanced', ...)`) are computed on the *SMOTE-resampled* training labels and converted to a single `scale_pos_weight` passed to XGBoost.
10. **Train** XGBoost on the resampled 80% training partition, for the fixed `n_estimators=200`, with no early stopping (see [Section 9](#9-training-is-fixed-no-early-stopping-no-automatic-tuning)).

## 8. How Data Leakage Is Avoided

- The 80/20 split happens **before** any preprocessing object is created — `FeatureEngineer`, `SimpleImputer`, `StandardScaler`, `LabelEncoder`, and `SMOTE` are all fit exclusively on the 80% training partition.
- The internal 20% test set and the external validation file are only ever passed through `FeatureEngineer.transform()` (never `.fit()` / `.fit_transform()`).
- 5-fold cross-validation (`ModelTrainer.cross_validate_pipeline`, diagnostic only — it does not select the final model or its hyperparameters) refits a **fresh** `FeatureEngineer` and re-applies SMOTE independently inside each fold, on that fold's training rows only, then scores the fold's held-out rows with a fresh model. No fold's held-out rows ever influence that fold's fit.
- `--mode internal-eval` and `--mode external-eval` (see [Section 11](#11-internal-evaluation)/[12](#12-external-evaluation)) load the saved `FeatureEngineer` and model and only ever call `.transform()` / `.predict()` — they never fit or refit anything.
- The internal 20% test set is also passed to XGBoost's `fit(..., eval_set=...)` for per-round metric logging only — no early stopping is configured (see next section), so this does not influence which iteration's model is kept; training always runs the full `n_estimators=200`.
- Automated leakage-guard tests live in `tests/test_text_pipeline_leakage_guard.py`.

## 9. Training Is Fixed: No Early Stopping, No Automatic Tuning

The canonical model is trained **once per run**, with a single fixed
`xgboost.XGBClassifier` hyperparameter set — `Config.XGBOOST_PARAMS`:
`max_depth=6`, `learning_rate=0.05`, `n_estimators=200`,
`min_child_weight=3`, `gamma=0.1`, `subsample=0.8`, `colsample_bytree=0.8`,
`reg_alpha=0.05`, `reg_lambda=1.0`, plus `scale_pos_weight` computed as
described in [Section 7](#7-preprocessing-order-as-executed), and
`Config.RANDOM_STATE=42` (overridable via `--seed`, propagated consistently
to the split, SMOTE, and the model).

- **No early stopping.** An `EARLY_STOPPING_ROUNDS` setting previously
  existed in `Config` but was never actually wired into
  `SurgeryClassifier`'s estimator construction or its `.fit()` call — it
  had no runtime effect, so it has been removed rather than fixed or wired
  up. Training always runs the full `n_estimators=200`.
- **No automatic hyperparameter search.** `Config` still defines a
  `RandomizedSearchCV`-based helper (`ModelTrainer.hyperparameter_tuning()`,
  `PARAM_GRID`, `RANDOM_SEARCH_PARAMS`) for manual/offline experimentation,
  but `Config.ENABLE_HYPERPARAMETER_TUNING` defaults to `False` and the CLI
  (`main.py`) always calls `train_pipeline(..., hyperparameter_tuning=False)`.
  **The canonical model reported by this pipeline has never been produced
  by this pipeline's own automatic hyperparameter search** — its
  hyperparameters are the fixed values above. Do not enable
  `ENABLE_HYPERPARAMETER_TUNING` or call `hyperparameter_tuning()` as part
  of the canonical public pipeline; it exists only as a separate,
  non-canonical helper.
- `tests/test_text_pipeline_fixed_hyperparameters.py` pins all of the above
  as regression tests.

## 10. Feature-Set History (What Changed and Why)

An earlier pass found that `Config.FEATURE_COLUMNS` nominally listed 7
columns (including `是否配合`/`年龄`) while the model only ever used 5, due
to a naming mismatch between `FEATURE_COLUMNS` and
`CATEGORICAL_FEATURES`/`NUMERICAL_FEATURES`. This has since been resolved
by making `FEATURE_COLUMNS` equal to the 5 real features (this is a config
correction, not a change to the model's actual training data, algorithm,
or results — the 5 features, their preprocessing, and their order are
byte-for-byte the same as before). See [Section 4](#4-model-features-used).

## 11. Internal Evaluation

`--mode internal-eval` re-scores an already-trained model on the internal
20% test partition **without retraining or refitting anything**. It
reloads the input file given by `--input-data`, reproduces the identical
80/20 split (same file, same `Config.TEST_SIZE`, same `--seed`), and
transforms the reconstructed 20% partition with the `FeatureEngineer`
saved under `<output-dir>/models/feature_engineer.pkl`, then predicts with
the model saved under `<output-dir>/models/`. **The `--seed` must match the
one used for the original `--mode train` run**, or the reconstructed
partition will not be the model's actual held-out set.

## 12. External Evaluation

`--mode external-eval` scores an already-trained model against a new file
(`--input-data`) that was never used for training. It loads the saved
`FeatureEngineer` and model and only ever calls `.transform()` / `.predict()`
— it never fits or refits any imputer, scaler, encoder, or the model
itself, satisfying "external data may only use the preprocessor/model
artifacts saved at training time."

## 13. Supported File Formats

`DataLoader` dispatches on file extension:

- `.csv` → `pandas.read_csv(..., encoding='utf-8-sig')` (reads correctly whether or not the file has a UTF-8 BOM, e.g. from Excel's "CSV UTF-8" save option)
- `.xlsx` / `.xls` → `pandas.read_excel(...)` (unchanged from before)

Both formats are read into an identical in-memory `DataFrame` shape and
then go through the exact same preprocessing/training code — the file
format has no effect on model training. The public examples under
`examples/synthetic_tabular_data/` are `.csv` (the repository's root
`.gitignore` excludes `*.xlsx` but allow-lists `examples/**/*.csv`); `.xlsx`/`.xls`
remain fully supported for your own data.

## 14. Saved Artifacts

Under `<output-dir>/models/` (default `text_pipeline/outputs/models/`):

- `feature_engineer.pkl` — pickled dict with the fitted numeric imputer, categorical imputer, `StandardScaler`, per-column `LabelEncoder`s, feature name list, and the `Config` used.
- `best_xgboost_model.pkl` — pickled dict with the fitted `XGBClassifier`, model type, feature names, feature importances, and the `Config` used.
- `training_history.pkl` — training time, sample counts, SMOTE/class-weight flags, and (if `--mode train`) the leakage-free cross-validation results.

Under `<output-dir>/eval_internal/` and `<output-dir>/eval_external/`:
`classification_report.txt` and evaluation plots (confusion matrix, ROC
curve, precision-recall curve, feature importance, metrics comparison).
None of these outputs include per-row predictions, raw feature values, or
any `PRESERVE_COLUMNS` identity data — only aggregate metrics and plots.

## 15. Installing Dependencies

Runtime only (train/evaluate):

```bash
pip install -r text_pipeline/requirements.txt
```

Runtime + test suite (`tests/` under the repository root):

```bash
pip install -r text_pipeline/requirements.txt
pip install -r text_pipeline/requirements-dev.txt
```

## 16. Running the Pipeline

All paths default to files inside this repository and are resolved with
`pathlib.Path`, so the same commands work on Windows and Linux. Run from
the repository root.

**With the bundled synthetic example data** (see
[`examples/synthetic_tabular_data/README.md`](../examples/synthetic_tabular_data/README.md)) —
the defaults already point here, so no flags are required:

```bash
python -m text_pipeline.src.main --mode train
```

**With your own compliant data** (never real patient data committed to this
repo — see [Section 17](#17-real-patient-data-is-not-included); `.csv`,
`.xlsx`, or `.xls` all work):

```bash
python -m text_pipeline.src.main \
  --mode train \
  --input-data /path/to/your_train.csv \
  --external-data /path/to/your_external_validation.csv \
  --output-dir /path/to/your_outputs \
  --seed 42
```

Then, without retraining:

```bash
# Re-score the same model on its internal 20% test partition (seed must match the train run)
python -m text_pipeline.src.main --mode internal-eval \
  --input-data /path/to/your_train.csv --output-dir /path/to/your_outputs --seed 42

# Score the same model against a new external file
python -m text_pipeline.src.main --mode external-eval \
  --input-data /path/to/another_compliant_file.csv --output-dir /path/to/your_outputs
```

Run `python -m text_pipeline.src.main --help` for the full flag list.

## 17. Real Patient Data Is Not Included

No real patient data, real predictions, or real trained model weights are
part of this repository. `--input-data` / `--external-data` must point to
data you are authorized to use; this module performs no data collection or
transmission of its own.

## 18. Synthetic Example Data

The files under `examples/synthetic_tabular_data/` are entirely
artificially generated (see that directory's own README and
`generate_synthetic_data.py`). They exist only to exercise the code's
interface and are not derived from, and do not represent, any real patient
population or clinical distribution.
