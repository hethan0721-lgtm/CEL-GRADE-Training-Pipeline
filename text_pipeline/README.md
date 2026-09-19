# M0_CLIN: XGBoost Clinical Tabular Pipeline

## 1. Overview

This module provides an end-to-end pipeline for training and evaluating a binary XGBoost classifier that predicts whether surgery is required from structured clinical variables.

The pipeline includes:

1. Tabular data loading and validation
2. Data cleaning
3. Stratified train-test splitting
4. Feature preprocessing
5. Training-set resampling
6. XGBoost model training
7. Five-fold cross-validation
8. Internal and external evaluation
9. Model and preprocessing artifact storage

The module is operated through a command-line interface and supports both Windows and Linux.

## 2. Repository Structure

```text
text_pipeline/
├── README.md
├── __init__.py
├── requirements.txt
├── requirements-dev.txt
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── models.py
    ├── train.py
    ├── evaluate.py
    └── main.py
```

| File                     | Description                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| `config.py`              | Defines the input schema, preprocessing settings, model parameters, random seed, and output paths |
| `data_loader.py`         | Loads, selects, and cleans tabular input data                                                     |
| `feature_engineering.py` | Applies numerical imputation, scaling, and categorical encoding                                   |
| `models.py`              | Defines the XGBoost classifier interface                                                          |
| `train.py`               | Performs SMOTE, class-weight calculation, model fitting, and cross-validation                     |
| `evaluate.py`            | Computes evaluation metrics and generates plots and reports                                       |
| `main.py`                | Provides the command-line interface for training and evaluation                                   |

The `outputs/` directory is created at runtime and is excluded from version control.

## 3. Input Data

The documented runtime dependencies support the following formats:

* `.csv`
* `.xlsx`

The data-loading code also recognizes legacy `.xls` files. Reading `.xls` files uses the `xlrd` package included in the default runtime requirements.

CSV files are read using UTF-8-compatible encoding. Excel files are read through the pandas Excel interface.

Each input table must contain seven model features and one binary target variable.

| Model order | Clinical variable                       | Data type     | Processing                                               |
| ----------- | --------------------------------------- | ------------- | -------------------------------------------------------- |
| 1           | Corrected visual acuity                 | Numerical     | Median imputation and standardization                    |
| 2           | Corrected spherical power in diopters   | Numerical     | Median imputation and standardization                    |
| 3           | Corrected cylindrical power in diopters | Numerical     | Median imputation and standardization                    |
| 4           | IOLMaster cylinder power in diopters    | Numerical     | Median imputation and standardization                    |
| 5           | Age                                     | Numerical     | Median imputation and standardization                    |
| 6           | Cooperation with examination            | Categorical   | String cleaning and label encoding                       |
| 7           | Dislocation severity                    | Categorical   | String cleaning and label encoding                       |
| Target      | Need for surgery                        | Binary target | Non-surgery is encoded as 0, and surgery is encoded as 1 |

Class 1 is the positive class for all binary evaluation metrics. The prediction decision threshold is 0.5 on the model's estimated probability of class 1.

The exact machine-readable column names, accepted categorical values, target labels, and label mappings are defined in:

* `Config.FEATURE_COLUMNS`
* `Config.NUMERICAL_FEATURES`
* `Config.CATEGORICAL_FEATURES`
* `Config.TARGET_COLUMN`
* `Config.LABEL_MAPPING`

The bundled synthetic CSV files provide valid input templates.

Input files do not need to place their columns in the model's fixed feature order. Columns are selected by their machine-readable names and then reordered internally according to `Config.FEATURE_COLUMNS` before they enter the model.

All seven configured feature columns and the target column must be present. Additional columns, including anonymous sample identifiers, are excluded from the final model feature matrix.

Target values must belong to the supported label set defined in `Config.LABEL_MAPPING`. Missing or unsupported target values are not valid training inputs and should be corrected or removed before the pipeline is run.

Rows in which every numerical model feature is missing or invalid should also be reviewed before training. Numerical imputation supports individual missing feature values, but callers should not rely on every execution path to silently remove rows containing no valid numerical information.

## 4. Preprocessing Pipeline

The preprocessing sequence is:

1. Load the training and external evaluation tables.
2. Select the configured feature and target columns.
3. Reorder the model features according to `Config.FEATURE_COLUMNS`.
4. Clean numerical values and convert them to floating-point values.
5. Convert unparseable numerical values to missing values.
6. Trim categorical strings and standardize their in-memory representation.
7. Convert categorical missing values into an explicit `nan` string category.
8. Encode the binary target using the configured label mapping.
9. Create a stratified 80/20 training and internal test split.
10. Fit all feature-processing objects on the 80% training partition only.
11. Transform the internal test partition and external dataset using the fitted training preprocessor.
12. Apply SMOTE only to the processed training partition.
13. Fit the XGBoost classifier.
14. Evaluate the saved model on the internal test partition and external dataset.

Numerical features are median-imputed and standardized with `StandardScaler`.

Categorical features are cleaned as strings and encoded with `LabelEncoder`.

The current pipeline intentionally preserves its historical categorical missing-value behavior. A missing categorical value is converted into the literal string `nan` before categorical encoding. It is therefore treated as an explicit category rather than being replaced with the most frequent category.

Changing this behavior would alter the fitted preprocessing objects and could change model predictions. It should therefore be treated as a separate model-behavior change rather than a documentation-only modification.

The fitted feature-processing components are saved with the trained model and reused during internal and external evaluation.

## 5. Training Configuration

The classifier uses a fixed XGBoost configuration.

| Parameter              |             Value |
| ---------------------- | ----------------: |
| Objective              | `binary:logistic` |
| Evaluation metrics     |  `logloss`, `auc` |
| Maximum tree depth     |                 6 |
| Learning rate          |              0.05 |
| Number of estimators   |               200 |
| Minimum child weight   |                 3 |
| Gamma                  |               0.1 |
| Subsample ratio        |               0.8 |
| Column subsample ratio |               0.8 |
| L1 regularization      |              0.05 |
| L2 regularization      |               1.0 |
| Default random seed    |                42 |

The model is trained for 200 estimators without early stopping.

Automated hyperparameter search is not part of the production training workflow.

SMOTE is applied only to the training partition with:

* `sampling_strategy='auto'`
* `k_neighbors=5`
* Default random seed of 42

Balanced class weights are calculated from the resampled training labels and converted into the `scale_pos_weight` value used by XGBoost.

Five-fold stratified cross-validation is performed for internal diagnostic evaluation. Each fold independently fits its preprocessing components and applies SMOTE only to the training rows of that fold.

Cross-validation is not used to select hyperparameters and does not replace the final fitted model.

## 6. Data Leakage Prevention

The pipeline separates model fitting from evaluation data processing.

* The 80/20 split is created before any imputer, scaler, encoder, or SMOTE object is fitted.
* Numerical imputation and standardization are fitted only on the 80% training partition.
* Categorical encoding is fitted only on the 80% training partition.
* SMOTE is fitted and applied only to the training partition.
* The internal 20% test partition is transformed using the preprocessor fitted on the training partition.
* The external dataset is transformed using the same fitted training preprocessor.
* Neither the internal test partition nor the external dataset is passed to the classifier's `.fit()` call.
* Neither evaluation dataset is used as an `eval_set` during final model fitting.
* No early stopping is configured in the production training workflow.
* Internal and external evaluation modes do not refit the preprocessor or classifier.
* Each cross-validation fold creates and fits independent preprocessing and resampling components.

Automated leakage checks are provided in `tests/test_text_pipeline_leakage_guard.py` and the related text-pipeline validation tests.

## 7. Installation

Run all commands from the repository root.

Python 3.9 or later is required. The pipeline was developed and tested primarily with Python 3.11.

Install the runtime dependencies:

```bash
python -m pip install -r text_pipeline/requirements.txt
```

Install the runtime and development dependencies:

```bash
python -m pip install -r text_pipeline/requirements-dev.txt
```

The development requirements include the runtime dependencies and `pytest`.

The runtime requirements include `openpyxl` for `.xlsx` files and `xlrd` for legacy `.xls` files.

## 8. Command-Line Usage

Display the available command-line options:

```bash
python -m text_pipeline.src.main --help
```

### Train with the bundled synthetic data

The default input paths point to the synthetic training and external evaluation files in `examples/synthetic_tabular_data/`.

```bash
python -m text_pipeline.src.main --mode train
```

### Train with custom data

```bash
python -m text_pipeline.src.main \
  --mode train \
  --input-data /path/to/training_data.csv \
  --external-data /path/to/external_data.csv \
  --output-dir /path/to/outputs \
  --seed 42
```

The same command structure can be used with `.xlsx` input files. Legacy `.xls` input is supported through the included `xlrd` dependency.

### Reproduce the internal evaluation

```bash
python -m text_pipeline.src.main \
  --mode internal-eval \
  --input-data /path/to/training_data.csv \
  --output-dir /path/to/outputs \
  --seed 42
```

The input file and seed must match the training run so that the same internal 20% test partition can be reconstructed.

### Evaluate an external dataset

```bash
python -m text_pipeline.src.main \
  --mode external-eval \
  --input-data /path/to/external_data.csv \
  --output-dir /path/to/outputs
```

External evaluation loads the saved model and fitted feature processor. It does not retrain the model or refit any preprocessing component.

## 9. Outputs

The default output directory is:

```text
text_pipeline/outputs/
```

### Model Artifacts

The following files are saved under `<output-dir>/models/`:

| File                     | Description                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `feature_engineer.pkl`   | Fitted preprocessing components and feature metadata                                                             |
| `best_xgboost_model.pkl` | Trained XGBoost model, feature names, and feature-importance values                                              |
| `training_history.pkl`   | Training duration, sample counts, resampling information, class-weight information, and cross-validation results |

### Internal Evaluation Outputs

Internal evaluation results are saved under:

```text
<output-dir>/eval_internal/
```

### External Evaluation Outputs

External evaluation results are saved under:

```text
<output-dir>/eval_external/
```

Generated evaluation files include:

* Classification report
* Confusion matrix
* Receiver operating characteristic curve
* Precision-recall curve
* Feature-importance plot
* Metrics-comparison plot

Evaluation outputs contain aggregate metrics and plots. They do not contain raw input rows or row-level prediction files.

## 10. Reproducibility and Testing

The default random seed is 42.

The command-line `--seed` argument controls the random state used by:

* The stratified train-test split
* SMOTE
* XGBoost
* Cross-validation

Fixed random seeds improve reproducibility within the same software and hardware environment. Exact numerical equivalence across different operating systems, dependency versions, processors, or numerical libraries is not guaranteed.

Run the text-pipeline test suite without collecting the image-pipeline tests:

```bash
python -m pytest -q tests/test_text_pipeline_*.py
```

The tests cover:

* Module imports
* Seven-feature input schema validation
* Fixed feature order
* Synthetic data loading
* Feature-set consistency
* Anonymous identifier exclusion
* Target exclusion from the feature matrix
* Data leakage prevention
* Categorical missing-value handling
* Unseen categorical-value behavior
* Fixed model parameters
* Training-only SMOTE behavior
* Internal test isolation
* Path portability
* Privacy safeguards
* End-to-end execution with synthetic data

The test suite uses synthetic data only. It does not require or access real clinical data.

## 11. Data and Model Availability

The official, hash-verified M0_CLIN weights and fitted preprocessing
artifacts are included in this repository at `pretrained/` (see the
[repository-root `MODEL_WEIGHTS.md`](../MODEL_WEIGHTS.md) for checksums,
the required feature order, a loading example, and known limitations).

This repository does not include:

* Real patient data
* Identifiable clinical information
* Real patient-level predictions
* The private clinical dataset used to train the released weights
* Clinical performance results derived from private datasets

The files in [`examples/synthetic_tabular_data/`](../examples/synthetic_tabular_data/) are generated entirely by software. They are provided only to demonstrate the expected input structure, verify the software interface, and support automated testing.

The synthetic files are not derived from a real patient population. They must not be used to estimate clinical performance, validate a medical hypothesis, or draw scientific conclusions.

Users must provide their own authorized data in the required schema when running the pipeline on research datasets.

Sensitive, restricted, or identifiable clinical data must not be committed to a public repository.

## 12. Intended Use and Limitations

This software is provided for research and reproducibility purposes.

It is not a certified medical device and must not be used as the sole basis for diagnosis, treatment selection, surgical decision-making, or other direct clinical decisions.

Users are responsible for:

* Verifying the input schema
* Confirming data quality
* Confirming that all target labels are valid
* Reviewing missing or invalid feature values
* Protecting sensitive information
* Obtaining all required institutional and ethical approvals
* Validating model performance on an appropriate target population

Results obtained from user-provided data may vary because of differences in population characteristics, data collection procedures, software versions, hardware, and preprocessing decisions.
