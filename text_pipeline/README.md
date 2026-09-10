# M0_CLIN: XGBoost Clinical Tabular Pipeline

## 1. Overview

This module provides an end-to-end pipeline for training and evaluating a binary XGBoost classifier that predicts the need for surgery from structured clinical variables.

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
| `data_loader.py`         | Loads, validates, filters, and cleans tabular input data                                          |
| `feature_engineering.py` | Applies imputation, scaling, and categorical encoding                                             |
| `models.py`              | Defines the XGBoost classifier interface                                                          |
| `train.py`               | Performs SMOTE, class-weight calculation, model fitting, and cross-validation                     |
| `evaluate.py`            | Computes evaluation metrics and generates plots and reports                                       |
| `main.py`                | Provides the command-line interface for training and evaluation                                   |

The `outputs/` directory is created at runtime and is excluded from version control.

## 3. Input Data

The pipeline supports the following file formats:

* `.csv`
* `.xlsx`
* `.xls`

CSV files are read with UTF-8-compatible encoding. Excel files are read through the pandas Excel interface.

Each input table must contain five model features and one binary target variable.

| Clinical variable                       | Data type     | Processing                                   |
| --------------------------------------- | ------------- | -------------------------------------------- |
| Dislocation severity                    | Categorical   | String cleaning and label encoding           |
| Corrected visual acuity                 | Numerical     | Median imputation and standardization        |
| Corrected spherical power in diopters   | Numerical     | Median imputation and standardization        |
| Corrected cylindrical power in diopters | Numerical     | Median imputation and standardization        |
| IOLMaster cylinder power in diopters    | Numerical     | Median imputation and standardization        |
| Need for surgery                        | Binary target | Non-surgery is encoded as 0 and surgery as 1 |

The exact machine-readable column names and raw label values are defined in `Config.FEATURE_COLUMNS`, `Config.TARGET_COLUMN`, and `Config.LABEL_MAPPING`. Input files must follow that schema. The bundled synthetic CSV files provide valid templates.

Only configured model features and the target variable enter the training matrix. Other columns are excluded from model fitting.

Rows with missing or unsupported target values are removed before training. Rows in which all numerical model features are missing are also removed.

## 4. Preprocessing Pipeline

The preprocessing sequence is:

1. Retain the configured feature and target columns.
2. Clean numerical values and convert them to floating-point values.
3. Convert unparseable numerical values to `NaN`.
4. Clean categorical values and convert them to strings.
5. Convert categorical missing values into an explicit `nan` category.
6. Remove rows with missing or unsupported target values.
7. Encode the binary target.
8. Create a stratified 80/20 train-test split.
9. Fit the feature-processing objects on the 80% training partition.
10. Apply the fitted processing objects to the internal test partition and external dataset.
11. Apply SMOTE only to the processed training partition.
12. Fit the XGBoost classifier.

Numerical features are median-imputed and standardized with `StandardScaler`. The categorical feature is encoded with `LabelEncoder`.

The fitted feature-processing components are saved with the trained model and reused during evaluation.

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

The model is trained for 200 estimators without early stopping. Automated hyperparameter search is not part of the training workflow.

SMOTE is applied only to the training partition with:

* `sampling_strategy='auto'`
* `k_neighbors=5`
* Default random seed of 42

Balanced class weights are calculated from the resampled training labels and converted to `scale_pos_weight` for XGBoost.

Five-fold stratified cross-validation is performed for internal diagnostic evaluation. Each fold independently fits its preprocessing components and applies SMOTE only to the training rows of that fold. Cross-validation does not select model hyperparameters or replace the final fitted model.

## 6. Data Leakage Prevention

The pipeline separates model fitting from evaluation data processing.

* The 80/20 split is created before any imputer, scaler, encoder, or SMOTE object is fitted.
* Numerical imputation and standardization are fitted only on the 80% training partition.
* Categorical encoding is fitted only on the 80% training partition.
* SMOTE is fitted and applied only to the training partition.
* The internal 20% test partition is processed with the fitted training preprocessor.
* The external dataset is processed with the same fitted preprocessor.
* Internal and external evaluation modes do not refit the preprocessor or classifier.
* Each cross-validation fold creates and fits independent preprocessing and resampling components.

Automated leakage checks are provided in `tests/test_text_pipeline_leakage_guard.py`.

## 7. Installation

Run all commands from the repository root.

Install runtime dependencies:

```bash
pip install -r text_pipeline/requirements.txt
```

Install runtime and development dependencies:

```bash
pip install -r text_pipeline/requirements-dev.txt
```

The development requirements include the runtime dependencies and `pytest`.

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
python -m text_pipeline.src.main --mode train --input-data /path/to/training_data.csv --external-data /path/to/external_data.csv --output-dir /path/to/outputs --seed 42
```

The same command structure can be used with `.xlsx` or `.xls` input files.

### Reproduce the internal evaluation

```bash
python -m text_pipeline.src.main --mode internal-eval --input-data /path/to/training_data.csv --output-dir /path/to/outputs --seed 42
```

The input file and seed must match the training run so that the same internal 20% test partition is reconstructed.

### Evaluate an external dataset

```bash
python -m text_pipeline.src.main --mode external-eval --input-data /path/to/external_data.csv --output-dir /path/to/outputs
```

External evaluation loads the saved model and fitted feature processor. It does not retrain or refit any component.

## 9. Outputs

The default output directory is `text_pipeline/outputs/`.

### Model artifacts

The following files are saved under `<output-dir>/models/`:

| File                     | Description                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `feature_engineer.pkl`   | Fitted preprocessing components and feature metadata                                                             |
| `best_xgboost_model.pkl` | Trained XGBoost model, feature names, and feature importance values                                              |
| `training_history.pkl`   | Training duration, sample counts, resampling information, class-weight information, and cross-validation results |

### Evaluation outputs

Internal evaluation results are saved under:

```text
<output-dir>/eval_internal/
```

External evaluation results are saved under:

```text
<output-dir>/eval_external/
```

Generated evaluation files include:

* Classification report
* Confusion matrix
* Receiver operating characteristic curve
* Precision-recall curve
* Feature importance plot
* Metrics comparison plot

Evaluation outputs contain aggregate metrics and plots. They do not contain raw input rows or row-level prediction files.

## 10. Reproducibility and Testing

The default random seed is 42. The command-line `--seed` argument controls the random state used by:

* The stratified train-test split
* SMOTE
* XGBoost
* Cross-validation

To run the text-pipeline test suite without collecting the image-pipeline tests:

```bash
python -m pytest -q tests/test_text_pipeline_*.py
```

The tests cover:

* Module imports
* Input schema validation
* Synthetic data loading
* Feature-set consistency
* Data leakage prevention
* Categorical missing-value handling
* Fixed training parameters
* Path portability
* Privacy safeguards
* End-to-end execution with synthetic data

## 11. Data and Model Availability

This repository does not include real patient data, real patient-level predictions, or trained model weights.

The files in [`examples/synthetic_tabular_data/`](../examples/synthetic_tabular_data/) are artificially generated and are provided only to demonstrate the expected input structure and test the software interface. They are not derived from a real patient population and must not be used to estimate clinical performance.

Users must provide their own authorized data in the required schema when running the pipeline on research datasets. Sensitive or identifiable clinical data must not be committed to a public repository.
