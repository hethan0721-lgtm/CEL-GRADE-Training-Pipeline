# CEL-GRADE Training Pipeline

## Overview

This repository provides the training and evaluation pipelines for two models developed in the CEL-GRADE project:

* **M0_CLIN**: An XGBoost binary classifier based on structured clinical variables for predicting surgical recommendation.
* **M0_IMG**: A ResNet-50 image classifier for classifying ophthalmic images into mild, moderate, and severe categories.

Both modules include source code, command-line interfaces, dependency files, automated tests, and usage documentation. Input and output locations are managed through command-line arguments or repository-relative paths.

This repository does not include real patient data, research images, patient-level predictions, or trained model weights.

## Repository Structure

```text
CEL-GRADE-Training-Pipeline/
├── README.md
├── .gitignore
├── image_pipeline/
│   ├── README.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── src/
├── text_pipeline/
│   ├── README.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── src/
├── examples/
│   └── synthetic_tabular_data/
└── tests/
```

| Directory                          | Description                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `text_pipeline/`                   | M0_CLIN XGBoost pipeline for training and evaluating structured clinical data                               |
| `image_pipeline/`                  | M0_IMG ResNet-50 pipeline for image classification                                                          |
| `examples/synthetic_tabular_data/` | Artificially generated tabular data for demonstrating the text-pipeline input format and software interface |
| `tests/`                           | Automated tests for the text and image pipelines                                                            |

Detailed information about input formats, model architectures, training parameters, command-line usage, and output files is available in the corresponding documentation:

* [Text Pipeline Documentation](text_pipeline/README.md)
* [Image Pipeline Documentation](image_pipeline/README.md)
* [Synthetic Tabular Data Documentation](examples/synthetic_tabular_data/README.md)

## Pipeline Overview

### M0_CLIN: Structured Clinical Data Model

M0_CLIN uses five structured clinical features to train an XGBoost binary classifier.

The main workflow includes:

1. Loading and validating tabular data.
2. Cleaning numerical, categorical, and target variables.
3. Creating a stratified training and internal test split.
4. Fitting the imputation, standardization, and categorical encoding components only on the training partition.
5. Applying SMOTE only to the training partition.
6. Training an XGBoost classifier with fixed hyperparameters.
7. Performing five-fold stratified cross-validation.
8. Evaluating the model on the internal test partition and an external dataset.
9. Saving the trained model, preprocessing components, aggregate metrics, and evaluation plots.

The module includes fully artificial tabular datasets for verifying installation, input structure, and end-to-end execution. These synthetic data do not represent a real patient population and must not be used to estimate clinical performance.

For full details, see [`text_pipeline/README.md`](text_pipeline/README.md).

### M0_IMG: Ophthalmic Image Classification Model

M0_IMG uses a ResNet-50 model initialized with ImageNet-1K V1 pretrained weights to perform three-class image classification.

The main workflow includes:

1. Loading images from three class directories and assigning labels.
2. Applying random undersampling with a fixed upper limit to the specified class.
3. Creating image-level stratified training, validation, and test partitions.
4. Applying random image augmentation only to the training partition.
5. Building the ResNet-50 model with a custom classification head.
6. Freezing the initial convolutional layer and fine-tuning the remaining network layers.
7. Training the model using Focal Loss, the Adam optimizer, learning-rate scheduling, and early stopping.
8. Saving the checkpoint with the highest validation accuracy.
9. Evaluating the model on the test partition and supporting single-image prediction.

The module does not include research images or trained model weights. Users must provide authorized image data that follow the required directory structure.

For full details, see [`image_pipeline/README.md`](image_pipeline/README.md).

## Installation

Separate virtual environments are recommended for the two modules to avoid dependency conflicts.

### Text Pipeline Dependencies

Install the runtime dependencies:

```bash
python -m pip install -r text_pipeline/requirements.txt
```

Install the runtime and testing dependencies:

```bash
python -m pip install -r text_pipeline/requirements-dev.txt
```

### Image Pipeline Dependencies

Install the runtime dependencies:

```bash
python -m pip install -r image_pipeline/requirements.txt
```

Install the runtime and testing dependencies:

```bash
python -m pip install -r image_pipeline/requirements-dev.txt
```

When running the image pipeline with CUDA, install mutually compatible versions of `torch` and `torchvision` for the local operating system and CUDA environment.

## Quick Start

Run all commands from the repository root.

### Text Pipeline

Display the available command-line options:

```bash
python -m text_pipeline.src.main --help
```

Run the text pipeline with the bundled synthetic tabular data:

```bash
python -m text_pipeline.src.main --mode train
```

### Image Pipeline

Display the available command-line options:

```bash
python -m image_pipeline.src.main --help
```

Train the image model with a user-provided dataset:

```bash
python -m image_pipeline.src.main \
  --mode train \
  --data-dir /path/to/dataset \
  --output-dir /path/to/output \
  --device auto \
  --seed 42
```

The image dataset must contain three class directories named `0`, `1`, and `2`. See the image-pipeline documentation for the complete dataset structure and additional execution modes.

## Testing

### Text Pipeline Tests

```bash
python -m pytest -q tests --ignore-glob="tests/test_image_pipeline_*.py"
```

### Image Pipeline Tests

```bash
python -m pytest -q tests --ignore-glob="tests/test_text_pipeline_*.py"
```

The automated tests cover:

* Module imports
* Configuration and model parameters
* Input validation
* Data splitting
* Data leakage safeguards
* Random-seed propagation and reproducibility
* Model architecture and training behavior
* Command-line validation
* Path portability
* Privacy and sensitive-file safeguards
* End-to-end execution with artificial data

The image-pipeline tests generate temporary, non-medical solid-color images during execution. They do not use or retain real patient images.

## Reproducibility

Both pipelines use a fixed default random seed and allow the random seed to be specified through the command-line interface.

For the text pipeline, the random seed controls:

* The stratified training and test split
* SMOTE
* XGBoost
* Five-fold cross-validation

For the image pipeline, the random seed controls:

* Python, NumPy, and PyTorch random operations
* Class undersampling
* Training, validation, and test splitting
* DataLoader generators
* DataLoader worker initialization

Fixed random seeds improve reproducibility within the same software and hardware environment. Exact numerical equivalence across different operating systems, hardware platforms, CUDA versions, or dependency versions is not guaranteed.

## Data and Model Availability

This repository does not include:

* Real patient data
* Ophthalmic research images
* Identifiable patient information
* Patient-level predictions
* Trained model weights

The tabular files in `examples/synthetic_tabular_data/` are entirely generated by software. They are provided only to demonstrate the expected input structure, verify the software interface, and support automated testing. They are not derived from real patients and must not be used to train a clinical model, estimate clinical performance, or draw medical conclusions.

Users must provide their own authorized data in the format required by the corresponding module. Sensitive, restricted, or identifiable clinical data must not be committed to a public code repository.
