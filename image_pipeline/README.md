# M0_IMG: ResNet-50 Image Classification Pipeline

## 1. Overview

This module provides an end-to-end pipeline for training, evaluating, and applying a three-class image classifier.

The classification task assigns each ophthalmic image to one of three severity categories:

* Mild
* Moderate
* Severe

The pipeline includes:

1. Image discovery and label assignment
2. Class-specific undersampling
3. Stratified train-validation-test splitting
4. Image preprocessing and augmentation
5. ResNet-50 model construction
6. Model training and checkpoint selection
7. Test-set evaluation
8. Single-image prediction

The module is operated through a command-line interface and supports both CPU and CUDA devices.

## 2. Repository Structure

```text
image_pipeline/
├── README.md
├── __init__.py
├── requirements.txt
├── requirements-dev.txt
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    ├── models.py
    ├── train.py
    ├── evaluate.py
    └── main.py
```

| File            | Description                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------- |
| `config.py`     | Defines image preprocessing, training parameters, class mappings, random seed, and output settings |
| `data_utils.py` | Discovers images, assigns labels, performs undersampling and splitting, and creates data loaders   |
| `models.py`     | Defines the ResNet-50 classifier and custom classification head                                    |
| `train.py`      | Implements Focal Loss, optimization, scheduling, early stopping, and checkpoint storage            |
| `evaluate.py`   | Computes evaluation metrics and generates plots and reports                                        |
| `main.py`       | Provides the command-line interface for training, evaluation, and prediction                       |

The output directory is created at runtime and is excluded from version control.

## 3. Dataset Structure

The pipeline expects a user-provided image directory containing three class subdirectories:

```text
dataset/
├── 0/
├── 1/
└── 2/
```

The class mapping is:

| Directory | Class    |
| --------- | -------- |
| `0`       | Mild     |
| `1`       | Moderate |
| `2`       | Severe   |

Labels are assigned from the immediate class-directory name. Image filenames are not used to determine labels.

Supported image formats are matched case-insensitively:

* `.jpg`
* `.jpeg`
* `.png`
* `.bmp`
* `.tiff`

Each of the three class directories must exist before training or evaluation begins.

## 4. Data Preparation and Splitting

The data-preparation sequence is:

1. Discover supported image files in the `0`, `1`, and `2` directories.
2. Assign the corresponding class label to each image.
3. Randomly undersample class `1` to a maximum of 1,200 images.
4. Retain all available images from classes `0` and `2`.
5. Create an image-level stratified train-validation-test split.
6. Build separate data loaders for the three partitions.

The default split is approximately:

| Partition  | Proportion |
| ---------- | ---------: |
| Training   |        65% |
| Validation |        15% |
| Test       |        20% |

The default random seed is 42. The split is stratified so that the class distribution is maintained across the three partitions.

The splitting unit is an individual image.

## 5. Image Preprocessing and Augmentation

All images are converted to RGB before transformation.

### Training transformations

The training transformation sequence is:

1. `Resize`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomVerticalFlip(p=0.3)`
4. `RandomRotation(degrees=15)`
5. `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)`
6. `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))`
7. `RandomResizedCrop(size=224, scale=(0.85, 1.0))`
8. `ToTensor`
9. ImageNet normalization

### Validation and test transformations

Validation and test images do not receive random augmentation.

```text
Resize(224 x 224) -> ToTensor -> ImageNet normalization
```

The normalization values are:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Random augmentation is applied only to the training partition.

## 6. Model Architecture

The classifier uses a ResNet-50 backbone initialized with:

```python
ResNet50_Weights.IMAGENET1K_V1
```

The initial `conv1` convolution is frozen. The following components remain trainable:

* `bn1`
* `layer1`
* `layer2`
* `layer3`
* `layer4`
* The complete custom classification head

The backbone output is processed by:

```python
AdaptiveAvgPool2d((7, 7))
```

The classification head is:

```text
Flatten
-> Dropout(0.4)
-> Linear(2048 * 7 * 7, 512)
-> BatchNorm1d(512)
-> ReLU
-> Dropout(0.3)
-> Linear(512, 128)
-> BatchNorm1d(128)
-> ReLU
-> Dropout(0.2)
-> Linear(128, 3)
```

The final layer produces logits for the Mild, Moderate, and Severe classes.

## 7. Training Configuration

The pipeline uses the following fixed training configuration:

| Component                   | Setting             |
| --------------------------- | ------------------- |
| Loss function               | Focal Loss          |
| Focal Loss class weights    | `[1.0, 1.2, 10.0]`  |
| Focal Loss gamma            | `2.5`               |
| Label smoothing             | `0.05`              |
| Optimizer                   | Adam                |
| Learning rate               | `5e-5`              |
| Weight decay                | `1e-4`              |
| Batch size                  | `16`                |
| Maximum epochs              | `200`               |
| Scheduler                   | `ReduceLROnPlateau` |
| Scheduler mode              | `min`               |
| Scheduler factor            | `0.5`               |
| Scheduler patience          | `7`                 |
| Early-stopping monitor      | Validation loss     |
| Early-stopping patience     | `15`                |
| Checkpoint-selection metric | Validation accuracy |

The checkpoint with the highest validation accuracy is saved as `best_model.pth`.

The learning-rate scheduler and early-stopping mechanism monitor validation loss. Training stops when the validation loss does not improve for 15 consecutive epochs or when the maximum of 200 epochs is reached.

Mixed-precision training is enabled automatically when CUDA is used. Standard full-precision training is used on CPU.

The pipeline uses fixed hyperparameters and does not perform automated hyperparameter search.

## 8. Installation

Run all commands from the repository root.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install runtime dependencies:

```bash
python -m pip install -r image_pipeline/requirements.txt
```

Install runtime and development dependencies:

```bash
python -m pip install -r image_pipeline/requirements-dev.txt
```

The development requirements include the runtime dependencies and `pytest`.

`torch` and `torchvision` must be installed as a compatible pair. CUDA users should select the appropriate PyTorch installation for their operating system and CUDA environment.

## 9. Command-Line Usage

Display all available options:

```bash
python -m image_pipeline.src.main --help
```

### Train a model

```bash
python -m image_pipeline.src.main --mode train --data-dir /path/to/dataset --output-dir /path/to/output --device auto --seed 42
```

Training requires a dataset containing the `0`, `1`, and `2` class directories.

### Evaluate a trained model

```bash
python -m image_pipeline.src.main --mode evaluate --data-dir /path/to/dataset --checkpoint /path/to/best_model.pth --output-dir /path/to/output --device auto --seed 42
```

Evaluation applies the same deterministic loading and stratified splitting procedure to the supplied dataset. Metrics are calculated on the resulting test partition, which represents approximately 20% of the supplied images.

When the same dataset and seed used for training are supplied, the command reconstructs the corresponding internal held-out test partition.

### Predict a single image

```bash
python -m image_pipeline.src.main --mode predict --image /path/to/image.jpg --checkpoint /path/to/best_model.pth --device auto --seed 42
```

Available device options are:

* `auto`
* `cpu`
* `cuda`

The `auto` option uses CUDA when it is available and otherwise uses the CPU. Selecting `cuda` when CUDA is unavailable produces an explicit error.

## 10. Output Files

Training and evaluation artifacts are stored under the selected output directory:

```text
output-dir/
├── models/
│   ├── best_model.pth
│   └── label_mapping.pkl
└── results/
    ├── training_history.png
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── class_performance.png
```

| File                        | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `best_model.pth`            | Model checkpoint with the highest validation accuracy |
| `label_mapping.pkl`         | Mapping between numerical labels and class names      |
| `training_history.png`      | Training and validation loss and accuracy curves      |
| `classification_report.txt` | Precision, recall, F1-score, and support by class     |
| `confusion_matrix.png`      | Test-set confusion matrix                             |
| `class_performance.png`     | Class-level performance comparison                    |

These files are generated at runtime and are not included in the repository.

## 11. Reproducibility and Testing

The default random seed is 42. The `--seed` argument controls:

* Python random operations
* NumPy random operations
* PyTorch CPU random operations
* PyTorch CUDA random operations
* Class-1 undersampling
* Train-validation-test splitting
* DataLoader generators
* DataLoader worker initialization

The pipeline also sets:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings improve reproducibility on the same software and hardware environment. Exact numerical equivalence across different operating systems, hardware, CUDA versions, or dependency versions is not guaranteed.

Run the image-pipeline tests without collecting the text-pipeline tests:

```bash
python -m pytest -q tests/test_image_pipeline_*.py
```

The tests cover:

* Module imports
* Configuration values
* Command-line validation
* Image discovery and class assignment
* Image-level stratified splitting
* Image transformations
* Random-seed propagation
* DataLoader reproducibility
* ResNet-50 model structure
* Pretrained-weight selection
* Layer-freezing behavior
* Focal Loss behavior
* Path portability
* Privacy and sensitive-file safeguards

The tests create temporary, non-medical solid-color images with Pillow. They do not use real patient images, download pretrained weights, execute a complete training run, or retain generated images after testing.

## 12. Data and Model Availability

This repository does not include research images, patient data, trained model checkpoints, or patient-level predictions.

Users must provide their own authorized and appropriately labeled image data in the directory structure described above.

Sensitive medical images and identifiable filenames must not be committed to a public repository. The temporary images created by the test suite are artificial and are used only to verify the software interface.
