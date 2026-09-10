# M0_IMG ResNet-50 Image Classification Pipeline

This module is the canonical M0_IMG image training and evaluation
pipeline. It performs a three-class severity classification task (Mild,
Moderate, Severe) on user-supplied ophthalmic images.

This repository does **not** contain any research images, patient data,
trained model checkpoints, or paper result files. The code itself is
agnostic to where images came from: labels are derived purely from the
`0` / `1` / `2` subfolder an image is placed in. This module does not
implement or discuss patient-level splitting, external validation
cohorts, or any comparison against other model variants -- it is a
single, self-contained image-classification training/evaluation/predict
pipeline, described here strictly as it behaves.

## 1. Overview

- **Task**: 3-class severity classification (Mild / Moderate / Severe)
  from a single ophthalmic image.
- **Input**: a user-provided image directory with `0/`, `1/`, `2/`
  class subfolders (see [Dataset structure](#3-dataset-structure)).
- **Model**: ResNet-50 backbone (ImageNet-pretrained) with a custom
  classification head.
- **Interface**: a single command-line entry point supporting `train`,
  `evaluate`, and `predict` modes.

## 2. Pipeline

The real, code-verified execution sequence is:

1. Read the user-provided `0/`, `1/`, `2/` class directories.
2. Assign labels purely from the directory name (`0`, `1`, `2`).
3. Apply random undersampling to class `1`, capped at a maximum of
   1200 images (classes `0` and `2` are not capped).
4. Perform an image-level, stratified train/validation/test split with
   `random_state=42`.
5. Resulting split proportions are approximately:
   - Training: 65%
   - Validation: 15%
   - Test: 20%
6. Only the training split receives random image augmentation.
7. The validation and test splits only go through `Resize`, `ToTensor`,
   and ImageNet normalization.
8. A ResNet-50 backbone is created using `ResNet50_Weights.IMAGENET1K_V1`
   pretrained weights.
9. The initial `conv1` convolution is frozen; the rest of the ResNet
   backbone (`bn1`, `layer1`-`layer4`) and the custom classification
   head are fine-tuned.
10. Training uses Focal Loss, the Adam optimizer, a
    `ReduceLROnPlateau` learning-rate scheduler, and early stopping.
11. The checkpoint with the highest validation accuracy is saved as
    `best_model.pth`.
12. The best checkpoint is reloaded and evaluated on the held-out test
    split.

**Note:** `layer1` and `layer2` of the ResNet-50 backbone are **not**
frozen -- only the stem `conv1` convolution is. See
[Model architecture](#6-model-architecture) below.

## 3. Dataset structure

This pipeline does not ship with any dataset. Users must supply their
own image directory with the following structure:

```
dataset/
├── 0/
├── 1/
└── 2/
```

## 4. Class mapping

| Folder | Class    |
|--------|----------|
| `0`    | Mild     |
| `1`    | Moderate |
| `2`    | Severe   |

Supported image formats (matched case-insensitively):

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tiff`

Notes:

- Labels come entirely from the subfolder name -- there is no Excel,
  CSV, or other label file involved.
- Filenames themselves are never used to derive a label.
- This repository does not provide any example medical images.
- The automated test suite only creates solid-color, non-medical
  placeholder images inside pytest's temporary directories at test run
  time; nothing is retained in the repository afterward.

## 5. Data preprocessing and augmentation

Training split transform order:

1. `Resize`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomVerticalFlip(p=0.3)`
4. `RandomRotation(degrees=15)`
5. `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)`
6. `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))`
7. `RandomResizedCrop(size=224, scale=(0.85, 1.0))`
8. `ToTensor`
9. ImageNet normalization

Validation and test split transforms:

```
Resize(224 x 224) -> ToTensor -> ImageNet normalization
```

Normalization values:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 6. Model architecture

- **Backbone**: ResNet-50
- **Pretrained weights**: `torchvision.models.ResNet50_Weights.IMAGENET1K_V1`
- **Frozen**: the initial `conv1` convolution only
- **Fine-tuned**: `bn1`, `layer1`, `layer2`, `layer3`, `layer4`, and the
  entire custom classification head
- **Pooling**: `AdaptiveAvgPool2d((7, 7))`

Classification head:

```
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

## 7. Training configuration

- **Loss**: Focal Loss
  - `alpha = [1.0, 1.2, 10.0]` (Mild, Moderate, Severe)
  - `gamma = 2.5`
  - `label_smoothing = 0.05`
- **Optimizer**: Adam
  - `learning_rate = 5e-5`
  - `weight_decay = 1e-4`
- **Batch size**: 16
- **Maximum epochs**: 200
- **Scheduler**: `ReduceLROnPlateau`
  - `mode = 'min'`
  - `factor = 0.5`
  - `patience = 7`
- **Early stopping**: monitors validation loss, `patience = 15`
- **Best checkpoint selection**: highest validation accuracy
- **Mixed precision training**: automatically enabled when training on
  a CUDA device (`torch.cuda.amp.autocast` / `GradScaler`); not used on
  CPU
- Automatic hyperparameter search is not part of this pipeline; all
  hyperparameters above are fixed values in the source code.

## 8. Installation

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install runtime dependencies:

```bash
python -m pip install -r image_pipeline/requirements.txt
```

Install test dependencies (adds `pytest` on top of the runtime deps):

```bash
python -m pip install -r image_pipeline/requirements-dev.txt
```

`torch` and `torchvision` must be installed as a mutually compatible
pair. GPU users should follow PyTorch's official installation
instructions for their local CUDA toolkit version instead of relying on
the generic PyPI wheels in `requirements.txt`.

**Environment records** (two distinct things -- do not conflate them):

- **Historical environment record** (found in the original, pre-cleanup
  source tree's `requirements.txt`, used only as evidence for the
  version *ranges* in `image_pipeline/requirements.txt`):
  - `torch 2.0.1`
  - `torchvision 0.15.2`
- **Public pipeline verification environment** (used to verify that
  this cleaned-up, public version of the pipeline imports, runs its
  CLI, and passes its test suite -- this is **not** a claim about the
  original training environment):
  - Python 3.13.5
  - `torch 2.8.0+cu126`
  - `torchvision 0.23.0+cu126`
  - `pytest 8.4.2`

## 9. Command-line usage

Replace all paths below with your own; they are placeholders only.

**Train**:

```bash
python -m image_pipeline.src.main --mode train --data-dir /path/to/dataset --output-dir /path/to/output --device auto --seed 42
```

**Evaluate**:

```bash
python -m image_pipeline.src.main --mode evaluate --data-dir /path/to/dataset --checkpoint /path/to/best_model.pth --output-dir /path/to/output --device auto --seed 42
```

**Predict** (single image):

```bash
python -m image_pipeline.src.main --mode predict --image /path/to/image.jpg --checkpoint /path/to/best_model.pth --device auto --seed 42
```

**Important -- what `--mode evaluate` actually does**: it does *not*
evaluate every image found under `--data-dir`. It re-runs the exact
same loading and image-level stratified split described in
[Pipeline](#2-pipeline) (same `test_size=0.20`, `val_size=0.15`,
`random_state=42` by default) against whatever `--data-dir` you supply,
and then scores the model only on the resulting **test** partition
(~20%) of that directory. If you point `--data-dir` at the same
directory used for training, this reproduces the internal held-out test
split. If you point it at a different directory, the code still splits
it 65/15/20 and evaluates only the ~20% "test" slice of *that*
directory -- it is not a pre-defined external validation set and should
not be described as one.

## 10. Output files

```
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

These are runtime-generated artifacts. None of them are included in
this repository.

## 11. Reproducibility

`--seed` (default `42`) is propagated to `Config.RANDOM_STATE` and used
to seed:

- Python's `random` module
- NumPy
- PyTorch (CPU and CUDA RNG)
- the class-1 undersampling step
- the train/validation/test split
- the DataLoader generators and worker processes

In addition, `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False` are set.

This maximizes run-to-run consistency on the same machine, but bit-for-
bit identical results across different hardware, CUDA versions, or
library versions are not guaranteed.

## 12. Tests

Run the image pipeline test suite with:

```bash
python -m pytest -q tests/test_image_pipeline_*.py
```

Last verified result (see the M0_IMG pipeline cleanup report for
details):

```
69 passed, 0 failed, 0 skipped, 0 warnings
```

The test suite does not use real patient images, does not download
ImageNet weights, and does not run a full training loop -- it uses
programmatically generated, non-medical synthetic images, `pretrained=False`
model construction (or a monkeypatched `torchvision.models.resnet50`),
and source-level contract checks instead.

## 13. Data and model availability

- Research images are not publicly included in this repository.
- Patient-level data are not publicly included in this repository.
- Trained model checkpoints are not included in this repository.
- Users must supply their own appropriately labeled images in the
  structure described in [Dataset structure](#3-dataset-structure).
- Do not commit sensitive medical images or identifiable filenames to a
  public repository.
- The repository's own tests generate only temporary, non-medical,
  programmatically created images that are discarded at the end of each
  test run.

## 14. Notes on method fidelity

- This public version preserves the model architecture, training
  hyperparameters, image augmentation, data split logic, and checkpoint
  selection rule that were actually in effect in M0_IMG.
- The engineering cleanup performed to prepare this module for public
  release was limited to: converting to relative package imports,
  parameterizing paths through the CLI instead of hardcoded defaults,
  adding explicit randomness control, consolidating configuration
  values that previously lived only as hardcoded literals, and updating
  a deprecated torchvision weights API call to its current equivalent.
- The pretrained-weights API call now explicitly uses
  `ResNet50_Weights.IMAGENET1K_V1` (rather than the newer `DEFAULT`,
  which would resolve to `IMAGENET1K_V2`), preserving the same
  ImageNet initialization semantics that the historical `pretrained=True`
  call used.
