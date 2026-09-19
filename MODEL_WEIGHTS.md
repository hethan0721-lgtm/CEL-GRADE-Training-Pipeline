# CEL-GRADE Model Weights

This document describes the two officially released, hash-verified model
weight sets for the CEL-GRADE project: **M0_CLIN** (text/tabular) and
**M0_IMG** (image). It is the canonical reference for file locations,
checksums, and loading instructions.

Both weight sets were produced by frozen, audited training runs and were
verified before release: a full parameter-level and prediction-level
consistency audit confirmed the published artifacts exactly reproduce the
authoritative training run's results (see the per-model sections below for
the specific evidence).

**Research use only.** Neither model is a validated, standalone clinical
decision-making tool. See "Intended use" in each section.

---

## Text model — M0_CLIN (seven-feature XGBoost)

- **Model**: XGBoost binary classifier (`xgboost.sklearn.XGBClassifier`)
- **Task**: Surgery-recommendation binary classification (surgery vs.
  non-surgery) from seven structured clinical features
- **Internal test set**: N = 313
- **External validation set**: N = 371

### File locations (included directly in this Git repository)

```text
text_pipeline/pretrained/
├── best_xgboost_model.pkl        # the trained XGBoost classifier (wrapped, see below)
├── feature_engineer.pkl          # fitted imputers, scaler, and label encoders
├── feature_configuration.json    # human-readable feature order / encoding reference
└── SHA256SUMS.txt                # checksums for the three files above
```

### The seven features, in their fixed required order

1. `矫正视力` (corrected visual acuity)
2. `矫正球镜度数(D)` (corrected spherical power, D)
3. `矫正柱镜度数(D)` (corrected cylindrical power, D)
4. `IOLMaster-Cyl(D)`
5. `年龄` (age)
6. `是否配合检查` (cooperation with examination: 是/否)
7. `脱位程度` (dislocation degree: 轻/中/重)

This exact order is enforced by `FeatureEngineer.feature_names` /
`SurgeryClassifier.feature_names` at load time; it is also recorded in
`feature_configuration.json` for reference.

### SHA256 checksums

```text
best_xgboost_model.pkl
27a93862070cd469b241f4c31c6332b4b451a177853f41d5cbb337b02f25c93e

feature_engineer.pkl
9232a304b2db4fdfd2a2f8e26fca274b8d8bc97cad4a7a67b2088f5a15299d50
```

Also available at `text_pipeline/pretrained/SHA256SUMS.txt`.

### Loading example

```python
from text_pipeline.src.models import SurgeryClassifier
from text_pipeline.src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer.load("text_pipeline/pretrained/feature_engineer.pkl")
classifier = SurgeryClassifier.load("text_pipeline/pretrained/best_xgboost_model.pkl")

# X must be a DataFrame with exactly the 7 columns above, in that order.
X, _ = engineer.transform(your_dataframe)
predictions = classifier.predict(X)
probabilities = classifier.predict_proba(X)
```

`SurgeryClassifier.load()` / `FeatureEngineer.load()` transparently handle
these specific frozen files: a plain `pickle.load()` is tried first, and
only on `ModuleNotFoundError` does the code fall back to a narrow,
whitelist-only compatibility loader (`text_pipeline/src/compatibility_loader.py`)
that maps a single legacy class reference to its current equivalent. This
does not affect a freshly trained model's normal save/load round trip.

### Python pickle safety reminder

Pickle files can execute arbitrary code on load. **Only ever load
`best_xgboost_model.pkl` / `feature_engineer.pkl` from this official
repository**, and verify their SHA256 against `SHA256SUMS.txt` above before
loading a copy obtained any other way (e.g., a fork, a mirror, or a
downloaded archive). Do not load pickle files from untrusted sources.

### Missing / unknown categorical values

The frozen `feature_engineer.pkl`'s `LabelEncoder`s for `是否配合检查` and
`脱位程度` were fit only on the real training data, which contained **zero**
missing values in either categorical column. As a result, these encoders do
not recognize a missing/`"nan"` category. If you pass in data with missing
values in either categorical column, `transform()` will raise `ValueError:
y contains previously unseen labels`. Handle missing categorical values
(e.g., drop the row or impute a valid category) before calling
`engineer.transform()` — do not attempt to bypass this by editing the
frozen encoders.

### Evidence of correctness (summary)

- The two `.pkl` files in this repository are byte-identical (SHA256
  match) to the authoritative training run's output files.
- Reloading them via the public pipeline's own `SurgeryClassifier.load()`
  / `FeatureEngineer.load()` and re-running inference on the original
  313-row internal test partition and 371-row external validation set
  reproduces the frozen `y_true`/`y_pred` labels exactly (313/313 and
  371/371) and reproduces `predict_proba()` exactly at the model's native
  float32 precision.

---

## Image model — M0_IMG (ResNet-50, three-class)

- **Model**: ResNet-50 backbone + custom classification head
  (`image_pipeline.src.models.ResNet50Classifier`)
- **Task**: Ophthalmic image severity classification, three classes
- **Labels**:
  - `0` = Mild
  - `1` = Moderate
  - `2` = Severe
- **Internal test set**: N = 574
- **External validation set**: N = 393

### Release files (GitHub Release, not committed to Git history)

Because of size, the image model weights are **not** stored in this Git
repository. They are published as assets on the GitHub Release:

**https://github.com/hethan0721-lgtm/hethan0721-lgtm-CEL-GRADE-Training-Pipeline/releases/tag/model-weights-v1.0.0**

Release assets:

```text
best_model_inference.pth   # inference-only checkpoint
label_mapping.pkl          # {0: '轻', 1: '中', 2: '重'}
SHA256SUMS.txt             # checksums for the two files above
```

### Where the original checkpoint came from

`best_model_inference.pth` was derived from the authoritative training
run's full checkpoint (epoch 25, best validation accuracy **0.7721**) by
removing the optimizer state and training-history buffers, which are only
needed to resume training and are not used for inference. **No model
parameter was modified, renamed, or removed** in this process:

- All 654 tensors in `model_state_dict` (including the unused, vestigial
  1000-class `resnet.fc` layer inherited from ImageNet pretraining) are
  present with identical keys, key order, shapes, dtypes, and values
  (`torch.equal`) between the original checkpoint and this inference-only
  file.
- Loading `best_model_inference.pth` into
  `image_pipeline.src.models.create_model(num_classes=3, pretrained=False)`
  with `model.load_state_dict(..., strict=True)` succeeds with zero
  missing and zero unexpected keys.
- On a fixed-seed synthetic input, the original checkpoint and
  `best_model_inference.pth` produce **bitwise-identical** logits, softmax
  probabilities, and argmax predictions (`torch.equal`, no approximation).

### SHA256 checksums

```text
Original full checkpoint (epoch=25, best_val_acc=0.7721; NOT published as a
Git or Release file due to size — kept internally):
f72897b4f99fd5916bdba81e99ff2557ba4601e2fe78c1bee06cdbd2f3480e1f

best_model_inference.pth (the actual Release asset, 308,390,091 bytes):
52d52beff82ef582ae22c99955cf461bcc394b5435ec7b69c80bfb781d426b47

label_mapping.pkl:
57bdad64003c4e5cf169734bfc5d15937be4c552562205141017d8310fe7a59e
```

### Download and directory layout

Download `best_model_inference.pth` and `label_mapping.pkl` from the
Release above and place them together in the same directory — the loading
code looks for `label_mapping.pkl` next to the checkpoint:

```text
image_pipeline/models/
├── best_model_inference.pth
└── label_mapping.pkl
```

### Loading example

```python
import pickle
import torch

from image_pipeline.src.models import create_model
from image_pipeline.src.config import Config

model = create_model(num_classes=3, input_size=Config.IMAGE_SIZE, pretrained=False)
checkpoint = torch.load("image_pipeline/models/best_model_inference.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

with open("image_pipeline/models/label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)  # {0: '轻', 1: '中', 2: '重'}
```

Verify the downloaded files' SHA256 against the values above (or
`SHA256SUMS.txt` in the Release) before loading.

### Intended use

Both models are research artifacts released for reproducibility and
independent verification of the CEL-GRADE study. They are **not** validated
for, and must not be used as, an independent medical decision-making tool
outside of the original study's clinical validation context.
