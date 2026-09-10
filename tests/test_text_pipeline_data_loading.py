"""
Synthetic data reading test for M0_CLIN (text_pipeline).

Confirms the bundled fully-synthetic example files
(examples/synthetic_tabular_data/synthetic_{train,external}.xlsx) load and
clean successfully through the real DataLoader path, with both target
classes present (required for the stratified split and SMOTE).
"""

from pathlib import Path

import pytest

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_tabular_data'
TRAIN_FILE = EXAMPLES_DIR / 'synthetic_train.csv'
EXTERNAL_FILE = EXAMPLES_DIR / 'synthetic_external.csv'


def _synthetic_config():
    class SyntheticConfig(Config):
        TRAIN_DATA_FILE = str(TRAIN_FILE)
        VAL_DATA_FILE = str(EXTERNAL_FILE)

    return SyntheticConfig()


def test_synthetic_example_files_exist():
    assert TRAIN_FILE.exists(), (
        "Missing examples/synthetic_tabular_data/synthetic_train.csv -- "
        "run generate_synthetic_data.py in that directory."
    )
    assert EXTERNAL_FILE.exists(), (
        "Missing examples/synthetic_tabular_data/synthetic_external.csv -- "
        "run generate_synthetic_data.py in that directory."
    )


@pytest.fixture
def cleaned_synthetic_data():
    loader = DataLoader(config=_synthetic_config())
    loader.load_train_val_data(verbose=False)
    train_data, val_data = loader.clean_train_val_data(verbose=False)
    return train_data, val_data


def test_synthetic_data_loads_and_cleans(cleaned_synthetic_data):
    train_data, val_data = cleaned_synthetic_data
    assert len(train_data) > 0
    assert len(val_data) > 0

    allowed = set(Config.FEATURE_COLUMNS) | {Config.TARGET_COLUMN}
    assert set(train_data.columns) <= allowed
    assert set(val_data.columns) <= allowed


def test_synthetic_data_has_both_target_classes(cleaned_synthetic_data):
    train_data, val_data = cleaned_synthetic_data
    assert train_data[Config.TARGET_COLUMN].nunique() == 2
    assert val_data[Config.TARGET_COLUMN].nunique() == 2

    # Enough of the minority class that an 80/20 split plus 5-fold CV plus
    # SMOTE (k_neighbors=5) all have enough rows to work with.
    min_class_count = train_data[Config.TARGET_COLUMN].value_counts().min()
    assert min_class_count >= 20
