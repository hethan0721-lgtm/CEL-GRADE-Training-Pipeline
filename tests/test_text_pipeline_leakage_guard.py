"""
Data-leakage guard tests for M0_CLIN (text_pipeline).

Verifies the core scientific-integrity invariant this pipeline depends on:
imputers/scaler/encoders are fit ONLY on the training partition, and calling
.transform() on held-out data never refits them. This is a regression test
for the 2026-08-21 leakage fix described in text_pipeline/README.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader
from text_pipeline.src.feature_engineering import FeatureEngineer

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_tabular_data'
TRAIN_FILE = EXAMPLES_DIR / 'synthetic_train.csv'
EXTERNAL_FILE = EXAMPLES_DIR / 'synthetic_external.csv'


def _synthetic_config():
    class SyntheticConfig(Config):
        TRAIN_DATA_FILE = str(TRAIN_FILE)
        VAL_DATA_FILE = str(EXTERNAL_FILE)

    return SyntheticConfig()


def _split_train_80_test_20(config):
    loader = DataLoader(config=config)
    loader.load_train_val_data(verbose=False)
    train_data, _ = loader.clean_train_val_data(verbose=False)

    train_80, test_20 = train_test_split(
        train_data,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_data[config.TARGET_COLUMN],
    )
    return train_80.reset_index(drop=True), test_20.reset_index(drop=True)


def test_transform_on_holdout_does_not_refit_imputer_or_scaler():
    config = _synthetic_config()
    train_80, test_20 = _split_train_80_test_20(config)

    engineer = FeatureEngineer(config=config)
    engineer.fit(train_80, verbose=False)

    imputer_stats_before = engineer.numerical_imputer.statistics_.copy()
    scaler_mean_before = engineer.scaler.mean_.copy()
    scaler_scale_before = engineer.scaler.scale_.copy()

    # transform-only call on held-out data -- must be a pure read
    engineer.transform(test_20, verbose=False)

    assert np.allclose(engineer.numerical_imputer.statistics_, imputer_stats_before)
    assert np.allclose(engineer.scaler.mean_, scaler_mean_before)
    assert np.allclose(engineer.scaler.scale_, scaler_scale_before)


def test_fitted_statistics_reflect_only_the_train_partition():
    config = _synthetic_config()
    train_80, test_20 = _split_train_80_test_20(config)

    engineer = FeatureEngineer(config=config)
    engineer.fit(train_80, verbose=False)

    num_cols = [c for c in config.NUMERICAL_FEATURES if c in train_80.columns]
    expected_median = train_80[num_cols].median().values
    assert np.allclose(engineer.numerical_imputer.statistics_, expected_median, atol=1e-6)


def test_train_and_test_partitions_yield_different_fitted_statistics():
    """
    Sanity check that the test above is actually discriminating: fitting
    separately on the train partition vs. the test partition must NOT
    produce the same statistics (otherwise the previous assertion could
    pass by coincidence rather than by correctly scoping the fit).
    """
    config = _synthetic_config()
    train_80, test_20 = _split_train_80_test_20(config)

    engineer_train = FeatureEngineer(config=config)
    engineer_train.fit(train_80, verbose=False)

    engineer_test = FeatureEngineer(config=config)
    engineer_test.fit(test_20, verbose=False)

    assert not np.allclose(
        engineer_train.numerical_imputer.statistics_,
        engineer_test.numerical_imputer.statistics_,
    )


def test_label_encoder_classes_come_only_from_train_partition():
    config = _synthetic_config()
    train_80, test_20 = _split_train_80_test_20(config)

    engineer = FeatureEngineer(config=config)
    engineer.fit(train_80, verbose=False)

    for col, encoder in engineer.label_encoders.items():
        train_categories = set(train_80[col].astype(str).unique())
        assert set(encoder.classes_) <= train_categories | {'nan'}
