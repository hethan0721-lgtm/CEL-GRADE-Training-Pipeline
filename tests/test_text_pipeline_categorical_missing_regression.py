"""
Regression test for the historical categorical missing-value behaviour
(text_pipeline). Documented in text_pipeline/README.md and in code comments
in data_loader.py / feature_engineering.py: DataLoader casts categorical
columns to `str` before FeatureEngineer's categorical SimpleImputer ever
runs, so a real missing value becomes the literal string "nan" and is
label-encoded as its own explicit category -- it is NOT most-frequent
-imputed in practice, despite `Config.FILL_STRATEGY['categorical']`.

This pass intentionally preserves that behaviour (changing it would change
training results). This test exists to catch an accidental future change to
the fit order that would silently make the categorical imputer effective
and therefore change what the model is trained on.
"""

import numpy as np
import pandas as pd

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader
from text_pipeline.src.feature_engineering import FeatureEngineer


def _frame_with_missing_dislocation(n=30, missing_fraction=0.3):
    dislocation = np.array((['轻度', '中度', '重度'] * n)[:n], dtype=object)
    n_missing = int(n * missing_fraction)
    dislocation[:n_missing] = np.nan

    return pd.DataFrame({
        '脱位程度': dislocation,
        '矫正视力': [0.5] * n,
        '矫正球镜度数(D)': [-1.0] * n,
        '矫正柱镜度数(D)': [-0.5] * n,
        'IOLMaster-Cyl(D)': [1.0] * n,
        '是否需要手术': (['手术', '不手术'] * n)[:n],
    })


def test_missing_categorical_values_become_the_literal_string_nan_after_cleaning():
    df = _frame_with_missing_dislocation()
    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    # The real NaNs must have survived cleaning as the literal string "nan",
    # not been dropped, not been most-frequent-imputed to '轻度'/'中度'/'重度'.
    assert (cleaned['脱位程度'] == 'nan').sum() > 0
    assert cleaned['脱位程度'].isna().sum() == 0  # no actual NaN remains; it's a string now


def test_categorical_imputer_is_a_no_op_and_nan_is_a_real_label_class():
    df = _frame_with_missing_dislocation()
    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    engineer = FeatureEngineer()
    engineer.fit(cleaned, verbose=False)

    # SimpleImputer(strategy='most_frequent') fit on already-stringified data
    # sees no NaN, so its learned statistic reflects "nan" being present as
    # an ordinary category rather than something to impute away.
    assert 'nan' in engineer.label_encoders['脱位程度'].classes_

    X, y = engineer.transform(cleaned, verbose=False)
    encoded_nan_label = engineer.label_encoders['脱位程度'].transform(['nan'])[0]
    # Some rows in the transformed feature matrix are literally the "nan"
    # category's encoded value -- i.e. missingness reached the model as an
    # explicit category, not as an imputed real category.
    assert (X['脱位程度'] == encoded_nan_label).sum() > 0


def test_fill_strategy_categorical_config_does_not_claim_effective_imputation():
    """
    Config.FILL_STRATEGY['categorical'] is kept for structural/pickle
    compatibility only; this test just pins its value so a future edit that
    silently changes it (implying it became effective) gets caught.
    """
    assert Config.FILL_STRATEGY['categorical'] == 'most_frequent'
