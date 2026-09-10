"""
Input field validation tests for M0_CLIN (text_pipeline).

Covers Config.validate_config() sanity checks and DataLoader's whitelist
behaviour: only Config.FEATURE_COLUMNS + Config.TARGET_COLUMN should survive
cleaning, identity columns (Config.PRESERVE_COLUMNS) must be split off
rather than left in the training frame, and a missing target column must
raise rather than silently proceeding.
"""

import pandas as pd
import pytest

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader


def test_validate_config_accepts_the_shipped_defaults():
    assert Config.validate_config() is True


def test_validate_config_rejects_out_of_range_test_size():
    class BadConfig(Config):
        TEST_SIZE = 1.5

    with pytest.raises(ValueError):
        BadConfig.validate_config()


def test_validate_config_rejects_non_positive_xgboost_params():
    class BadConfig(Config):
        XGBOOST_PARAMS = {**Config.XGBOOST_PARAMS, 'max_depth': 0}

    with pytest.raises(ValueError):
        BadConfig.validate_config()


def _minimal_valid_frame(n=12):
    return pd.DataFrame({
        '脱位程度': (['轻度', '中度'] * n)[:n],
        '矫正视力': [0.5] * n,
        '矫正球镜度数(D)': [-1.0] * n,
        '矫正柱镜度数(D)': [-0.5] * n,
        'IOLMaster-Cyl(D)': [1.0] * n,
        '是否需要手术': (['手术', '不手术'] * n)[:n],
    })


def test_clean_data_raises_without_target_column():
    df = _minimal_valid_frame().drop(columns=[Config.TARGET_COLUMN])
    loader = DataLoader()
    loader.data = df
    with pytest.raises(ValueError):
        loader.clean_data(verbose=False)


def test_clean_data_drops_columns_outside_the_feature_whitelist():
    df = _minimal_valid_frame()
    df['姓名'] = ['someone'] * len(df)
    df['unrelated_metadata_column'] = list(range(len(df)))

    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    allowed = set(Config.FEATURE_COLUMNS) | {Config.TARGET_COLUMN}
    assert set(cleaned.columns) <= allowed
    assert '姓名' not in cleaned.columns
    assert 'unrelated_metadata_column' not in cleaned.columns


def test_clean_data_splits_off_preserve_columns_rather_than_keeping_them():
    df = _minimal_valid_frame()
    df['姓名'] = ['someone'] * len(df)
    df['身份证号'] = ['110101199001011234'] * len(df)
    df['省份'] = ['A省'] * len(df)

    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    for col in Config.PRESERVE_COLUMNS:
        assert col not in cleaned.columns

    preserved = loader.get_preserved_data()
    assert preserved is not None
    assert set(Config.PRESERVE_COLUMNS) <= set(preserved.columns)


def test_clean_data_encodes_target_with_label_mapping():
    df = _minimal_valid_frame()
    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)
    assert set(cleaned[Config.TARGET_COLUMN].unique()) <= set(Config.LABEL_MAPPING.values())


def test_legacy_columns_tolerated_but_never_reach_the_model():
    """
    是否配合/年龄 never entered the historical canonical model. An input
    file MAY still contain them (for backward compatibility with older raw
    exports), but DataLoader must silently drop them -- they must never
    appear in the cleaned frame that FeatureEngineer/the model sees.
    """
    df = _minimal_valid_frame()
    df['是否配合'] = ['配合', '不配合'] * (len(df) // 2)
    df['年龄'] = list(range(len(df)))

    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    assert '是否配合' not in cleaned.columns
    assert '年龄' not in cleaned.columns
    assert set(cleaned.columns) <= set(Config.FEATURE_COLUMNS) | {Config.TARGET_COLUMN}


def test_canonical_feature_columns_are_exactly_the_five_used_features():
    """
    Config.FEATURE_COLUMNS must equal CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    exactly (the 5 features that actually reach the trained model) -- no
    extra, unused entries (is否配合/年龄) and no missing ones.
    """
    assert set(Config.FEATURE_COLUMNS) == set(Config.CATEGORICAL_FEATURES) | set(Config.NUMERICAL_FEATURES)
    assert len(Config.FEATURE_COLUMNS) == 5
    assert Config.CATEGORICAL_FEATURES == ['脱位程度']
    assert set(Config.NUMERICAL_FEATURES) == {
        '矫正视力', '矫正球镜度数(D)', '矫正柱镜度数(D)', 'IOLMaster-Cyl(D)'
    }
    assert '是否配合' not in Config.FEATURE_COLUMNS
    assert '年龄' not in Config.FEATURE_COLUMNS
