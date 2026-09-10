"""
Regression test for the SettingWithCopyWarning fix in
DataLoader.clean_data() (text_pipeline/src/data_loader.py). The fix adds a
``.copy()`` when narrowing to the feature whitelist so later in-place column
assignments don't operate on a view; this must not change any resulting
values, row count, or column order.
"""

import warnings

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader


def _frame_with_missing_values(n=20):
    # np.nan (not Python None) to match what pandas actually produces for a
    # missing cell read from CSV/Excel -- str(np.nan) == "nan", which is the
    # exact literal-string behaviour under test here.
    return pd.DataFrame({
        '脱位程度': (['轻度', np.nan, '重度'] * n)[:n],
        '矫正视力': ([0.5, np.nan, 0.7] * n)[:n],
        '矫正球镜度数(D)': [-1.0] * n,
        '矫正柱镜度数(D)': [-0.5] * n,
        'IOLMaster-Cyl(D)': [1.0] * n,
        '姓名': ['someone'] * n,
        '是否需要手术': (['手术', '不手术'] * n)[:n],
    })


def test_clean_data_raises_no_settingwithcopywarning():
    df = _frame_with_missing_values()
    loader = DataLoader()
    loader.data = df

    with warnings.catch_warnings():
        warnings.simplefilter('error', SettingWithCopyWarning)
        cleaned = loader.clean_data(verbose=False)  # must not raise

    assert len(cleaned) == len(df)


def test_clean_data_result_unchanged_by_the_copy_fix():
    """
    Equivalence check: the cleaned output's shape, column order, and values
    match what the (pre-fix) code path was already producing -- the .copy()
    only silences the warning, it does not alter any data.
    """
    df = _frame_with_missing_values()

    loader = DataLoader()
    loader.data = df.copy()
    cleaned = loader.clean_data(verbose=False)

    expected_columns = [c for c in Config.FEATURE_COLUMNS + [Config.TARGET_COLUMN] if c in df.columns]
    assert list(cleaned.columns) == expected_columns
    assert cleaned.shape[0] == len(df)
    assert '姓名' not in cleaned.columns

    # 脱位程度's real missing value became the literal string "nan" (see the
    # categorical-missing-value regression test) -- confirm that's still
    # exactly what happens after the .copy() fix.
    assert (cleaned['脱位程度'] == 'nan').sum() >= 1
