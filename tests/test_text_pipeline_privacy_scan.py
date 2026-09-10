"""
Privacy field output scan test for M0_CLIN (text_pipeline).

Two things are checked:
  1. The bundled synthetic example files never contain a directly
     re-identifying column (name / ID-card number / medical record number /
     phone / address / province) or a value that looks like a Chinese
     ID-card number or mobile phone number.
  2. The real DataLoader cleaning path actually strips
     Config.PRESERVE_COLUMNS (姓名/身份证号/省份) out of the frame that
     reaches feature engineering -- i.e. even if a raw input file contains
     identity columns, they cannot flow into the model or its outputs.
"""

import re
from pathlib import Path

import pandas as pd

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_tabular_data'

FORBIDDEN_COLUMNS = {
    '姓名', '身份证号', '身份证', '病历号', '病历号码', '电话', '联系电话',
    '手机号', '地址', '家庭地址', '省份', '患者id', '患者ID', 'patient_id', 'name',
}

ID_CARD_RE = re.compile(r'(?<![\d.])\d{17}[\dXx](?!\d)')  # 18-digit Chinese ID card (not a float fragment)
PHONE_RE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')      # Chinese mobile number


def _iter_example_data_files():
    for path in EXAMPLES_DIR.rglob('*.xlsx'):
        yield path
    for path in EXAMPLES_DIR.rglob('*.csv'):
        yield path


def test_synthetic_examples_have_no_forbidden_identity_columns():
    checked_any = False
    for path in _iter_example_data_files():
        checked_any = True
        df = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
        lowered_cols = {str(c).strip().lower() for c in df.columns}
        offending = {c for c in FORBIDDEN_COLUMNS if c.lower() in lowered_cols}
        assert not offending, f"{path.name} contains forbidden identity columns: {offending}"
    assert checked_any, "No synthetic example data files were found to scan"


def test_synthetic_examples_have_no_id_card_or_phone_like_values():
    for path in _iter_example_data_files():
        df = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
        blob = df.astype(str).to_csv(index=False)
        assert not ID_CARD_RE.search(blob), f"{path.name} contains an ID-card-number-like value"
        assert not PHONE_RE.search(blob), f"{path.name} contains a phone-number-like value"


def test_preserve_columns_are_stripped_before_features_are_built():
    n = 12
    df = pd.DataFrame({
        '姓名': ['受试者'] * n,
        '身份证号': ['110101199001011234'] * n,
        '省份': ['A省'] * n,
        '脱位程度': (['轻度', '中度'] * n)[:n],
        '矫正视力': [0.5] * n,
        '矫正球镜度数(D)': [-1.0] * n,
        '矫正柱镜度数(D)': [-0.5] * n,
        'IOLMaster-Cyl(D)': [1.0] * n,
        '是否需要手术': (['手术', '不手术'] * n)[:n],
    })

    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    for col in Config.PRESERVE_COLUMNS:
        assert col not in cleaned.columns, (
            f"Identity column {col!r} leaked into the cleaned training frame"
        )

    # The identity values are captured separately but must never be merged
    # back into the feature/label frame that reaches the model.
    preserved = loader.get_preserved_data()
    assert preserved is not None
    assert set(preserved.columns) == set(Config.PRESERVE_COLUMNS)
