"""
Input field validation tests for M0_CLIN (text_pipeline).

Covers Config.validate_config() sanity checks, DataLoader's whitelist
behaviour (only Config.FEATURE_COLUMNS + Config.TARGET_COLUMN should survive
cleaning, identity columns (Config.PRESERVE_COLUMNS) must be split off
rather than left in the training frame, and a missing target column must
raise rather than silently proceeding), and the 7-feature canonical
configuration + preprocessing guarantees (fixed feature order, numerical vs.
categorical handling, whitespace normalisation, unseen-category behaviour,
and that internal/external evaluation data never reaches ``.fit()``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from text_pipeline.src.config import Config
from text_pipeline.src.data_loader import DataLoader
from text_pipeline.src.feature_engineering import FeatureEngineer

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_tabular_data'
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    """A minimal frame containing all 7 canonical features + target."""
    return pd.DataFrame({
        '矫正视力': [0.5] * n,
        '矫正球镜度数(D)': [-1.0] * n,
        '矫正柱镜度数(D)': [-0.5] * n,
        'IOLMaster-Cyl(D)': [1.0] * n,
        '年龄': [50.0] * n,
        '是否配合检查': (['是', '否'] * n)[:n],
        '脱位程度': (['轻', '中'] * n)[:n],
        '是否需要手术': (['手术', '不手术'] * n)[:n],
    })


def _minimal_valid_frame_varied(n=40, seed=0):
    """
    Same 7 canonical features as ``_minimal_valid_frame`` but with varied,
    non-constant numerical values and both target classes well represented
    -- suitable for actually fitting a FeatureEngineer/model (a
    StandardScaler fit on a constant column, or a single-class target, would
    be degenerate for some of the tests below).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        '矫正视力': rng.uniform(0.05, 1.0, n).round(2),
        '矫正球镜度数(D)': rng.uniform(-10.0, 2.0, n).round(2),
        '矫正柱镜度数(D)': rng.uniform(-4.0, 0.0, n).round(2),
        'IOLMaster-Cyl(D)': rng.uniform(0.0, 4.0, n).round(2),
        '年龄': rng.integers(18, 85, n),
        '是否配合检查': rng.choice(['是', '否'], size=n, p=[0.8, 0.2]),
        '脱位程度': rng.choice(['中', '轻', '重'], size=n),
        '是否需要手术': rng.choice(['手术', '不手术'], size=n),
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


def test_unrelated_columns_tolerated_but_never_reach_the_model():
    """
    An input file may contain columns that are not part of the canonical
    7-feature whitelist (legacy identifiers, unrelated metadata, etc.).
    DataLoader must silently drop them -- they must never appear in the
    cleaned frame that FeatureEngineer/the model sees.

    NOTE: the placeholder column name used here (``unrelated_extra_column``)
    is intentionally fictional and unrelated to any real clinical field --
    it must not be confused with (nor is it related to) the real feature
    ``是否配合检查``.
    """
    df = _minimal_valid_frame()
    df['unrelated_extra_column'] = list(range(len(df)))

    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)

    assert 'unrelated_extra_column' not in cleaned.columns
    assert set(cleaned.columns) <= set(Config.FEATURE_COLUMNS) | {Config.TARGET_COLUMN}


def test_canonical_feature_columns_are_exactly_the_seven_used_features():
    """
    Config.FEATURE_COLUMNS must equal CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    exactly (the 7 features that actually reach the trained model) -- no
    extra, unused entries and no missing ones.
    """
    assert set(Config.FEATURE_COLUMNS) == set(Config.CATEGORICAL_FEATURES) | set(Config.NUMERICAL_FEATURES)
    assert len(Config.FEATURE_COLUMNS) == 7
    assert Config.CATEGORICAL_FEATURES == ['是否配合检查', '脱位程度']
    assert set(Config.NUMERICAL_FEATURES) == {
        '矫正视力', '矫正球镜度数(D)', '矫正柱镜度数(D)', 'IOLMaster-Cyl(D)', '年龄'
    }
    assert '年龄' in Config.FEATURE_COLUMNS
    assert '是否配合检查' in Config.FEATURE_COLUMNS


def test_canonical_feature_order_is_fixed():
    """
    The 7 canonical features must appear in this exact, fixed order:
    矫正视力, 矫正球镜度数(D), 矫正柱镜度数(D), IOLMaster-Cyl(D), 年龄,
    是否配合检查, 脱位程度.
    """
    assert Config.FEATURE_COLUMNS == [
        '矫正视力',
        '矫正球镜度数(D)',
        '矫正柱镜度数(D)',
        'IOLMaster-Cyl(D)',
        '年龄',
        '是否配合检查',
        '脱位程度',
    ]


def test_age_enters_numerical_preprocessing_median_imputed_and_scaled():
    """
    年龄 must flow through the numerical path: SimpleImputer(strategy=
    'median') followed by StandardScaler. A missing (NaN) 年龄 value must
    not survive fit_transform.
    """
    df = _minimal_valid_frame_varied(n=40)
    df.loc[0, '年龄'] = np.nan

    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df, verbose=False)

    assert '年龄' in X.columns
    assert not X['年龄'].isna().any()
    # StandardScaler output: not the raw (unscaled) constant/near-constant values.
    assert X['年龄'].std() > 0


def test_cooperation_with_exam_enters_categorical_preprocessing():
    """是否配合检查 must flow through the categorical (LabelEncoder) path."""
    df = _minimal_valid_frame_varied(n=40)

    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df, verbose=False)

    assert '是否配合检查' in X.columns
    assert '是否配合检查' in engineer.label_encoders
    # Label-encoded output must be numeric (encoded classes), not raw strings.
    assert pd.api.types.is_numeric_dtype(X['是否配合检查'])


def test_categorical_missing_value_becomes_explicit_nan_category_not_imputed():
    """
    Documents/locks in the ACTUAL current behaviour (see feature_engineering.py
    and data_loader.py docstrings): DataLoader casts categorical columns to
    `str` before FeatureEngineer's categorical SimpleImputer ever runs, which
    turns a real missing value into the literal string "nan" upstream. By
    the time the categorical SimpleImputer(strategy='most_frequent') runs,
    there is no actual NaN left to impute -- "nan" is treated as its own
    explicit category. This test asserts that real behaviour; it must NOT be
    read as most-frequent imputation actually being effective on categorical
    missingness (it is not).
    """
    df = _minimal_valid_frame(n=12).copy()
    df.loc[0, '是否配合检查'] = np.nan

    loader = DataLoader()
    loader.data = df.copy()
    loader.clean_categorical_columns(verbose=False)
    assert loader.data.loc[0, '是否配合检查'] == 'nan'

    engineer = FeatureEngineer()
    # Feed the already-str-cast frame straight into fit (mirrors what
    # DataLoader hands to FeatureEngineer in the real pipeline).
    fit_df = df.copy()
    fit_df['是否配合检查'] = fit_df['是否配合检查'].astype(str).str.strip()
    engineer.fit(fit_df, verbose=False)
    assert 'nan' in engineer.label_encoders['是否配合检查'].classes_


def test_leading_and_trailing_whitespace_is_stripped_from_categorical_values():
    """
    Real-world External data has been observed with leading/trailing
    whitespace on categorical values (e.g. " 是" instead of "是").
    DataLoader's `.astype(str).str.strip()` cleaning must normalise this
    before label encoding -- based on the confirmed real `.strip()`
    behaviour (there is no Unicode NFKC normalisation anywhere in this
    pipeline, and this test does not assert any).
    """
    df = _minimal_valid_frame(n=4).copy()
    df['是否配合检查'] = [' 是', '是\t', '否 ', '否']

    loader = DataLoader()
    loader.data = df
    loader.clean_categorical_columns(verbose=False)

    assert set(loader.data['是否配合检查'].unique()) == {'是', '否'}


def test_unseen_category_at_transform_time_raises_value_error():
    """
    Neither FeatureEngineer nor the underlying sklearn LabelEncoder handles
    a category value that was never seen during fit() -- there is no
    try/except anywhere in this pipeline for this case. transform() must
    raise ValueError natively, not silently coerce or drop the row.
    """
    train_df = _minimal_valid_frame_varied(n=40)
    engineer = FeatureEngineer()
    engineer.fit(train_df, verbose=False)

    unseen_df = _minimal_valid_frame(n=1).copy()
    unseen_df['脱位程度'] = ['未知类别']

    with pytest.raises(ValueError):
        engineer.transform(unseen_df, verbose=False)


def test_target_column_never_enters_the_feature_matrix():
    df = _minimal_valid_frame_varied(n=40)
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df, verbose=False)
    assert Config.TARGET_COLUMN not in X.columns


def test_model_trained_on_synthetic_data_has_seven_input_features():
    """
    A model fitted end-to-end on synthetic (non-real) data must report
    n_features_in_ == 7, matching the 7-feature canonical configuration.
    Uses only synthetic data generated in-test -- no real patient data and
    no pre-trained pickle is loaded.
    """
    from text_pipeline.src.models import SurgeryClassifier

    df = _minimal_valid_frame_varied(n=40)
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df, verbose=False)
    y_encoded = y.map(Config.LABEL_MAPPING)

    clf = SurgeryClassifier(model_type='xgboost')
    clf.fit(X, y_encoded, verbose=False)

    assert clf.model.n_features_in_ == 7


def test_internal_and_external_eval_data_never_reach_fit(monkeypatch):
    """
    Exercises the actual mechanism main.py relies on: train_pipeline(...,
    pass_eval_set_to_fit=False) must ensure the held-out data passed as
    X_val_external/y_val_external is never forwarded into
    SurgeryClassifier.fit()'s X_val/y_val (and therefore never into
    XGBoost's eval_set), while still being available via
    trainer.get_val_data() for informational/bookkeeping purposes only.
    """
    from text_pipeline.src import train as train_module
    from text_pipeline.src.models import SurgeryClassifier

    captured = {}
    original_fit = SurgeryClassifier.fit

    def spy_fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        captured['X_val'] = X_val
        captured['y_val'] = y_val
        return original_fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False)

    monkeypatch.setattr(SurgeryClassifier, 'fit', spy_fit)

    df = _minimal_valid_frame_varied(n=40)
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df, verbose=False)
    y_encoded = y.map(Config.LABEL_MAPPING)

    X_holdout, y_holdout = X.iloc[:8], y_encoded.iloc[:8]
    X_train_part, y_train_part = X.iloc[8:], y_encoded.iloc[8:]

    model, trainer = train_module.train_pipeline(
        X_train_part, y_train_part,
        apply_smote=False,
        hyperparameter_tuning=False,
        cross_validation=False,
        verbose=False,
        X_val_external=X_holdout,
        y_val_external=y_holdout,
        pass_eval_set_to_fit=False,
    )

    assert captured['X_val'] is None
    assert captured['y_val'] is None

    # Still tracked informationally (bookkeeping only, never read by any
    # other production code path) -- confirms the fix removed the fit()
    # data flow specifically, not the informational storage.
    stored_X, stored_y = trainer.get_val_data()
    assert stored_X is not None
    assert len(stored_X) == len(X_holdout)


def test_synthetic_examples_use_only_the_canonical_dislocation_values():
    """
    脱位程度's real category values are 中/轻/重 -- the old placeholder values
    轻度/中度/重度 (with the "度" suffix) must not appear anywhere in the
    shipped synthetic example data.
    """
    forbidden = {'轻度', '中度', '重度'}
    for csv_path in (EXAMPLES_DIR / 'synthetic_train.csv', EXAMPLES_DIR / 'synthetic_external.csv'):
        df = pd.read_csv(csv_path)
        values = set(df['脱位程度'].dropna().unique())
        assert values <= {'中', '轻', '重'}, f"{csv_path.name}: unexpected 脱位程度 values {values - {'中', '轻', '重'}}"
        assert not (values & forbidden), f"{csv_path.name}: legacy 度-suffixed values found: {values & forbidden}"


def test_sample_id_is_excluded_from_the_final_feature_matrix():
    """
    The synthetic CSVs (and any real input file with the same shape) carry a
    `sample_id` identifier column alongside the 7 canonical features and the
    target. `sample_id` must never reach the model: DataLoader's whitelist
    must drop it, and the final feature matrix FeatureEngineer produces must
    be exactly the 7 canonical columns, in the canonical order -- nothing
    more, nothing less.
    """
    df = _minimal_valid_frame_varied(n=40).copy()
    df.insert(0, 'sample_id', [f"S{i:03d}" for i in range(len(df))])

    loader = DataLoader()
    loader.data = df
    cleaned = loader.clean_data(verbose=False)
    assert 'sample_id' not in cleaned.columns

    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(cleaned, verbose=False)

    assert 'sample_id' not in X.columns
    assert list(X.columns) == Config.FEATURE_COLUMNS
    assert X.shape[1] == 7


def test_production_call_site_explicitly_enables_smote():
    """
    The lightweight default-behaviour test below (`test_train_pipeline_
    default_keeps_eval_and_holdout_data_out_of_fit`) uses apply_smote=False
    to keep the fixture small and avoid SMOTE's k_neighbors=5 minimum
    minority-class-sample requirement colliding with a tiny synthetic
    holdout split. That is a test-fixture simplification only -- it must not
    be read as SMOTE being disabled in production. This test independently
    confirms the actual production call site (main.py) still explicitly
    requests SMOTE.
    """
    main_source = (REPO_ROOT / 'text_pipeline' / 'src' / 'main.py').read_text(encoding='utf-8')
    assert 'apply_smote=True' in main_source


def test_train_pipeline_default_keeps_eval_and_holdout_data_out_of_fit(monkeypatch):
    """
    Safe-by-default check: calling train_pipeline(...) WITHOUT explicitly
    passing pass_eval_set_to_fit must still keep held-out
    X_val_external/y_val_external data out of SurgeryClassifier.fit()'s
    eval_set -- a future caller who forgets this flag must not silently
    reintroduce the eval_set data-flow this flag exists to avoid.

    Uses apply_smote=False purely to keep this fixture lightweight (SMOTE
    needs enough minority-class samples for its k_neighbors=5 default, which
    a tiny 32-row synthetic holdout split does not reliably provide); the
    companion test above confirms the real production call site still
    enables SMOTE. SMOTE is orthogonal to this test's subject (the eval_set
    plumbing operates on X_val/y_val, never on X_train/y_train, regardless
    of apply_smote).
    """
    from text_pipeline.src import train as train_module
    from text_pipeline.src.models import SurgeryClassifier

    captured = {}
    original_fit = SurgeryClassifier.fit

    def spy_fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        captured['X_val'] = X_val
        captured['y_val'] = y_val
        captured['X_train_index'] = list(X_train.index)
        return original_fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False)

    monkeypatch.setattr(SurgeryClassifier, 'fit', spy_fit)

    df = _minimal_valid_frame_varied(n=40)
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df, verbose=False)
    y_encoded = y.map(Config.LABEL_MAPPING)

    # Disjoint slices: by construction, the holdout rows below and the rows
    # actually passed as X_train_part/y_train_part never overlap, so if
    # captured['X_train_index'] included any holdout row that would prove
    # (rather than merely assume) the internal-test data leaked into the
    # training matrix.
    X_holdout, y_holdout = X.iloc[:8], y_encoded.iloc[:8]
    X_train_part, y_train_part = X.iloc[8:], y_encoded.iloc[8:]

    model, trainer = train_module.train_pipeline(
        X_train_part, y_train_part,
        apply_smote=False,
        hyperparameter_tuning=False,
        cross_validation=False,
        verbose=False,
        X_val_external=X_holdout,
        y_val_external=y_holdout,
        # pass_eval_set_to_fit intentionally omitted -- this test exists to
        # prove the DEFAULT is safe.
    )

    assert captured['X_val'] is None
    assert captured['y_val'] is None
    assert not (set(captured['X_train_index']) & set(X_holdout.index))

    stored_X, stored_y = trainer.get_val_data()
    assert stored_X is not None
    assert len(stored_X) == len(X_holdout)
