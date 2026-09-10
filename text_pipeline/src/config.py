#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for the surgery classification system.
Defines data paths, feature sets, model parameters, and training options.
"""

from pathlib import Path


class Config:
    """Configuration holder for all training and model parameters."""

    # Data files
    TRAIN_DATA_FILE = 'new_data/文本-N.xlsx'  # train
    VAL_DATA_FILE = 'new_data/文本-外.xlsx'      # val

    DATA_FILES = [
        'new_data/train.xlsx',
    ]

    # Target and preserved columns
    TARGET_COLUMN = '是否需要手术'

    PRESERVE_COLUMNS = ['姓名', '身份证号', '省份']

    # Categorical features actually used by FeatureEngineer/the trained model.
    CATEGORICAL_FEATURES = ['脱位程度']

    # Numerical features actually used by FeatureEngineer/the trained model.
    NUMERICAL_FEATURES = [
        '矫正视力',
        '矫正球镜度数(D)',
        '矫正柱镜度数(D)',
        'IOLMaster-Cyl(D)'
    ]

    # Canonical feature whitelist: the 5 features that actually reach the
    # trained XGBoost model (CATEGORICAL_FEATURES + NUMERICAL_FEATURES).
    # `是否配合` and `年龄` are deliberately NOT listed here -- they never
    # entered the historical canonical model, and must not be added back.
    # `DataLoader` drops any input column that is not in this list (and not
    # in PRESERVE_COLUMNS), so an input spreadsheet MAY contain extra
    # columns (是否配合, 年龄, legacy identifiers, etc.) for backward
    # compatibility with older raw files -- they are simply ignored and can
    # never reach the model.
    FEATURE_COLUMNS = [
        '脱位程度',
        '矫正视力',
        '矫正球镜度数(D)',
        '矫正柱镜度数(D)',
        'IOLMaster-Cyl(D)'
    ]

    # Feature importance order (must match FEATURE_COLUMNS -- see the note above).
    FEATURE_WEIGHT_INFO = {
        'description': 'Feature importance ranking',
        'order': [
            '脱位程度',
            '矫正视力',
            '矫正球镜度数(D)',
            '矫正柱镜度数(D)',
            'IOLMaster-Cyl(D)'
        ]
    }

    # Label mapping
    LABEL_MAPPING = {
        '不手术': 0,
        '手术': 1
    }

    LABEL_NAMES = {
        0: '不手术',
        1: '手术'
    }

    # English label names for plotting
    LABEL_NAMES_EN = {
        0: 'Non-Surgery',
        1: 'Surgery'
    }

    # Chinese to English feature name mapping for plotting
    FEATURE_NAME_MAPPING = {
        '脱位程度': 'Dislocation Degree',
        '矫正视力': 'Corrected Vision',
        '矫正球镜度数(D)': 'Spherical Power(D)',
        '矫正柱镜度数(D)': 'Cylindrical Power(D)',
        'IOLMaster-Cyl(D)': 'IOLMaster-Cyl(D)'
    }

    # Preprocessing.
    # NOTE (historical behaviour, intentionally preserved -- do not "fix"):
    # DataLoader casts categorical columns to `str` (clean_categorical_columns
    # / _clean_dataframe) BEFORE FeatureEngineer's categorical SimpleImputer
    # ever runs. That cast turns real missing values into the literal string
    # "nan", so by the time the categorical SimpleImputer(strategy=
    # 'most_frequent') below runs, there is no actual NaN left for it to
    # impute -- "nan" is treated as its own explicit category and label-
    # encoded as such. FILL_STRATEGY['categorical'] and the categorical
    # imputer are kept for historical/structural compatibility with the
    # existing saved pipeline shape; they are NOT reliably performing
    # most-frequent imputation on categorical missingness. See
    # text_pipeline/README.md for the full explanation. This pass
    # deliberately does not change the fit order to make the imputer
    # effective, since that would change training results.
    FILL_STRATEGY = {
        'numerical': 'median',
        'categorical': 'most_frequent'
    }

    # Dataset split (仅在未使用独立验证集时生效)
    TEST_SIZE = 0.2
    VAL_SIZE = 0
    RANDOM_STATE = 42

    # 是否使用独立的验证/测试集文件
    USE_SEPARATE_VAL_FILE = True

    # Imbalanced data handling
    SMOTE_CONFIG = {
        'enabled': True,
        'sampling_strategy': 'auto',
        'k_neighbors': 5,
        'random_state': 42
    }

    USE_CLASS_WEIGHT = True

    # XGBoost configuration
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }

    # NOTE: an `EARLY_STOPPING_ROUNDS` setting previously existed here but was
    # never actually wired into SurgeryClassifier's estimator construction or
    # its `.fit()` call -- it had no runtime effect. It has been removed
    # rather than fixed: the canonical model has never used early stopping,
    # and this pass does not add it (that would change training results).
    # `XGBOOST_PARAMS['n_estimators'] = 200` above always trains to
    # completion with no early stopping.

    # Legacy hyperparameter-search helper (ModelTrainer.hyperparameter_tuning
    # in train.py). NOT part of the canonical training path: main.py always
    # calls train_pipeline(..., hyperparameter_tuning=False), so the
    # canonical model is always trained with the single fixed XGBOOST_PARAMS
    # set above, never via RandomizedSearchCV. ENABLE_HYPERPARAMETER_TUNING
    # defaults to False so this legacy path is never invoked by default; it
    # is retained only for manual/offline experimentation and must not be
    # flipped on as part of the canonical public pipeline.
    ENABLE_HYPERPARAMETER_TUNING = False
    TUNING_METHOD = 'random'

    # RandomizedSearchCV parameters (legacy helper only -- see note above)
    RANDOM_SEARCH_PARAMS = {
        'n_iter': 50,
        'cv': 5,
        'scoring': 'f1',
        'n_jobs': -1,
        'verbose': 2,
        'random_state': 42
    }

    # Parameter search space (legacy helper only -- see note above)
    PARAM_GRID = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.05, 0.1],
        'reg_lambda': [0.5, 1.0, 1.5]
    }

    # Cross-validation
    CV_FOLDS = 5
    STRATIFIED = True

    # Output paths (relative to the text_pipeline directory; overridable via CLI --output-dir)
    MODEL_SAVE_DIR = 'outputs/models'
    RESULTS_SAVE_DIR = 'outputs/results'

    BEST_MODEL_NAME = 'best_xgboost_model.pkl'
    FEATURE_IMPORTANCE_NAME = 'feature_importance.png'

    # Evaluation metrics
    EVALUATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc'
    ]

    # Plotting config
    PLOT_CONFIG = {
        'style': 'seaborn-v0_8-darkgrid',
        'figure_size': (12, 8),
        'dpi': 150,
        'font_size': 12,
        'title_size': 16
    }

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

    # Misc
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    SAVE_INTERMEDIATE_RESULTS = True

    @classmethod
    def get_absolute_paths(cls, base_dir=None):
        """Return absolute paths for data, model, and result directories.

        Uses ``pathlib.Path`` so relative defaults resolve correctly on both
        Windows and Linux. If any of the configured path attributes is
        already absolute (e.g. supplied via the CLI), joining leaves it
        untouched (``pathlib`` drops the left-hand side when the right-hand
        side is absolute), so runtime overrides always win over the default.
        """
        if base_dir is None:
            # Resolve the project root (the parent directory of this module, i.e. text_pipeline/)
            base_dir = Path(__file__).resolve().parent.parent
        else:
            base_dir = Path(base_dir)

        return {
            'data_files': [str(base_dir / f) for f in cls.DATA_FILES],
            'train_data_file': str(base_dir / cls.TRAIN_DATA_FILE),
            'val_data_file': str(base_dir / cls.VAL_DATA_FILE),
            'model_save_dir': str(base_dir / cls.MODEL_SAVE_DIR),
            'results_save_dir': str(base_dir / cls.RESULTS_SAVE_DIR)
        }

    @classmethod
    def validate_config(cls):
        """Validate configuration values and raise if invalid."""
        errors = []

        # Validate proportions
        if not 0 < cls.TEST_SIZE < 1:
            errors.append("TEST_SIZE must be between 0 and 1")

        if not 0 <= cls.VAL_SIZE < 1:
            errors.append("VAL_SIZE must be between 0 and 1")

        if cls.TEST_SIZE + cls.VAL_SIZE >= 1:
            errors.append("TEST_SIZE + VAL_SIZE must be less than 1")

        # Validate critical XGBoost parameters
        if cls.XGBOOST_PARAMS['max_depth'] <= 0:
            errors.append("max_depth must be greater than 0")

        if cls.XGBOOST_PARAMS['learning_rate'] <= 0:
            errors.append("learning_rate must be greater than 0")

        if cls.XGBOOST_PARAMS['n_estimators'] <= 0:
            errors.append("n_estimators must be greater than 0")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    @classmethod
    def print_config(cls):
        """Pretty-print the current configuration."""
        print("\n" + "=" * 70)
        print(" " * 20 + "Surgery Classification Configuration")
        print("=" * 70)

        print(f"\n[Data Configuration]")
        print(f"  Data files: {len(cls.DATA_FILES)}")
        print(f"  Target column: {cls.TARGET_COLUMN}")
        print(f"  Categorical features: {cls.CATEGORICAL_FEATURES}")
        print(f"  Numerical features: {cls.NUMERICAL_FEATURES}")

        print(f"\n[Model Configuration]")
        print(f"  Model type: XGBoost")
        print(f"  Max depth: {cls.XGBOOST_PARAMS['max_depth']}")
        print(f"  Learning rate: {cls.XGBOOST_PARAMS['learning_rate']}")
        print(f"  Number of trees: {cls.XGBOOST_PARAMS['n_estimators']}")

        print(f"\n[Imbalanced Learning]")
        print(f"  SMOTE enabled: {cls.SMOTE_CONFIG['enabled']}")
        print(f"  Use class weight: {cls.USE_CLASS_WEIGHT}")

        print(f"\n[Training Configuration]")
        print(f"  Test size: {cls.TEST_SIZE}")
        print(f"  Validation size: {cls.VAL_SIZE}")
        print(f"  CV folds: {cls.CV_FOLDS}")
        print(f"  Early stopping: disabled (not wired into training; n_estimators={cls.XGBOOST_PARAMS['n_estimators']} always runs to completion)")
        print(f"  Hyper-parameter tuning: {cls.ENABLE_HYPERPARAMETER_TUNING} (legacy helper; canonical model always uses fixed XGBOOST_PARAMS)")

        print(f"\n[Output Paths]")
        print(f"  Models directory: {cls.MODEL_SAVE_DIR}")
        print(f"  Results directory: {cls.RESULTS_SAVE_DIR}")

        print("=" * 70 + "\n")


# 创建默认配置实例
config = Config()
