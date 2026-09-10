#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature engineering module responsible for preprocessing, encoding,
scaling, and related transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

from .config import Config


class FeatureEngineer:
    """Encapsulates the feature engineering workflow for the project."""

    def __init__(self, config=None):
        """
        Initialize the feature engineering pipeline.

        Args:
            config: Optional configuration object.
        """
        self.config = config if config is not None else Config()

        # Preprocessing artifacts
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}

        # Feature name book-keeping
        self.feature_names = []
        self.original_features = []

        # Flag indicating whether the pipeline has been fitted
        self.is_fitted = False

    def fit(self, data, verbose=True):
        """
        Fit all feature engineering components using the provided dataset.

        Args:
            data: Training dataset.
            verbose: Whether to print progress information.

        Returns:
            self
        """
        if verbose:
            print("\n" + "="*70)
            print(" " * 23 + "Fitting Feature Engineer")
            print("="*70)

        # 1. Handle numerical features
        if verbose:
            print("\n1. Fitting numerical preprocessors")

        numerical_cols = [col for col in self.config.NUMERICAL_FEATURES if col in data.columns]

        if len(numerical_cols) > 0:
            # Missing-value imputer
            self.numerical_imputer = SimpleImputer(strategy=self.config.FILL_STRATEGY['numerical'])
            self.numerical_imputer.fit(data[numerical_cols])

            # Standard scaler
            self.scaler = StandardScaler()
            imputed_data = self.numerical_imputer.transform(data[numerical_cols])
            self.scaler.fit(imputed_data)

            if verbose:
                print(f"  ✓ Numerical features: {len(numerical_cols)}")
                print(f"    Strategy: {self.config.FILL_STRATEGY['numerical']} imputation + standardisation")

        # 2. Handle categorical features
        if verbose:
            print("\n2. Fitting categorical encoders")

        categorical_cols = [col for col in self.config.CATEGORICAL_FEATURES if col in data.columns]

        if len(categorical_cols) > 0:
            # Historical categorical "imputer" -- kept for structural/pickle
            # compatibility with the existing saved pipeline shape, but it is
            # NOT effective most-frequent imputation in practice: DataLoader
            # already cast these columns to `str` upstream, so real missing
            # values have already become the literal string "nan" by the
            # time this SimpleImputer ever sees the data. There is no actual
            # NaN left for it to impute; "nan" simply passes through and is
            # label-encoded as its own explicit category below. This is the
            # historical, intentionally-preserved behaviour (not a bug this
            # pass fixes) -- see text_pipeline/README.md.
            self.categorical_imputer = SimpleImputer(strategy=self.config.FILL_STRATEGY['categorical'])
            self.categorical_imputer.fit(data[categorical_cols])

            # Label encoders
            imputed_data = self.categorical_imputer.transform(data[categorical_cols])
            imputed_df = pd.DataFrame(imputed_data, columns=categorical_cols)

            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(imputed_df[col].astype(str))
                self.label_encoders[col] = le

            if verbose:
                print(f"  ✓ Categorical features: {len(categorical_cols)}")
                for col in categorical_cols:
                    unique_values = imputed_df[col].nunique()
                    print(f"    - {col}: {unique_values} categories")

        # 3. Persist feature names for downstream use
        self.original_features = numerical_cols + categorical_cols
        self.feature_names = self.original_features.copy()

        self.is_fitted = True

        if verbose:
            print("\n3. Feature engineering pipeline ready")
            print(f"  Total features: {len(self.feature_names)}")
            print("="*70)

        return self

    def transform(self, data, verbose=True):
        """
        Transform the provided dataset using the fitted preprocessing steps.

        Args:
            data: Dataset to transform.
            verbose: Whether to print progress information.

        Returns:
            DataFrame: Transformed feature matrix.
            Series: Target labels (if present in the input).
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform().")

        if verbose:
            print(f"\nTransforming data ({data.shape[0]} samples)...")

        # Extract target column if present
        y = None
        if self.config.TARGET_COLUMN in data.columns:
            y = data[self.config.TARGET_COLUMN].copy()
            # Remove target column from feature set
            data = data.drop(columns=[self.config.TARGET_COLUMN])

        transformed_data = {}

        # 1. Transform numerical features
        numerical_cols = [col for col in self.config.NUMERICAL_FEATURES if col in data.columns]

        if len(numerical_cols) > 0 and self.numerical_imputer is not None:
            # Impute missing values
            imputed = self.numerical_imputer.transform(data[numerical_cols])
            # Standardise
            scaled = self.scaler.transform(imputed)

            # Insert back into dictionary
            for i, col in enumerate(numerical_cols):
                transformed_data[col] = scaled[:, i]

        # 2. Transform categorical features
        categorical_cols = [col for col in self.config.CATEGORICAL_FEATURES if col in data.columns]

        if len(categorical_cols) > 0 and self.categorical_imputer is not None:
            # Impute missing values
            imputed = self.categorical_imputer.transform(data[categorical_cols])
            imputed_df = pd.DataFrame(imputed, columns=categorical_cols, index=data.index)

            # Label encode
            for col in categorical_cols:
                if col in self.label_encoders:
                    transformed_data[col] = self.label_encoders[col].transform(imputed_df[col].astype(str))

        # Construct the transformed frame
        X_transformed = pd.DataFrame(transformed_data, index=data.index)

        # Ensure consistent column ordering
        X_transformed = X_transformed[self.feature_names]

        if verbose:
            print(f"  ✓ Transformation complete: {X_transformed.shape}")

        return X_transformed, y

    def fit_transform(self, data, verbose=True):
        """
        Fit the preprocessing pipeline and immediately transform the dataset.

        Args:
            data: Training dataset.
            verbose: Whether to print progress information.

        Returns:
            DataFrame: Transformed features.
            Series: Target labels (if present).
        """
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def get_feature_names(self):
        """Return a copy of the feature name list."""
        return self.feature_names.copy()

    def get_feature_info(self):
        """Return metadata about the engineered features."""
        info = {
            'total_features': len(self.feature_names),
            'numerical_features': [col for col in self.config.NUMERICAL_FEATURES if col in self.feature_names],
            'categorical_features': [col for col in self.config.CATEGORICAL_FEATURES if col in self.feature_names],
            'feature_names': self.feature_names
        }

        # Include encoding information for categorical features
        if self.label_encoders:
            info['categorical_encoding'] = {}
            for col, le in self.label_encoders.items():
                info['categorical_encoding'][col] = {
                    'classes': le.classes_.tolist(),
                    'n_classes': len(le.classes_)
                }

        return info

    def save(self, save_dir=None):
        """
        Persist the fitted feature engineering pipeline to disk.

        Args:
            save_dir: Directory where the artifact should be stored. Defaults
                to ``self.config``'s resolved model save directory (so this
                respects any CLI/output-dir override) rather than a hardcoded
                path relative to the current working directory.
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before saving.")

        if save_dir is None:
            save_dir = self.config.get_absolute_paths()['model_save_dir']

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, 'feature_engineer.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({
                'numerical_imputer': self.numerical_imputer,
                'categorical_imputer': self.categorical_imputer,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'original_features': self.original_features,
                'config': self.config
            }, f)

        print(f"\n✓ Feature engineer saved to: {save_path}")

        return save_path

    @classmethod
    def load(cls, save_path='outputs/models/feature_engineer.pkl'):
        """
        Load a previously persisted feature engineering pipeline.

        Args:
            save_path: Path to the saved artifact.

        Returns:
            FeatureEngineer: The restored feature engineer instance.
        """
        with open(save_path, 'rb') as f:
            data = pickle.load(f)

        # Create a new instance
        engineer = cls(config=data['config'])

        # Restore state
        engineer.numerical_imputer = data['numerical_imputer']
        engineer.categorical_imputer = data['categorical_imputer']
        engineer.scaler = data['scaler']
        engineer.label_encoders = data['label_encoders']
        engineer.feature_names = data['feature_names']
        engineer.original_features = data['original_features']
        engineer.is_fitted = True

        print(f"\n✓ Feature engineer loaded from {save_path}")

        return engineer

    def print_summary(self):
        """Print a human-readable summary of the feature pipeline."""
        if not self.is_fitted:
            print("FeatureEngineer has not been fitted")
            return

        print("\n" + "="*70)
        print(" " * 23 + "Feature Engineering Summary")
        print("="*70)

        print("\n[Feature Statistics]")
        print(f"  Total features: {len(self.feature_names)}")

        numerical_cols = [col for col in self.config.NUMERICAL_FEATURES if col in self.feature_names]
        categorical_cols = [col for col in self.config.CATEGORICAL_FEATURES if col in self.feature_names]

        print(f"  Numerical features: {len(numerical_cols)}")
        for col in numerical_cols:
            print(f"    - {col}")

        print(f"  Categorical features: {len(categorical_cols)}")
        for col in categorical_cols:
            if col in self.label_encoders:
                n_classes = len(self.label_encoders[col].classes_)
                print(f"    - {col} ({n_classes} categories)")

        print("\n[Preprocessing]")
        print(f"  Numerical: {self.config.FILL_STRATEGY['numerical']} imputation + StandardScaler")
        print(f"  Categorical: missing values become an explicit \"nan\" category (upstream str-cast; "
              f"{self.config.FILL_STRATEGY['categorical']} imputer is a historical no-op) + LabelEncoder")

        print("="*70 + "\n")


def prepare_features(data, verbose=True, save=True, engineer=None, config=None):
    """
    Convenience helper that prepares features using the default configuration.

    Args:
        data: Raw dataset.
        verbose: Whether to print progress information.
        save: Whether to persist the fitted feature engineer.
        engineer: 可选，已拟合的FeatureEngineer实例。如果提供，则仅进行transform（用于验证集）
        config: Optional ``Config`` (or subclass) instance. Only used when
            ``engineer`` is None (i.e. when fitting a new engineer); ignored
            otherwise since an already-fitted engineer carries its own config.

    Returns:
        X: Transformed features.
        y: Target labels.
        engineer: The fitted feature engineer instance.
    """
    if engineer is not None:
        # 使用已有的engineer进行transform（用于验证集）
        X, y = engineer.transform(data, verbose=verbose)
        return X, y, engineer

    # 创建新的engineer并拟合
    engineer = FeatureEngineer(config=config)
    X, y = engineer.fit_transform(data, verbose=verbose)

    if verbose:
        engineer.print_summary()

    if save:
        engineer.save()

    return X, y, engineer
