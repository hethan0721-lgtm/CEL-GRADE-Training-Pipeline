#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definition module containing XGBoost and alternative classifiers.
"""

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os

from .config import Config


class SurgeryClassifier:
    """Unified classifier interface supporting XGBoost, Random Forest, and Logistic Regression."""

    def __init__(self, model_type='xgboost', config=None):
        """
        Initialize the classifier with the specified model type.

        Args:
            model_type: One of {'xgboost', 'random_forest', 'logistic'}.
            config: Optional configuration object.
        """
        self.config = config if config is not None else Config()
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance_ = None

        # Instantiate the underlying estimator
        self._create_model()

    def _create_model(self):
        """Instantiate the underlying estimator based on the configured type."""
        if self.model_type == 'xgboost':
            self.model = self._create_xgboost()
        elif self.model_type == 'random_forest':
            self.model = self._create_random_forest()
        elif self.model_type == 'logistic':
            self.model = self._create_logistic()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_xgboost(self):
        """Build an XGBClassifier with the configured parameters."""
        params = self.config.XGBOOST_PARAMS.copy()
        return xgb.XGBClassifier(**params)

    def _create_random_forest(self):
        """Build a Random Forest classifier with sensible defaults."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )

    def _create_logistic(self):
        """Build a Logistic Regression classifier with sensible defaults."""
        return LogisticRegression(
            max_iter=1000,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the classifier on the provided dataset.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            verbose: Whether to print progress information.

        Returns:
            self
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 20 + f"Training {self.model_type.upper()} Model")
            print(f"{'='*70}")
            print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            if X_val is not None:
                print(f"Validation set: {X_val.shape[0]} samples")

        # Save feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)

        # Special handling for XGBoost
        if self.model_type == 'xgboost':
            # Convert DataFrame to numpy array (required by XGBoost API)
            if hasattr(X_train, 'values'):
                X_train_array = X_train.values
            else:
                X_train_array = X_train

            if X_val is not None and y_val is not None:
                # Convert validation set
                if hasattr(X_val, 'values'):
                    X_val_array = X_val.values
                else:
                    X_val_array = X_val

                # Use validation set and early stopping
                eval_set = [(X_train_array, y_train), (X_val_array, y_val)]

                self.model.fit(
                    X_train_array, y_train,
                    eval_set=eval_set,
                    verbose=verbose
                )
            else:
                self.model.fit(X_train_array, y_train, verbose=verbose)
        else:
            # Other models train directly
            self.model.fit(X_train, y_train)

        self.is_fitted = True

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_

        if verbose:
            print(f"{'='*70}")
            print("✓ Model training complete")
            print(f"{'='*70}\n")

        return self

    def predict(self, X):
        """
        Predict class labels for the provided features.

        Args:
            X: Feature matrix.

        Returns:
            array: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict().")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the provided features.

        Args:
            X: Feature matrix.

        Returns:
            array: Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict_proba().")

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n=None):
        """
        Retrieve feature importance scores.

        Args:
            top_n: Optional limit for the number of top features; None returns all.

        Returns:
            dict: Mapping of feature names to importance scores (sorted descending).
        """
        if self.feature_importance_ is None:
            return None

        if self.feature_names is None:
            return None

        importance_dict = dict(zip(self.feature_names, self.feature_importance_))

        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        if top_n is not None:
            sorted_importance = sorted_importance[:top_n]

        return dict(sorted_importance)

    def save(self, save_path):
        """
        Persist the fitted model to disk.

        Args:
            save_path: Destination path for the serialized model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance_,
                'config': self.config
            }, f)

        print(f"\n✓ Model saved to: {save_path}")

        return save_path

    @classmethod
    def load(cls, save_path):
        """
        Load a previously saved model from disk.

        Args:
            save_path: Path to the serialized model.

        Returns:
            SurgeryClassifier: The restored classifier instance.
        """
        with open(save_path, 'rb') as f:
            data = pickle.load(f)

        # Create new instance
        classifier = cls(model_type=data['model_type'], config=data['config'])

        # Restore state
        classifier.model = data['model']
        classifier.feature_names = data['feature_names']
        classifier.feature_importance_ = data['feature_importance']
        classifier.is_fitted = True

        print(f"\n✓ Model loaded from {save_path}")

        return classifier

    def get_params(self):
        """Return the underlying estimator's parameters."""
        if self.model is None:
            return None

        return self.model.get_params()

    def set_params(self, **params):
        """Set parameters on the underlying estimator."""
        if self.model is None:
            raise ValueError("No model has been instantiated.")

        self.model.set_params(**params)
        return self

    def print_info(self):
        """Print a human-readable summary of the model."""
        print(f"\n{'='*70}")
        print(f" " * 25 + "Model Information")
        print(f"{'='*70}")

        print("\n[Basic Information]")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Training status: {'Fitted' if self.is_fitted else 'Not fitted'}")

        if self.feature_names:
            print(f"  Number of features: {len(self.feature_names)}")

        if self.model_type == 'xgboost' and self.is_fitted:
            print("\n[XGBoost Parameters]")
            params = self.model.get_params()
            important_params = [
                'max_depth', 'learning_rate', 'n_estimators',
                'min_child_weight', 'gamma', 'subsample',
                'colsample_bytree', 'reg_alpha', 'reg_lambda'
            ]
            for param in important_params:
                if param in params:
                    print(f"  {param}: {params[param]}")

        if self.feature_importance_ is not None:
            print("\n[Feature Importance] (Top 10)")
            top_features = self.get_feature_importance(top_n=10)
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                print(f"  {i}. {feature}: {importance:.4f}")

        print(f"{'='*70}\n")


def create_model(model_type='xgboost', config=None):
    """
    Factory function for instantiating a classifier.

    Args:
        model_type: Type of classifier to create.
        config: Optional configuration object.

    Returns:
        SurgeryClassifier: A fresh classifier instance.
    """
    return SurgeryClassifier(model_type=model_type, config=config)
