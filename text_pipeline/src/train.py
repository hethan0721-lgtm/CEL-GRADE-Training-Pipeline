#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module responsible for model fitting, hyper-parameter tuning,
cross-validation, and related utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import time

from .config import Config
from .models import SurgeryClassifier
from .feature_engineering import FeatureEngineer


class ModelTrainer:
    """Coordinates model training, validation, and persistence."""

    def __init__(self, config=None):
        """
        Initialize the trainer with an optional configuration.

        Args:
            config: Optional ``Config`` instance.
        """
        self.config = config if config is not None else Config()
        self.model = None
        self.best_model = None
        self.training_history = {}
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        # 储存外部提供的验证集
        self.external_val_data = None
        self.external_val_labels = None

    def split_data(self, X, y, verbose=True):
        """
        Split the dataset into train, validation, and test partitions.

        Args:
            X: Feature matrix.
            y: Target labels.
            verbose: Whether to print progress information.

        Returns:
            Tuple containing train/validation/test features and labels.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 25 + "Splitting Dataset")
            print(f"{'='*70}")

        # Create indices array to track the splits
        indices = np.arange(len(X))

        # Split off the test set first
        X_temp, X_test, y_temp, y_test, indices_temp, indices_test = train_test_split(
            X, y, indices,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        # From the remainder, optionally carve out the validation set
        if self.config.VAL_SIZE > 0:
            val_size_adjusted = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
            X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
                X_temp, y_temp, indices_temp,
                test_size=val_size_adjusted,
                random_state=self.config.RANDOM_STATE,
                stratify=y_temp
            )
        else:
            # No validation split; all remaining data is used for training
            X_train, y_train, indices_train = X_temp, y_temp, indices_temp
            X_val, y_val, indices_val = None, None, None

        # Store indices for later use
        self.train_indices = indices_train
        self.val_indices = indices_val
        self.test_indices = indices_test

        if verbose:
            print("\nSplit complete:")
            print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
            if X_val is not None:
                print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
            print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

            print("\nClass distribution per split:")
            splits = [('Train', y_train), ('Test', y_test)]
            if y_val is not None:
                splits.insert(1, ('Validation', y_val))
            for name, labels in splits:
                counts = pd.Series(labels).value_counts().sort_index()
                print(f"  {name}:")
                for label, count in counts.items():
                    label_name = self.config.LABEL_NAMES.get(label, label)
                    print(f"    {label_name}: {count} ({count/len(labels)*100:.1f}%)")

            print(f"{'='*70}\n")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def set_external_val_data(self, X_val, y_val):
        """
        设置外部提供的验证集（用于验证和评估）。

        Args:
            X_val: 验证集特征
            y_val: 验证集标签
        """
        self.external_val_data = X_val
        self.external_val_labels = y_val

    def get_val_data(self):
        """获取验证集数据"""
        return self.external_val_data, self.external_val_labels

    def apply_smote(self, X_train, y_train, verbose=True):
        """
        Apply SMOTE oversampling to the training split if enabled.

        Args:
            X_train: Training features.
            y_train: Training labels.
            verbose: Whether to print progress information.

        Returns:
            Tuple ``(X_resampled, y_resampled)``.
        """
        if not self.config.SMOTE_CONFIG['enabled']:
            if verbose:
                print("\n⚠ SMOTE disabled; skipping oversampling")
            return X_train, y_train

        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 23 + "Applying SMOTE Oversampling")
            print(f"{'='*70}")
            print("\nOriginal training distribution:")
            counts = pd.Series(y_train).value_counts().sort_index()
            for label, count in counts.items():
                label_name = self.config.LABEL_NAMES.get(label, label)
                print(f"  {label_name}: {count}")

        try:
            # Instantiate SMOTE
            smote = SMOTE(
                sampling_strategy=self.config.SMOTE_CONFIG['sampling_strategy'],
                k_neighbors=self.config.SMOTE_CONFIG['k_neighbors'],
                random_state=self.config.SMOTE_CONFIG['random_state']
            )

            # Apply SMOTE (convert DataFrame to numpy array if needed)
            # Preserve column order for reconstruction
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
                X_array = X_train.values
            else:
                feature_names = None
                X_array = X_train

            X_resampled, y_resampled = smote.fit_resample(X_array, y_train)

            # Recreate DataFrame if necessary
            if feature_names is not None:
                X_resampled = pd.DataFrame(X_resampled, columns=feature_names)

            if verbose:
                print("\nTraining distribution after SMOTE:")
                counts = pd.Series(y_resampled).value_counts().sort_index()
                for label, count in counts.items():
                    label_name = self.config.LABEL_NAMES.get(label, label)
                    print(f"  {label_name}: {count}")

                print(f"\n  Original samples: {len(y_train)}")
                print(f"  After SMOTE:     {len(y_resampled)}")
                print(f"  Added samples:   {len(y_resampled) - len(y_train)}")
                print(f"{'='*70}\n")

            return X_resampled, y_resampled

        except Exception as e:
            print(f"\n⚠ Failed to apply SMOTE: {e}")
            print("  Continuing with the original training set")
            return X_train, y_train

    def calculate_class_weights(self, y_train, verbose=True):
        """
        Compute class weights for imbalanced datasets.

        Args:
            y_train: Training labels.
            verbose: Whether to print the computed weights.

        Returns:
            dict: Mapping of label -> weight.
        """
        if not self.config.USE_CLASS_WEIGHT:
            return None

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )

        class_weights = dict(zip(classes, weights))

        if verbose:
            print("\nComputed class weights:")
            for label, weight in class_weights.items():
                label_name = self.config.LABEL_NAMES.get(label, label)
                print(f"  {label_name}: {weight:.4f}")

        return class_weights

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                    apply_smote=True, use_class_weight=True, verbose=True):
        """
        Train the underlying classifier with optional SMOTE and class weighting.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            apply_smote: Whether to oversample using SMOTE.
            use_class_weight: Whether to apply class weighting.
            verbose: Whether to print progress information.

        Returns:
            Trained ``SurgeryClassifier`` instance.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 28 + "Training Started")
            print(f"{'='*70}")

        start_time = time.time()

        # 1. Optionally apply SMOTE
        if apply_smote:
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train, verbose=verbose)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # 2. Optionally compute class weights
        class_weights = None
        if use_class_weight:
            class_weights = self.calculate_class_weights(y_train_resampled, verbose=verbose)

        # 3. Instantiate model
        self.model = SurgeryClassifier(model_type='xgboost', config=self.config)

        # 4. Configure class weighting (if available)
        if class_weights is not None and hasattr(self.model.model, 'set_params'):
            # Compute scale_pos_weight for XGBoost
            if len(class_weights) == 2:
                scale_pos_weight = class_weights[0] / class_weights[1]
                self.model.model.set_params(scale_pos_weight=scale_pos_weight)
                if verbose:
                    print(f"  scale_pos_weight set to: {scale_pos_weight:.4f}\n")

        # 5. Fit model
        self.model.fit(
            X_train_resampled, y_train_resampled,
            X_val, y_val,
            verbose=verbose
        )

        # 6. Record training metadata
        training_time = time.time() - start_time
        self.training_history = {
            'training_time': training_time,
            'training_samples': len(y_train_resampled),
            'original_samples': len(y_train),
            'smote_applied': apply_smote,
            'class_weight_used': use_class_weight
        }

        if verbose:
            print("\nTraining completed!")
            print(f"  Training time: {training_time:.2f} seconds")
            print(f"{'='*70}\n")

        return self.model

    def hyperparameter_tuning(self, X_train, y_train, verbose=True):
        """
        Perform hyper-parameter tuning using RandomizedSearchCV.

        NOT part of the canonical training path. ``main.py`` / the CLI always
        calls ``train_pipeline(..., hyperparameter_tuning=False)``, so this
        method is never invoked by the public pipeline; the canonical model
        is always trained with the single fixed ``Config.XGBOOST_PARAMS``
        set. ``Config.ENABLE_HYPERPARAMETER_TUNING`` defaults to ``False``
        for the same reason. This method is retained only as a manual/
        offline experimentation helper -- calling it directly (with
        ``ENABLE_HYPERPARAMETER_TUNING`` explicitly set ``True``) does not
        affect, and must not be treated as describing, the canonical model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            verbose: Whether to print progress information.

        Returns:
            Tuple ``(best_model, best_params)`` with the tuned estimator and parameters.
        """
        if not self.config.ENABLE_HYPERPARAMETER_TUNING:
            if verbose:
                print("\n⚠ Hyper-parameter tuning is disabled")
            return None, None

        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 22 + "Hyper-parameter Tuning")
            print(f"{'='*70}")
            print(f"\nMethod: {self.config.TUNING_METHOD}")
            print(f"Iterations: {self.config.RANDOM_SEARCH_PARAMS['n_iter']}")
            print(f"CV folds: {self.config.RANDOM_SEARCH_PARAMS['cv']}")
            print(f"Scoring: {self.config.RANDOM_SEARCH_PARAMS['scoring']}")
            print("\nSearching for optimal parameters...")

        # Base model
        base_model = SurgeryClassifier(model_type='xgboost', config=self.config)

        # Configure randomized search
        search = RandomizedSearchCV(
            estimator=base_model.model,
            param_distributions=self.config.PARAM_GRID,
            n_iter=self.config.RANDOM_SEARCH_PARAMS['n_iter'],
            cv=self.config.RANDOM_SEARCH_PARAMS['cv'],
            scoring=self.config.RANDOM_SEARCH_PARAMS['scoring'],
            n_jobs=self.config.RANDOM_SEARCH_PARAMS['n_jobs'],
            verbose=self.config.RANDOM_SEARCH_PARAMS['verbose'],
            random_state=self.config.RANDOM_SEARCH_PARAMS['random_state']
        )

        # Execute search
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time

        if verbose:
            print("\n✓ Search complete!")
            print(f"  Elapsed time: {search_time:.2f} seconds")
            print(f"  Best score:  {search.best_score_:.4f}")
            print("\nBest parameters:")
            for param, value in search.best_params_.items():
                print(f"  {param}: {value}")
            print(f"{'='*70}\n")

        # Build the best model instance
        best_model = SurgeryClassifier(model_type='xgboost', config=self.config)
        best_model.model = search.best_estimator_
        best_model.is_fitted = True
        best_model.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        if hasattr(best_model.model, 'feature_importances_'):
            best_model.feature_importance_ = best_model.model.feature_importances_

        self.best_model = best_model

        return best_model, search.best_params_

    def cross_validate_pipeline(self, raw_data, verbose=True):
        """
        Fold-wise cross-validation with NO preprocessing/SMOTE leakage.

        Unlike ``cross_validate()`` (kept below only as a fallback for the
        unreachable single-file / USE_SEPARATE_VAL_FILE=False code path), this
        method operates on ``raw_data`` — the cleaned 80% training partition
        BEFORE feature engineering. For each of the 5 stratified folds it:
          1. Fits a brand-new ``FeatureEngineer`` (median/mode imputation,
             StandardScaler, LabelEncoder) on that fold's training rows only.
          2. Transforms that fold's held-out validation rows with the SAME
             fitted engineer (transform-only, no fit).
          3. Applies SMOTE to that fold's training partition only (same
             SMOTE_CONFIG as the final model; validation rows never resampled).
          4. Computes class weights / scale_pos_weight the same way
             ``train_model()`` does, on the SMOTE-resampled fold-training data.
          5. Fits a fresh XGBClassifier (same Config.XGBOOST_PARAMS) and scores
             it on the fold's held-out validation partition.

        This guarantees each fold's held-out rows never influence that fold's
        imputer/scaler/encoder/SMOTE/model fit — the fold-level analogue of
        the isolation already enforced for the internal 20% test set and the
        external validation set.

        Args:
            raw_data: Cleaned DataFrame (target column included, NOT yet
                feature-engineered) representing the 80% training partition.
            verbose: Whether to print per-fold progress.

        Returns:
            dict: Cross-validation scores keyed by metric name, in the same
                shape as ``cross_validate()`` (``{'scores', 'mean', 'std'}``
                per metric), for drop-in compatibility with existing callers.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 20 + "Cross-validation (fold-wise, leakage-free)")
            print(f"{'='*70}")
            print(f"\nFolds: {self.config.CV_FOLDS}")
            print(f"Stratified: {self.config.STRATIFIED}")

        y_full = raw_data[self.config.TARGET_COLUMN].reset_index(drop=True)
        raw_data = raw_data.reset_index(drop=True)

        if self.config.STRATIFIED:
            splitter = StratifiedKFold(
                n_splits=self.config.CV_FOLDS,
                shuffle=True,
                random_state=self.config.RANDOM_STATE
            )
            split_iter = splitter.split(raw_data, y_full)
        else:
            splitter = KFold(
                n_splits=self.config.CV_FOLDS,
                shuffle=True,
                random_state=self.config.RANDOM_STATE
            )
            split_iter = splitter.split(raw_data)

        metric_scorers = {
            'accuracy': lambda y_true, y_pred, y_proba: accuracy_score(y_true, y_pred),
            'precision': lambda y_true, y_pred, y_proba: precision_score(y_true, y_pred, average='binary'),
            'recall': lambda y_true, y_pred, y_proba: recall_score(y_true, y_pred, average='binary'),
            'f1': lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred, average='binary'),
            'roc_auc': lambda y_true, y_pred, y_proba: roc_auc_score(y_true, y_proba),
        }

        fold_scores = {metric: [] for metric in self.config.EVALUATION_METRICS}

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter, 1):
            fold_train_raw = raw_data.iloc[train_idx].reset_index(drop=True)
            fold_val_raw = raw_data.iloc[val_idx].reset_index(drop=True)

            # 1-2. Fresh preprocessing, fit on this fold's training rows only
            fold_engineer = FeatureEngineer(config=self.config)
            X_fold_train, y_fold_train = fold_engineer.fit_transform(fold_train_raw, verbose=False)
            X_fold_val, y_fold_val = fold_engineer.transform(fold_val_raw, verbose=False)

            # 3. SMOTE on this fold's training partition only (same config as final model)
            if self.config.SMOTE_CONFIG['enabled']:
                X_fold_train_res, y_fold_train_res = self.apply_smote(X_fold_train, y_fold_train, verbose=False)
            else:
                X_fold_train_res, y_fold_train_res = X_fold_train, y_fold_train

            # 4. Class weighting, same logic as train_model()
            fold_class_weights = None
            if self.config.USE_CLASS_WEIGHT:
                fold_class_weights = self.calculate_class_weights(y_fold_train_res, verbose=False)

            fold_model = SurgeryClassifier(model_type='xgboost', config=self.config)
            if fold_class_weights is not None and len(fold_class_weights) == 2:
                scale_pos_weight = fold_class_weights[0] / fold_class_weights[1]
                fold_model.model.set_params(scale_pos_weight=scale_pos_weight)

            # 5. Fit fresh model, score on this fold's held-out (transform-only) partition
            fold_model.fit(X_fold_train_res, y_fold_train_res, verbose=False)

            y_val_pred = fold_model.predict(X_fold_val)
            y_val_proba = fold_model.predict_proba(X_fold_val)[:, 1]

            for metric in self.config.EVALUATION_METRICS:
                score = metric_scorers[metric](y_fold_val, y_val_pred, y_val_proba)
                fold_scores[metric].append(score)

            if verbose:
                summary = "  ".join(f"{m}={fold_scores[m][-1]:.4f}" for m in self.config.EVALUATION_METRICS)
                print(f"  Fold {fold_idx}/{self.config.CV_FOLDS}: {summary}")

        results = {}
        for metric, scores in fold_scores.items():
            arr = np.array(scores)
            results[metric] = {'scores': arr, 'mean': arr.mean(), 'std': arr.std()}
            if verbose:
                print(f"\n{metric}: Mean = {arr.mean():.4f} (+/- {arr.std():.4f})")

        if verbose:
            print(f"\n{'='*70}\n")

        return results

    def cross_validate(self, X, y, model=None, verbose=True):
        """
        [Legacy — retained only as a fallback for the unreachable
        USE_SEPARATE_VAL_FILE=False / single-file code path, which is never
        exercised by ``main.py`` in production. Do NOT use this for the
        production 80/20-split path: it runs ``cross_val_score`` on data that
        was already imputed/scaled OUTSIDE this function and does not include
        SMOTE inside the fold loop. Use ``cross_validate_pipeline()`` instead,
        which is what the production path in ``train_pipeline()`` now calls.]

        Run cross-validation for the given model or the currently trained one.

        Args:
            X: Feature matrix.
            y: Target labels.
            model: Optional estimator; defaults to the internally trained model.
            verbose: Whether to print progress information.

        Returns:
            dict: Cross-validation scores keyed by metric name.
        """
        from sklearn.model_selection import cross_val_score

        if model is None:
            if self.model is None:
                raise ValueError("Train a model first or provide one explicitly.")
            model = self.model.model

        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 26 + "Cross-validation")
            print(f"{'='*70}")
            print(f"\nFolds: {self.config.CV_FOLDS}")
            print(f"Stratified: {self.config.STRATIFIED}")

        # Prepare splitter
        if self.config.STRATIFIED:
            cv = StratifiedKFold(
                n_splits=self.config.CV_FOLDS,
                shuffle=True,
                random_state=self.config.RANDOM_STATE
            )
        else:
            cv = self.config.CV_FOLDS

        # Evaluate each metric
        results = {}
        for metric in self.config.EVALUATION_METRICS:
            if verbose:
                print(f"\nScoring metric: {metric}")

            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=metric,
                n_jobs=-1
            )

            results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }

            if verbose:
                print(f"  Fold scores: {scores}")
                print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

        if verbose:
            print(f"\n{'='*70}\n")

        return results

    def save_model(self, save_dir=None):
        """
        Persist the trained model and training history to disk.

        Args:
            save_dir: Optional destination directory.
        """
        if self.model is None:
            raise ValueError("No trained model is available for saving.")

        if save_dir is None:
            paths = self.config.get_absolute_paths()
            save_dir = paths['model_save_dir']

        os.makedirs(save_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(save_dir, self.config.BEST_MODEL_NAME)
        self.model.save(model_path)

        # Save training history
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)

        print(f"✓ Training history saved to: {history_path}")

        return model_path


def train_pipeline(X, y, apply_smote=True, hyperparameter_tuning=False,
                   cross_validation=False, verbose=True,
                   X_val_external=None, y_val_external=None,
                   cv_raw_data=None, config=None):
    """
    End-to-end training pipeline with optional SMOTE, tuning, and CV.

    Args:
        X: Feature matrix (训练集，已完成 feature engineering).
        y: Target labels (训练集).
        apply_smote: Whether to apply SMOTE.
        hyperparameter_tuning: Whether to perform hyper-parameter search.
        cross_validation: Whether to run cross-validation diagnostics.
        verbose: Whether to print progress information.
        X_val_external: 外部验证集特征（可选，用于验证和评估）
        y_val_external: 外部验证集标签（可选）
        cv_raw_data: 可选，特征工程之前的原始（已清洗）80% 训练集 DataFrame
            （含 target 列）。提供时，cross-validation 使用
            ``ModelTrainer.cross_validate_pipeline()``，在每个 fold 内独立
            重新 fit 预处理与 SMOTE，避免 leakage。若为 None，则退回旧的
            ``cross_validate()``（仅用于没有独立验证集的历史/未使用路径）。
        config: Optional ``Config`` (or subclass) instance passed through to
            the ``ModelTrainer`` (and, transitively, the model/SMOTE/CV
            parameters it uses). Defaults to ``Config()`` when omitted.

    Returns:
        model: Trained classifier.
        trainer: ``ModelTrainer`` instance containing artifacts.
    """
    # Create trainer
    trainer = ModelTrainer(config=config)

    # 检查是否使用外部验证集
    use_external_val = X_val_external is not None and y_val_external is not None

    if use_external_val:
        # 使用外部验证集，训练集不做划分
        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 20 + "Using External Validation Set")
            print(f"{'='*70}")
            print(f"\n  Train set: {len(y)} samples")
            print(f"  Validation/Test set: {len(y_val_external)} samples")

            print("\n  Train set distribution:")
            train_counts = pd.Series(y).value_counts().sort_index()
            for label, count in train_counts.items():
                label_name = trainer.config.LABEL_NAMES.get(label, label)
                print(f"    {label_name}: {count} ({count/len(y)*100:.1f}%)")

            print("\n  Validation set distribution:")
            val_counts = pd.Series(y_val_external).value_counts().sort_index()
            for label, count in val_counts.items():
                label_name = trainer.config.LABEL_NAMES.get(label, label)
                print(f"    {label_name}: {count} ({count/len(y_val_external)*100:.1f}%)")
            print(f"{'='*70}\n")

        # 储存外部验证集
        trainer.set_external_val_data(X_val_external, y_val_external)

        X_train = X
        y_train = y
        X_val = X_val_external
        y_val = y_val_external
    else:
        # 使用原来的划分逻辑
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y, verbose=verbose)

    # Optional hyper-parameter tuning
    if hyperparameter_tuning:
        # Tune on train data
        best_model, best_params = trainer.hyperparameter_tuning(X_train, y_train, verbose=verbose)
        trainer.model = best_model
    else:
        # Standard training path
        model = trainer.train_model(
            X_train, y_train,
            X_val, y_val,
            apply_smote=apply_smote,
            verbose=verbose
        )

    # Optional cross-validation
    if cross_validation:
        if cv_raw_data is not None:
            cv_results = trainer.cross_validate_pipeline(cv_raw_data, verbose=verbose)
        else:
            # Fallback for the unreachable single-file path only (see docstring
            # on ModelTrainer.cross_validate) — not used by the production
            # USE_SEPARATE_VAL_FILE=True flow, which always passes cv_raw_data.
            cv_results = trainer.cross_validate(X_train, y_train, verbose=verbose)
        trainer.training_history['cv_results'] = cv_results

    # Persist artifacts
    trainer.save_model()

    return trainer.model, trainer
