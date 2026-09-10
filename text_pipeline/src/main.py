#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M0_CLIN text/tabular pipeline entry point.

Runs the XGBoost binary classifier (surgery-need prediction) end to end:
data loading -> split -> preprocessing -> training -> prediction -> evaluation
-> artifact saving. See ``text_pipeline/README.md`` for the full description
of the pipeline, feature set, and leakage-safety guarantees.

This module is CLI-driven (see ``build_arg_parser`` below) and contains no
hardcoded personal/absolute paths; all data and output locations are either
CLI arguments or paths relative to the repository, resolved with
``pathlib.Path`` so behaviour is identical on Windows and Linux.
"""

import argparse
import os
import warnings
from pathlib import Path

from .config import Config
from .data_loader import DataLoader
from .feature_engineering import prepare_features, FeatureEngineer
from .train import train_pipeline
from .evaluate import ModelEvaluator
from .models import SurgeryClassifier

warnings.filterwarnings("ignore")

# text_pipeline/src/main.py -> parents[0]=src, [1]=text_pipeline, [2]=repo root
_THIS_FILE = Path(__file__).resolve()
TEXT_PIPELINE_DIR = _THIS_FILE.parents[1]
REPO_ROOT = _THIS_FILE.parents[2]

DEFAULT_TRAIN_DATA = TEXT_PIPELINE_DIR.parent / 'examples' / 'synthetic_tabular_data' / 'synthetic_train.csv'
DEFAULT_EXTERNAL_DATA = TEXT_PIPELINE_DIR.parent / 'examples' / 'synthetic_tabular_data' / 'synthetic_external.csv'
DEFAULT_OUTPUT_DIR = TEXT_PIPELINE_DIR / 'outputs'


class RunConfig(Config):
    """
    CLI-overridable ``Config`` subclass, defined at module scope (rather than
    inside a function) so instances remain picklable -- the trained model and
    FeatureEngineer artifacts each embed a reference to the config they were
    built with, and ``pickle`` cannot resolve classes that only exist as
    local objects inside a function's closure.

    Only paths and the random seed are ever overridden (see
    ``build_run_config``); feature columns, target column, label mapping,
    imputation/scaling/encoding strategy, SMOTE settings, XGBoost
    hyperparameters (other than ``random_state``), the train/test split
    ratio, and evaluation metrics are intentionally left untouched so the
    scientific behaviour of the pipeline never changes based on CLI input.
    """
    pass


def build_run_config(input_data=None, external_data=None, output_dir=None, seed=None):
    """Apply CLI-supplied overrides to ``RunConfig`` and return an instance."""
    seed = Config.RANDOM_STATE if seed is None else seed
    output_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    RunConfig.RANDOM_STATE = seed
    RunConfig.XGBOOST_PARAMS = {**Config.XGBOOST_PARAMS, 'random_state': seed}
    RunConfig.SMOTE_CONFIG = {**Config.SMOTE_CONFIG, 'random_state': seed}
    RunConfig.RANDOM_SEARCH_PARAMS = {**Config.RANDOM_SEARCH_PARAMS, 'random_state': seed}
    RunConfig.MODEL_SAVE_DIR = str(output_dir / 'models')
    RunConfig.RESULTS_SAVE_DIR = str(output_dir / 'eval_external')

    RunConfig.TRAIN_DATA_FILE = str(Path(input_data).resolve()) if input_data else Config.TRAIN_DATA_FILE
    RunConfig.VAL_DATA_FILE = str(Path(external_data).resolve()) if external_data else Config.VAL_DATA_FILE

    return RunConfig(), output_dir


def _print_header(title):
    print("\n" + "=" * 70)
    print(" " * max(0, (70 - len(title)) // 2) + title)
    print("=" * 70)


def run_train(config, output_dir, verbose=True):
    """
    Leakage-safe training run.

    Order (unchanged from the original pipeline):
      1. Load + clean the training file and the external validation file.
      2. Split the cleaned training data 80/20 BEFORE any preprocessing is fit.
      3. Fit the FeatureEngineer (imputers, scaler, label encoders) on the 80%
         partition only; transform-only the 20% internal test set and the
         external validation set.
      4. Train XGBoost on the 80% partition (SMOTE + class weighting fit on
         that partition only); run leakage-free fold-wise cross-validation
         for diagnostics.
      5. Evaluate on the internal 20% test set and on the external validation
         set, saving metrics/plots. Save the model and FeatureEngineer.
    """
    _print_header("M0_CLIN XGBoost Text Pipeline - Training")

    config.validate_config()

    # ---- Step 1: Load and clean data ----
    _print_header("[Step 1/4] Data Loading")

    loader = DataLoader(config=config)
    train_data, val_data = loader.load_train_val_data(verbose=verbose)
    train_data, val_data = loader.clean_train_val_data(verbose=verbose)

    # ---- Step 2: Feature engineering ----
    _print_header("[Step 2/4] Feature Engineering")

    from sklearn.model_selection import train_test_split

    print("\nTraining data class distribution (raw, before split):")
    train_target_counts = train_data[config.TARGET_COLUMN].value_counts()
    n0 = train_target_counts.get(0, 0)
    n1 = train_target_counts.get(1, 0)
    print(f"  {config.LABEL_NAMES[0]}: {n0} samples")
    print(f"  {config.LABEL_NAMES[1]}: {n1} samples")
    if n1 > 0:
        print(f"  Imbalance ratio: 1:{n0 / n1:.1f}")

    train_data_80, test_data_20 = train_test_split(
        train_data,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_data[config.TARGET_COLUMN]
    )
    train_data_80 = train_data_80.reset_index(drop=True)
    test_data_20 = test_data_20.reset_index(drop=True)

    print(f"\n  Internal 80:20 split of training data (raw, pre-feature-engineering):")
    print(f"    Train (80%): {len(train_data_80)} samples")
    print(f"    Test  (20%): {len(test_data_20)} samples")

    # Fit ONLY on the 80% training partition; transform-only elsewhere.
    X_train_80, y_train_80, engineer = prepare_features(
        train_data_80, verbose=verbose, save=True, config=config
    )
    X_test_20, y_test_20, _ = prepare_features(
        test_data_20, verbose=False, save=False, engineer=engineer
    )
    X_val, y_val, _ = prepare_features(
        val_data, verbose=False, save=False, engineer=engineer
    )
    print(f"    External val: {len(y_val)} samples")

    # ---- Step 3: Train model ----
    _print_header("[Step 3/4] Model Training")

    model, trainer = train_pipeline(
        X_train_80, y_train_80,
        apply_smote=True,
        hyperparameter_tuning=False,
        cross_validation=True,
        verbose=verbose,
        X_val_external=X_test_20,
        y_val_external=y_test_20,
        cv_raw_data=train_data_80,
        config=config
    )

    # ---- Step 4: Evaluate model ----
    _print_header("[Step 4/4] Model Evaluation")

    internal_eval_dir = output_dir / 'eval_internal'
    internal_eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 70)
    print(" " * 22 + "Internal Test Set (20%) Evaluation")
    print("-" * 70)

    test_evaluator = ModelEvaluator(config=config)
    test_results = test_evaluator.evaluate(model, X_test_20, y_test_20, verbose=True)
    test_evaluator.generate_all_plots(model, save_dir=str(internal_eval_dir), show=False)
    test_evaluator.save_results(save_dir=str(internal_eval_dir))
    print(f"\n Internal test set evaluation saved to: {internal_eval_dir}")

    external_eval_dir = output_dir / 'eval_external'
    external_eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 70)
    print(" " * 22 + "External Validation Set Evaluation")
    print("-" * 70)

    ext_evaluator = ModelEvaluator(config=config)
    ext_results = ext_evaluator.evaluate(model, X_val, y_val, verbose=True)
    ext_evaluator.generate_all_plots(model, save_dir=str(external_eval_dir), show=False)
    ext_evaluator.save_results(save_dir=str(external_eval_dir))
    print(f"\n External validation set evaluation saved to: {external_eval_dir}")

    # ---- Feature importance overview (diagnostic printout only; no file output) ----
    _print_header("Feature Importance Analysis")

    print("\n[Expert-defined Feature Weights] (ordered from most to least important)")
    feature_order = config.FEATURE_WEIGHT_INFO['order']
    for i, feature in enumerate(feature_order, 1):
        print(f"  {i}. {feature}")

    print("\n[Model-learned Feature Importance]")
    feature_importance = model.get_feature_importance(top_n=10)
    if feature_importance:
        for i, (feature, importance) in enumerate(feature_importance.items(), 1):
            expert_rank = feature_order.index(feature) + 1 if feature in feature_order else '?'
            print(f"  {i:2d}. {feature:30s} {importance:.4f}  (expert rank: {expert_rank})")

    _print_header("Training Completed")
    print(f"\n Internal test set (20%) performance ({internal_eval_dir}):")
    for m in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        print(f"  {m}: {test_results[m]:.4f}")
    print(f"\n External validation set performance ({external_eval_dir}):")
    for m in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        print(f"  {m}: {ext_results[m]:.4f}")

    return model, trainer, test_results, ext_results


def run_internal_eval(config, output_dir, verbose=True):
    """
    Re-evaluate an already-trained model on the internal 20% test partition,
    without retraining or refitting any preprocessing step.

    Reconstructs the same 80/20 split used during ``train`` (same input file,
    same ``TEST_SIZE``, same ``--seed``) and scores the held-out 20% using the
    FeatureEngineer and model artifacts saved under ``<output_dir>/models``.
    The seed MUST match the one used for the original training run, or the
    reconstructed partition will not be the model's true held-out set.
    """
    _print_header("M0_CLIN XGBoost Text Pipeline - Internal Evaluation")

    models_dir = output_dir / 'models'
    engineer_path = models_dir / 'feature_engineer.pkl'
    model_path = models_dir / config.BEST_MODEL_NAME
    if not engineer_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No trained artifacts found under {models_dir}. Run --mode train first."
        )

    from sklearn.model_selection import train_test_split

    loader = DataLoader(config=config)
    raw_df = loader._load_single_file(config.get_absolute_paths()['train_data_file'], file_type='eval', verbose=verbose)
    if raw_df is None:
        raise ValueError(f"Failed to load input data file: {config.TRAIN_DATA_FILE}")
    clean_df, _ = loader._clean_dataframe(raw_df, preserve_name='internal-eval', verbose=verbose)

    _, test_data_20 = train_test_split(
        clean_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=clean_df[config.TARGET_COLUMN]
    )
    test_data_20 = test_data_20.reset_index(drop=True)
    print(f"\n  Reconstructed internal test partition: {len(test_data_20)} samples (seed={config.RANDOM_STATE})")

    engineer = FeatureEngineer.load(str(engineer_path))
    model = SurgeryClassifier.load(str(model_path))

    X_test, y_test = engineer.transform(test_data_20, verbose=verbose)

    internal_eval_dir = output_dir / 'eval_internal'
    internal_eval_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(config=config)
    results = evaluator.evaluate(model, X_test, y_test, verbose=True)
    evaluator.generate_all_plots(model, save_dir=str(internal_eval_dir), show=False)
    evaluator.save_results(save_dir=str(internal_eval_dir))
    print(f"\n Internal evaluation saved to: {internal_eval_dir}")

    return results


def run_external_eval(config, output_dir, verbose=True):
    """
    Evaluate an already-trained model on a caller-supplied external dataset.

    Loads the FeatureEngineer and model saved under ``<output_dir>/models``
    and only ever calls ``.transform()`` / ``.predict()`` on the new data --
    it never fits or refits any imputer, encoder, scaler, or the model
    itself. This is the only supported path for scoring external data.
    """
    _print_header("M0_CLIN XGBoost Text Pipeline - External Evaluation")

    models_dir = output_dir / 'models'
    engineer_path = models_dir / 'feature_engineer.pkl'
    model_path = models_dir / config.BEST_MODEL_NAME
    if not engineer_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No trained artifacts found under {models_dir}. Run --mode train first."
        )

    loader = DataLoader(config=config)
    raw_df = loader._load_single_file(config.get_absolute_paths()['train_data_file'], file_type='external', verbose=verbose)
    if raw_df is None:
        raise ValueError(f"Failed to load input data file: {config.TRAIN_DATA_FILE}")
    clean_df, _ = loader._clean_dataframe(raw_df, preserve_name='external-eval', verbose=verbose)

    engineer = FeatureEngineer.load(str(engineer_path))
    model = SurgeryClassifier.load(str(model_path))

    X_ext, y_ext = engineer.transform(clean_df, verbose=verbose)
    if y_ext is None or y_ext.isna().all():
        raise ValueError(
            f"External data file has no usable '{config.TARGET_COLUMN}' labels; "
            "labeled data is required to compute evaluation metrics."
        )

    external_eval_dir = output_dir / 'eval_external'
    external_eval_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(config=config)
    results = evaluator.evaluate(model, X_ext, y_ext, verbose=True)
    evaluator.generate_all_plots(model, save_dir=str(external_eval_dir), show=False)
    evaluator.save_results(save_dir=str(external_eval_dir))
    print(f"\n External evaluation saved to: {external_eval_dir}")

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="M0_CLIN: XGBoost binary classifier training/evaluation pipeline "
                    "(surgery-need prediction from tabular clinical features)."
    )
    parser.add_argument(
        '--mode', choices=['train', 'internal-eval', 'external-eval'], default='train',
        help="train: fit preprocessing + model and evaluate on the internal 20%% test "
             "split and an external validation file. internal-eval: re-score a "
             "previously trained model on the reconstructed internal test partition "
             "(no fitting). external-eval: score a previously trained model on a new "
             "external file (no fitting). Default: train."
    )
    parser.add_argument(
        '--input-data', type=str, default=None,
        help="Path to the main data file (.csv, .xlsx, or .xls): the training file "
             "for mode=train/internal-eval, or the file to score for mode=external-eval. "
             f"Defaults to the bundled synthetic example ({DEFAULT_TRAIN_DATA})."
    )
    parser.add_argument(
        '--external-data', type=str, default=None,
        help="Path to the external validation data file (.csv, .xlsx, or .xls). Only "
             f"used for mode=train (the leakage-safe pipeline trains on --input-data "
             f"and separately validates on this file). Defaults to the bundled "
             f"synthetic example ({DEFAULT_EXTERNAL_DATA})."
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help=f"Directory for model artifacts and evaluation outputs. "
             f"Defaults to {DEFAULT_OUTPUT_DIR}."
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help=f"Random seed, overriding Config.RANDOM_STATE (default {Config.RANDOM_STATE}). "
             "Propagated consistently to the train/test split, SMOTE, and the XGBoost "
             "model. For --mode internal-eval this MUST match the seed used for the "
             "original train run, or the reconstructed test partition will not match "
             "the model's actual held-out set."
    )
    parser.add_argument('--quiet', action='store_true', help="Reduce console output verbosity.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet

    input_data = args.input_data
    external_data = args.external_data
    if input_data is None and args.mode in ('train', 'internal-eval'):
        input_data = str(DEFAULT_TRAIN_DATA)
    if input_data is None and args.mode == 'external-eval':
        input_data = str(DEFAULT_EXTERNAL_DATA)
    if external_data is None and args.mode == 'train':
        external_data = str(DEFAULT_EXTERNAL_DATA)

    config, output_dir = build_run_config(
        input_data=input_data,
        external_data=external_data,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    try:
        if args.mode == 'train':
            run_train(config, output_dir, verbose=verbose)
        elif args.mode == 'internal-eval':
            run_internal_eval(config, output_dir, verbose=verbose)
        elif args.mode == 'external-eval':
            run_external_eval(config, output_dir, verbose=verbose)
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\n Runtime error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
