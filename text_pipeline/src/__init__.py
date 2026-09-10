#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M0_CLIN: XGBoost binary classifier pipeline for tabular clinical features.
"""

import sys as _sys

__version__ = '1.0.0'


def _ensure_utf8_stdio():
    """
    Some consoles (notably Windows with a non-UTF-8 locale codepage such as
    GBK) default stdout/stderr to an encoding that cannot represent the
    Unicode status glyphs used in this package's progress logging, which
    crashes with UnicodeEncodeError the moment anything is printed. This
    only affects how log text is encoded on the way out -- it never touches
    data, features, or model behaviour.
    """
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(_sys, stream_name, None)
        if stream is not None and hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


_ensure_utf8_stdio()

from .config import Config
from .data_loader import DataLoader, load_and_clean_data
from .feature_engineering import FeatureEngineer, prepare_features
from .models import SurgeryClassifier, create_model
from .train import ModelTrainer, train_pipeline
from .evaluate import ModelEvaluator, evaluate_model

__all__ = [
    'Config',
    'DataLoader',
    'load_and_clean_data',
    'FeatureEngineer',
    'prepare_features',
    'SurgeryClassifier',
    'create_model',
    'ModelTrainer',
    'train_pipeline',
    'ModelEvaluator',
    'evaluate_model'
]
