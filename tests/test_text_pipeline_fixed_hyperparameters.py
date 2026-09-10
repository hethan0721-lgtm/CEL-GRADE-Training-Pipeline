"""
Regression tests for the "no early stopping, no automatic hyperparameter
tuning" cleanup in text_pipeline. The canonical model is trained once with a
single fixed XGBOOST_PARAMS set (n_estimators=200, no early stopping); a
legacy RandomizedSearchCV helper exists but must never be exercised by the
default/public training path.
"""

from text_pipeline.src.config import Config
from text_pipeline.src.main import build_arg_parser
from text_pipeline.src.train import train_pipeline


def test_early_stopping_rounds_setting_no_longer_exists():
    # It was never wired into training; removed rather than fixed/added.
    assert not hasattr(Config, 'EARLY_STOPPING_ROUNDS')


def test_n_estimators_is_the_fixed_canonical_value():
    assert Config.XGBOOST_PARAMS['n_estimators'] == 200


def test_hyperparameter_tuning_disabled_by_default():
    assert Config.ENABLE_HYPERPARAMETER_TUNING is False


def test_train_pipeline_default_never_runs_hyperparameter_search():
    import inspect
    sig = inspect.signature(train_pipeline)
    assert sig.parameters['hyperparameter_tuning'].default is False


def test_cli_has_no_flag_to_enable_hyperparameter_tuning_or_early_stopping():
    parser = build_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    for forbidden in ('--tune', '--hyperparameter-tuning', '--early-stopping'):
        assert forbidden not in option_strings
