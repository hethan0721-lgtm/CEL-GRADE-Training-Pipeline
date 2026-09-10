"""
Configuration lock test for M0_IMG (image_pipeline).

Pins the real M0_IMG training configuration as read from
``image_pipeline.src.config.Config`` so any future accidental edit to
these values is caught immediately. Field names follow the actual source
code; only the numeric/structural values are asserted (per the M0_IMG
canonical-pipeline audit).
"""

from image_pipeline.src.config import Config


def test_image_size_is_224():
    assert Config.IMAGE_SIZE == 224


def test_pretrained_default_is_true():
    assert Config.PRETRAINED is True


def test_batch_size_is_16():
    assert Config.BATCH_SIZE == 16


def test_num_epochs_is_200():
    assert Config.NUM_EPOCHS == 200


def test_learning_rate_is_5e_minus_5():
    assert Config.LEARNING_RATE == 5e-5


def test_random_state_is_42():
    assert Config.RANDOM_STATE == 42


def test_train_val_test_split_proportions():
    assert Config.TEST_SIZE == 0.20
    assert Config.VAL_SIZE == 0.15


def test_three_classes_with_correct_label_order():
    # There is no literal NUM_CLASSES field in Config; the 3-class contract
    # is pinned via LABEL_MAPPING, which main.py's hardcoded
    # ``num_classes = 3`` must stay in sync with.
    assert Config.LABEL_MAPPING == {0: '轻', 1: '中', 2: '重'}
    assert Config.ENGLISH_LABEL_MAPPING == {
        '轻': 'Mild', '中': 'Moderate', '重': 'Severe'
    }


def test_class1_undersampling_cap_is_1200_others_unlimited():
    assert Config.MAX_SAMPLES_PER_CLASS == {0: None, 1: 1200, 2: None}
    assert Config.BALANCE_CLASSES is True


def test_scheduler_config_matches_train_py_call():
    assert Config.SCHEDULER['mode'] == 'min'
    assert Config.SCHEDULER['factor'] == 0.5
    assert Config.SCHEDULER['patience'] == 7


def test_normalization_uses_imagenet_mean_std():
    assert Config.NORMALIZATION['mean'] == [0.485, 0.456, 0.406]
    assert Config.NORMALIZATION['std'] == [0.229, 0.224, 0.225]


def test_supported_extensions_are_lowercase_only():
    assert Config.SUPPORTED_EXTENSIONS == ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


def test_validate_config_accepts_current_defaults():
    assert Config.validate_config() is True
