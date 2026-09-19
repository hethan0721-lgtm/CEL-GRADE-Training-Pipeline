"""
Release-verification tests for the official M0_CLIN (7-feature) weight
files published under ``text_pipeline/pretrained/`` and referenced by the
image-model GitHub Release.

These tests do not fit or train anything; they only load the already
frozen, hash-verified release artifacts and confirm they behave exactly as
documented in ``MODEL_WEIGHTS.md``.
"""

import hashlib
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = REPO_ROOT / 'text_pipeline' / 'pretrained'
MODEL_PATH = PRETRAINED_DIR / 'best_xgboost_model.pkl'
ENGINEER_PATH = PRETRAINED_DIR / 'feature_engineer.pkl'
FEATURE_CONFIG_PATH = PRETRAINED_DIR / 'feature_configuration.json'

EXPECTED_MODEL_SHA256 = '27a93862070cd469b241f4c31c6332b4b451a177853f41d5cbb337b02f25c93e'
EXPECTED_ENGINEER_SHA256 = '9232a304b2db4fdfd2a2f8e26fca274b8d8bc97cad4a7a67b2088f5a15299d50'
EXPECTED_IMAGE_INFERENCE_SHA256 = '52d52beff82ef582ae22c99955cf461bcc394b5435ec7b69c80bfb781d426b47'
EXPECTED_IMAGE_LABELMAP_SHA256 = '57bdad64003c4e5cf169734bfc5d15937be4c552562205141017d8310fe7a59e'
RELEASE_TAG = 'model-weights-v1.0.0'

EXPECTED_FEATURE_ORDER = [
    '矫正视力', '矫正球镜度数(D)', '矫正柱镜度数(D)', 'IOLMaster-Cyl(D)',
    '年龄', '是否配合检查', '脱位程度',
]


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


# 1. Text weight files exist -------------------------------------------------

def test_text_weight_files_exist():
    for path in (MODEL_PATH, ENGINEER_PATH, FEATURE_CONFIG_PATH, PRETRAINED_DIR / 'SHA256SUMS.txt'):
        assert path.is_file(), f"expected release file missing: {path}"


# 2 & 3. Hashes are correct ---------------------------------------------------

def test_model_pkl_sha256_matches_published_manifest():
    assert _sha256_of(MODEL_PATH) == EXPECTED_MODEL_SHA256


def test_feature_engineer_pkl_sha256_matches_published_manifest():
    assert _sha256_of(ENGINEER_PATH) == EXPECTED_ENGINEER_SHA256


def test_sha256sums_txt_matches_actual_files():
    text = (PRETRAINED_DIR / 'SHA256SUMS.txt').read_text(encoding='utf-8')
    recorded = {}
    for line in text.strip().splitlines():
        digest, name = line.split()
        recorded[name.lstrip('*')] = digest

    assert recorded['best_xgboost_model.pkl'] == EXPECTED_MODEL_SHA256
    assert recorded['feature_engineer.pkl'] == EXPECTED_ENGINEER_SHA256
    assert 'SHA256SUMS.txt' not in recorded, "SHA256SUMS.txt must not record its own hash"


# 4. The compatibility loader only accepts whitelisted classes ---------------

def test_compatibility_loader_maps_only_the_one_legacy_config_class():
    from text_pipeline.src.compatibility_loader import CompatibilityUnpickler
    import io

    u = CompatibilityUnpickler(io.BytesIO(b''))
    resolved = u.find_class('text_classification_7feature.code.config', 'Config')
    from text_pipeline.src.config import Config
    assert resolved is Config


def test_compatibility_loader_allows_known_safe_third_party_prefixes():
    from text_pipeline.src.compatibility_loader import CompatibilityUnpickler
    import io

    u = CompatibilityUnpickler(io.BytesIO(b''))
    # These are real classes/functions the frozen pickles actually reference;
    # resolving them must not raise.
    assert u.find_class('collections', 'OrderedDict') is not None
    assert u.find_class('builtins', 'bytearray') is not None
    assert u.find_class('numpy', 'dtype') is not None


def test_compatibility_loader_rejects_unknown_module_or_class():
    from text_pipeline.src.compatibility_loader import (
        CompatibilityUnpickler, UnknownLegacyReferenceError,
    )
    import io

    u = CompatibilityUnpickler(io.BytesIO(b''))
    with pytest.raises(UnknownLegacyReferenceError):
        u.find_class('text_classification_7feature.code.feature_engineering', 'FeatureEngineer')
    with pytest.raises(UnknownLegacyReferenceError):
        u.find_class('os', 'system')
    with pytest.raises(UnknownLegacyReferenceError):
        u.find_class('some.totally.unrelated.module', 'Anything')


# 5 & 6. Real load, model type, and 7-feature order ---------------------------

def test_release_weights_load_via_public_pipeline_and_have_correct_feature_order():
    from text_pipeline.src.models import SurgeryClassifier
    from text_pipeline.src.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer.load(str(ENGINEER_PATH))
    classifier = SurgeryClassifier.load(str(MODEL_PATH))

    assert classifier.model_type == 'xgboost'
    assert classifier.feature_names == EXPECTED_FEATURE_ORDER
    assert engineer.feature_names == EXPECTED_FEATURE_ORDER
    assert classifier.model.n_features_in_ == 7
    assert classifier.model.n_features_in_ != 5, "must not be the legacy 5-feature model"


# 7 & 8. predict / predict_proba on complete-case synthetic data -------------

def test_predict_and_predict_proba_on_complete_case_synthetic_data():
    from text_pipeline.src.models import SurgeryClassifier
    from text_pipeline.src.feature_engineering import FeatureEngineer
    from text_pipeline.src.data_loader import DataLoader
    from text_pipeline.src.config import Config

    engineer = FeatureEngineer.load(str(ENGINEER_PATH))
    classifier = SurgeryClassifier.load(str(MODEL_PATH))

    examples_dir = REPO_ROOT / 'examples' / 'synthetic_tabular_data'

    class RunConfig(Config):
        pass

    RunConfig.TRAIN_DATA_FILE = str(examples_dir / 'synthetic_train.csv')
    RunConfig.VAL_DATA_FILE = str(examples_dir / 'synthetic_external.csv')
    config = RunConfig()

    loader = DataLoader(config=config)
    train_data, val_data = loader.load_train_val_data(verbose=False)
    train_data, val_data = loader.clean_train_val_data(verbose=False)

    # The frozen LabelEncoders were fit on real data with zero missing
    # categorical values, so they cannot accept the literal "nan" string
    # DataLoader substitutes for missing categoricals -- restrict to
    # complete cases, matching the documented limitation in MODEL_WEIGHTS.md.
    mask_cols = ['是否配合检查', '脱位程度']
    complete = train_data[~train_data[mask_cols].isin(['nan']).any(axis=1)].reset_index(drop=True)
    assert len(complete) > 0

    X, _ = engineer.transform(complete, verbose=False)
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    assert predictions.shape == (len(complete),)
    assert probabilities.shape == (len(complete), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


# 9. .gitignore still blocks everything else ---------------------------------

def test_gitignore_still_blocks_other_weights_and_outputs():
    def is_ignored(rel_path):
        result = subprocess.run(
            ['git', 'check-ignore', '-q', rel_path],
            cwd=REPO_ROOT, capture_output=True,
        )
        return result.returncode == 0

    assert is_ignored('text_pipeline/outputs/models/best_xgboost_model.pkl')
    assert is_ignored('text_pipeline/outputs/models/training_history.pkl')
    assert is_ignored('text_pipeline/pretrained/some_other_model.pkl')
    assert is_ignored('image_pipeline/models/best_model.pth')
    assert is_ignored('anywhere/checkpoint.pt')

    # The two whitelisted files must NOT be ignored.
    assert not is_ignored('text_pipeline/pretrained/best_xgboost_model.pkl')
    assert not is_ignored('text_pipeline/pretrained/feature_engineer.pkl')


# 10. Privacy scan -------------------------------------------------------------

_FORBIDDEN_BYTE_PATTERNS = [
    ('name_cn', '姓名'.encode('utf-8')),
    ('id_card_cn', '身份证'.encode('utf-8')),
    ('admission_no_cn', '住院号'.encode('utf-8')),
    ('record_cn', '病历'.encode('utf-8')),
    ('id_card_number_shape', re.compile(rb'(?<![\d.])\d{17}[\dXx](?!\d)').pattern),
    ('phone_number_shape', re.compile(rb'(?<!\d)1[3-9]\d{9}(?!\d)').pattern),
    ('e_drive_eye_path', rb'[Ee]:[\\/]eye'),
    ('f_drive_eye_path', rb'[Ff]:[\\/]eye'),
    ('token', rb'token'),
    ('password', rb'password'),
    ('api_key', rb'api[_-]?key'),
    ('secret', rb'secret'),
]


@pytest.mark.parametrize('path', [MODEL_PATH, ENGINEER_PATH, FEATURE_CONFIG_PATH,
                                   REPO_ROOT / 'text_pipeline' / 'src' / 'compatibility_loader.py'])
def test_release_files_contain_no_pii_or_forbidden_paths(path):
    data = path.read_bytes()
    offenders = []
    for label, pattern in _FORBIDDEN_BYTE_PATTERNS:
        if re.search(pattern, data):
            offenders.append(label)
    assert not offenders, f"{path.name} matched forbidden pattern(s): {offenders}"


# 11. No .pth file ever tracked in Git ----------------------------------------

def test_no_pth_file_is_tracked_in_git():
    result = subprocess.run(
        ['git', 'ls-files'], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    tracked = result.stdout.splitlines()
    pth_files = [f for f in tracked if f.endswith('.pth')]
    assert pth_files == [], f"found .pth file(s) tracked in git: {pth_files}"


# 12. README release tag / filenames / SHA256 are correct --------------------

def test_readme_references_correct_release_tag_and_hashes():
    readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    model_weights = (REPO_ROOT / 'MODEL_WEIGHTS.md').read_text(encoding='utf-8')

    assert RELEASE_TAG in readme
    assert RELEASE_TAG in model_weights

    for doc in (readme, model_weights):
        assert 'best_model_inference.pth' in doc
        assert 'label_mapping.pkl' in doc

    assert EXPECTED_MODEL_SHA256 in model_weights
    assert EXPECTED_ENGINEER_SHA256 in model_weights
    assert EXPECTED_IMAGE_INFERENCE_SHA256 in model_weights
    assert EXPECTED_IMAGE_LABELMAP_SHA256 in model_weights
