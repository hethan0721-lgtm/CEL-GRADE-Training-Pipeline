"""
Temp-directory data loading and image-level split tests for M0_IMG
(image_pipeline), using only programmatically generated synthetic
images -- solid-color 32x32 PNGs created with Pillow's Image.new(),
never real patient photos. No model training happens in this file.
"""

import re

from PIL import Image

from image_pipeline.src.config import Config
from image_pipeline.src.data_utils import load_and_preprocess_data, split_data

_COLORS = {0: (200, 50, 50), 1: (50, 200, 50), 2: (50, 50, 200)}


def _build_synthetic_dataset(root, per_class=20):
    """Create root/{0,1,2}/synthetic_XXXX.png -- solid-color, non-medical."""
    for label in (0, 1, 2):
        class_dir = root / str(label)
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            Image.new('RGB', (32, 32), _COLORS[label]).save(
                class_dir / f'synthetic_{i:04d}.png'
            )
    return root


def test_only_supported_extensions_are_read(tmp_path):
    data_dir = _build_synthetic_dataset(tmp_path / 'temporary_dataset', per_class=5)
    (data_dir / '0' / 'not_an_image.txt').write_text('junk, not an image')

    img_paths, labels = load_and_preprocess_data(str(data_dir), balance_classes=False)

    assert all(
        any(p.lower().endswith(ext) for ext in Config.SUPPORTED_EXTENSIONS)
        for p in img_paths
    )
    assert not any(p.endswith('not_an_image.txt') for p in img_paths)


def test_all_three_classes_discovered_with_correct_labels(tmp_path):
    data_dir = _build_synthetic_dataset(tmp_path / 'temporary_dataset', per_class=20)
    img_paths, labels = load_and_preprocess_data(str(data_dir), balance_classes=False)

    assert set(labels) == {0, 1, 2}
    assert labels.count(0) == 20
    assert labels.count(1) == 20
    assert labels.count(2) == 20
    assert len(img_paths) == 60


def test_only_synthetic_filenames_are_present(tmp_path):
    data_dir = _build_synthetic_dataset(tmp_path / 'temporary_dataset', per_class=5)
    img_paths, labels = load_and_preprocess_data(str(data_dir), balance_classes=False)

    pattern = re.compile(r'synthetic_\d{4}\.png$')
    assert all(pattern.search(p) for p in img_paths), (
        "Only the programmatically generated synthetic_NNNN.png filenames "
        "should ever be discovered by the loader in this test suite -- "
        "no patient names, dates, or eye-side info."
    )


def test_image_level_stratified_split_has_no_path_overlap(tmp_path):
    data_dir = _build_synthetic_dataset(tmp_path / 'temporary_dataset', per_class=20)
    img_paths, labels = load_and_preprocess_data(str(data_dir), balance_classes=False)

    train_data, val_data, test_data = split_data(
        img_paths, labels,
        test_size=Config.TEST_SIZE, val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE, apply_augmentation=False
    )

    train_paths = set(train_data[0])
    val_paths = set(val_data[0])
    test_paths = set(test_data[0])

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    total = len(train_paths) + len(val_paths) + len(test_paths)
    assert total == 60

    # Roughly matches test_size=0.20 / val_size=0.15 given stratified rounding
    # on a small synthetic sample (verified empirically: 39/9/12 for N=60).
    assert 8 <= len(test_paths) <= 16
    assert 6 <= len(val_paths) <= 13


def test_same_seed_gives_identical_split(tmp_path):
    data_dir = _build_synthetic_dataset(tmp_path / 'temporary_dataset', per_class=20)
    img_paths, labels = load_and_preprocess_data(str(data_dir), balance_classes=False)

    split_a = split_data(img_paths, labels, test_size=0.2, val_size=0.15,
                          random_state=42, apply_augmentation=False)
    split_b = split_data(img_paths, labels, test_size=0.2, val_size=0.15,
                          random_state=42, apply_augmentation=False)

    assert list(split_a[0][0]) == list(split_b[0][0])
    assert list(split_a[1][0]) == list(split_b[1][0])
    assert list(split_a[2][0]) == list(split_b[2][0])


def test_different_seed_gives_different_split(tmp_path):
    data_dir = _build_synthetic_dataset(tmp_path / 'temporary_dataset', per_class=20)
    img_paths, labels = load_and_preprocess_data(str(data_dir), balance_classes=False)

    split_a = split_data(img_paths, labels, test_size=0.2, val_size=0.15,
                          random_state=42, apply_augmentation=False)
    split_b = split_data(img_paths, labels, test_size=0.2, val_size=0.15,
                          random_state=123, apply_augmentation=False)

    assert list(split_a[0][0]) != list(split_b[0][0])
