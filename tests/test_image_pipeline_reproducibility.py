"""
Randomness / reproducibility control tests for M0_IMG (image_pipeline).

Covers set_global_seed(), the seeded DataLoader generators, the
worker_init_fn, and the undersampling seed source. No CUDA training is
started; DataLoader tests use fake string "paths" that are never opened
(EyeDataset only reads a path lazily in __getitem__, which these tests
never call), and the one test needing real files uses Pillow's
Image.new() to create synthetic images in tmp_path.
"""

import random

import numpy as np
import torch
from PIL import Image

from image_pipeline.src.config import Config
from image_pipeline.src.data_utils import (
    create_data_loaders,
    load_and_preprocess_data,
    _seed_worker,
)
from image_pipeline.src.main import set_global_seed


def test_set_global_seed_reproduces_python_random_sequence():
    set_global_seed(123)
    seq1 = [random.random() for _ in range(5)]
    set_global_seed(123)
    seq2 = [random.random() for _ in range(5)]
    assert seq1 == seq2


def test_set_global_seed_reproduces_numpy_sequence():
    set_global_seed(123)
    seq1 = np.random.rand(5).tolist()
    set_global_seed(123)
    seq2 = np.random.rand(5).tolist()
    assert seq1 == seq2


def test_set_global_seed_reproduces_torch_sequence():
    set_global_seed(123)
    seq1 = torch.rand(5).tolist()
    set_global_seed(123)
    seq2 = torch.rand(5).tolist()
    assert seq1 == seq2


def test_set_global_seed_sets_cudnn_flags():
    set_global_seed(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_data_loaders_use_seeded_generator_matching_random_state():
    # Fake string "paths" -- create_data_loaders only builds samplers here,
    # it never calls EyeDataset.__getitem__, so no file is ever opened.
    train_data = (['a'] * 6, [0, 1, 2, 0, 1, 2])
    val_data = (['b'] * 6, [0, 1, 2, 0, 1, 2])
    test_data = (['c'] * 6, [0, 1, 2, 0, 1, 2])

    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        train_transform=None, val_transform=None,
        batch_size=2, num_workers=0, random_state=999
    )

    for loader in (train_loader, val_loader, test_loader):
        assert loader.generator is not None
        assert loader.generator.initial_seed() == 999
        assert loader.worker_init_fn is _seed_worker


def test_data_loaders_default_to_config_random_state(monkeypatch):
    monkeypatch.setattr(Config, 'RANDOM_STATE', 555)
    train_data = (['a'] * 4, [0, 1, 0, 1])
    val_data = (['b'] * 4, [0, 1, 0, 1])
    test_data = (['c'] * 4, [0, 1, 0, 1])

    train_loader, _, _ = create_data_loaders(
        train_data, val_data, test_data,
        train_transform=None, val_transform=None,
        batch_size=2, num_workers=0
    )
    assert train_loader.generator.initial_seed() == 555


def test_seed_worker_sets_numpy_and_python_random(monkeypatch):
    torch.manual_seed(1)
    calls = []

    original_np_seed = np.random.seed
    original_random_seed = random.seed

    def spy_np_seed(value):
        calls.append(('numpy', value))
        original_np_seed(value)

    def spy_random_seed(value):
        calls.append(('random', value))
        original_random_seed(value)

    monkeypatch.setattr(np.random, 'seed', spy_np_seed)
    monkeypatch.setattr(random, 'seed', spy_random_seed)

    _seed_worker(0)

    kinds = {kind for kind, _ in calls}
    assert 'numpy' in kinds
    assert 'random' in kinds


def test_undersampling_seed_follows_config_random_state_not_hardcoded_42(tmp_path, monkeypatch):
    data_dir = tmp_path / 'temporary_dataset'
    class_counts = {0: 5, 1: 8, 2: 5}
    for label, count in class_counts.items():
        class_dir = data_dir / str(label)
        class_dir.mkdir(parents=True)
        for i in range(count):
            Image.new('RGB', (32, 32), (10, 20, 30)).save(
                class_dir / f'synthetic_{i:04d}.png'
            )

    seen_seeds = []
    original_seed = np.random.seed

    def spy_seed(value):
        seen_seeds.append(value)
        original_seed(value)

    monkeypatch.setattr(np.random, 'seed', spy_seed)
    monkeypatch.setattr(Config, 'RANDOM_STATE', 999)

    load_and_preprocess_data(
        str(data_dir), balance_classes=True,
        max_samples_per_class={0: None, 1: 3, 2: None}
    )

    assert 999 in seen_seeds
    assert 42 not in seen_seeds
