"""
Image transform tests for M0_IMG (image_pipeline), using a single
programmatically generated solid-color synthetic image (Pillow
Image.new()) -- never a real patient photo. No random-augmentation
output is checked pixel-for-pixel, since that would not be a meaningful
assertion for randomized transforms.
"""

import torch
from PIL import Image

from image_pipeline.src.config import Config
from image_pipeline.src.data_utils import create_data_transforms


def _make_synthetic_image(path, size=(64, 64), color=(120, 60, 200)):
    Image.new('RGB', size, color).save(path)


def test_train_and_val_transforms_produce_correct_tensor_shape(tmp_path):
    img_path = tmp_path / 'synthetic_0001.png'
    _make_synthetic_image(img_path)

    train_transform, val_transform = create_data_transforms(Config.IMAGE_SIZE)

    image = Image.open(img_path).convert('RGB')
    train_tensor = train_transform(image)
    val_tensor = val_transform(image.copy())

    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(val_tensor, torch.Tensor)
    assert tuple(train_tensor.shape) == (3, 224, 224)
    assert tuple(val_tensor.shape) == (3, 224, 224)


def test_train_transform_includes_configured_augmentation_components():
    train_transform, _ = create_data_transforms(Config.IMAGE_SIZE)
    type_names = [type(t).__name__ for t in train_transform.transforms]
    assert type_names == [
        'Resize', 'RandomHorizontalFlip', 'RandomVerticalFlip',
        'RandomRotation', 'ColorJitter', 'RandomAffine',
        'RandomResizedCrop', 'ToTensor', 'Normalize',
    ]


def test_val_transform_has_no_random_augmentation():
    _, val_transform = create_data_transforms(Config.IMAGE_SIZE)
    type_names = [type(t).__name__ for t in val_transform.transforms]
    assert type_names == ['Resize', 'ToTensor', 'Normalize']


def test_normalize_uses_imagenet_mean_std():
    train_transform, val_transform = create_data_transforms(Config.IMAGE_SIZE)
    for compose in (train_transform, val_transform):
        normalize = compose.transforms[-1]
        assert list(normalize.mean) == Config.NORMALIZATION['mean']
        assert list(normalize.std) == Config.NORMALIZATION['std']
        assert list(normalize.mean) == [0.485, 0.456, 0.406]
        assert list(normalize.std) == [0.229, 0.224, 0.225]
