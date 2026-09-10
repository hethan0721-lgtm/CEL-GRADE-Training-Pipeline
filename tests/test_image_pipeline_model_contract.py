"""
Model and training contract tests for M0_IMG (image_pipeline).

Never downloads ImageNet weights. Architecture/freeze/hyperparameter
checks use a real, non-pretrained (pretrained=False) ResNet50Classifier
(structure inspection only, no forward()). The two pretrained=True
weights-selection tests monkeypatch torchvision.models.resnet50 itself,
so no real network call or download ever happens even when exercising
the pretrained=True code path.

The freeze-strategy tests below lock the canonical M0_IMG fine-tuning
behavior: only the ResNet-50 stem convolution (model.features[0], i.e.
conv1) is frozen; bn1 and layer1-layer4 of the backbone, plus the entire
custom classification head, remain trainable.
"""

import inspect

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights

from image_pipeline.src.config import Config
from image_pipeline.src import models as models_module
from image_pipeline.src.models import ResNet50Classifier, create_model
from image_pipeline.src import train as train_module
from image_pipeline.src.train import FocalLoss


def _fake_resnet_backbone():
    """
    A tiny stand-in with the same torchvision.models.resnet50() children()
    ordering/count (conv1, bn1, relu, maxpool, layer1-4, avgpool, fc) --
    used so create_model() can run its real construction path against a
    monkeypatched models.resnet50 without instantiating (or downloading)
    a real ResNet-50.
    """
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
        nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
        nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
        nn.Sequential(nn.Conv2d(4, 4, kernel_size=3)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Linear(4, 1000),
    )


def test_pretrained_false_passes_weights_none_to_resnet50(monkeypatch):
    captured = {}

    def fake_resnet50(*, weights=None, **kwargs):
        captured['weights'] = weights
        return _fake_resnet_backbone()

    monkeypatch.setattr(models_module.models, 'resnet50', fake_resnet50)

    create_model(num_classes=3, input_size=224, pretrained=False)

    assert 'weights' in captured
    assert captured['weights'] is None


def test_pretrained_true_passes_imagenet1k_v1_weights_to_resnet50(monkeypatch):
    captured = {}

    def fake_resnet50(*, weights=None, **kwargs):
        captured['weights'] = weights
        return _fake_resnet_backbone()

    monkeypatch.setattr(models_module.models, 'resnet50', fake_resnet50)

    # pretrained=True never actually downloads here because resnet50()
    # itself is monkeypatched above -- only the *argument* it receives is
    # being inspected.
    create_model(num_classes=3, input_size=224, pretrained=True)

    assert captured['weights'] is ResNet50_Weights.IMAGENET1K_V1


def test_models_source_never_references_default_weights_enum():
    source = inspect.getsource(models_module)
    assert 'ResNet50_Weights.DEFAULT' not in source
    assert 'IMAGENET1K_V1' in source


def test_canonical_model_is_resnet50_based():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    assert isinstance(model, ResNet50Classifier)
    assert type(model.resnet).__name__ == 'ResNet'


def test_pretrained_default_is_true_in_config():
    assert Config.PRETRAINED is True


def test_output_classes_is_three():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    last_linear = model.classifier[-1]
    assert isinstance(last_linear, nn.Linear)
    assert last_linear.out_features == 3


def test_classification_head_structure_matches_spec():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    layer_types = [type(m).__name__ for m in model.classifier]
    assert layer_types == [
        'Flatten', 'Dropout', 'Linear', 'BatchNorm1d', 'ReLU',
        'Dropout', 'Linear', 'BatchNorm1d', 'ReLU',
        'Dropout', 'Linear',
    ]

    dropouts = [m for m in model.classifier if isinstance(m, nn.Dropout)]
    assert [d.p for d in dropouts] == [0.4, 0.3, 0.2]

    linears = [m for m in model.classifier if isinstance(m, nn.Linear)]
    assert (linears[0].in_features, linears[0].out_features) == (2048 * 7 * 7, 512)
    assert (linears[1].in_features, linears[1].out_features) == (512, 128)
    assert (linears[2].in_features, linears[2].out_features) == (128, 3)


FREEZE_CONV1_LITERAL = "for param in model.features[0].parameters():"


def test_freeze_logic_source_matches_canonical_conv1_only_implementation():
    """
    Tripwire: locks train_model's freeze implementation to the canonical,
    intended M0_IMG fine-tuning strategy -- freezing only
    model.features[0] (the ResNet-50 stem conv1). If this text ever
    changes, this test fails first so the behavior tests below get
    updated deliberately instead of silently going stale.
    """
    source = inspect.getsource(train_module.train_model)
    assert FREEZE_CONV1_LITERAL in source
    assert "param.requires_grad = False" in source


def _apply_canonical_freeze(model):
    # Mirrors train_model's freeze block exactly (kept in sync by the
    # tripwire test above). Duplicated here rather than imported because
    # train.py does not expose freezing as a standalone function.
    if hasattr(model, 'features'):
        for param in model.features[0].parameters():
            param.requires_grad = False


def test_freeze_only_affects_stem_conv1_features0():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    _apply_canonical_freeze(model)

    conv1_params = list(model.features[0].parameters())
    assert len(conv1_params) >= 1
    assert all(not p.requires_grad for p in conv1_params)


def test_backbone_bn1_and_layer1_through_layer4_remain_trainable():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    _apply_canonical_freeze(model)

    # model.features is nn.Sequential(*resnet.children()[:-2]); index 0 is
    # conv1 (frozen above), 1 is bn1, and 4-7 are layer1-layer4 -- matching
    # torchvision.models.resnet50()'s children() ordering.
    for index in range(1, 8):
        block = model.features[index]
        params = list(block.parameters())
        assert all(p.requires_grad for p in params), (
            f"model.features[{index}] must remain fully trainable"
        )


def test_classification_head_remains_fully_trainable():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    _apply_canonical_freeze(model)
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_freeze_produces_expected_frozen_and_trainable_parameter_counts():
    model = create_model(num_classes=3, input_size=224, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())

    _apply_canonical_freeze(model)

    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv1_param_count = sum(p.numel() for p in model.features[0].parameters())

    assert frozen_count == conv1_param_count
    assert frozen_count > 0
    assert trainable_count > 0
    assert frozen_count + trainable_count == total_params


def test_scheduler_created_from_config_values():
    source = inspect.getsource(train_module.train_model)
    assert "mode=Config.SCHEDULER['mode']" in source
    assert "factor=Config.SCHEDULER['factor']" in source
    assert "patience=Config.SCHEDULER['patience']" in source


def test_scheduler_and_early_stopping_track_val_loss():
    source = inspect.getsource(train_module.train_model)
    assert "scheduler.step(val_loss)" in source
    assert "early_stopping_patience = 15" in source
    assert "if val_loss < best_val_loss:" in source


def test_best_model_saved_on_val_acc_improvement():
    source = inspect.getsource(train_module.train_model)
    assert "if val_acc > best_val_acc:" in source
    assert "torch.save(" in source


def test_focal_loss_hardcoded_hyperparameters_in_train_model():
    # alpha/gamma/label_smoothing for M0_IMG are hardcoded inside
    # train_model (not read from Config), so they are pinned here via a
    # source-level contract check rather than by calling train_model
    # (which would require a full model + real data loaders).
    source = inspect.getsource(train_module.train_model)
    assert "torch.FloatTensor([1.0, 1.2, 10.0])" in source
    assert "gamma=2.5" in source
    assert "label_smoothing=0.05" in source


def test_focal_loss_returns_finite_scalar_and_backprops():
    torch.manual_seed(0)
    logits = torch.randn(6, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1, 0, 2])
    alpha = torch.tensor([1.0, 1.2, 10.0])

    criterion = FocalLoss(alpha=alpha, gamma=2.5, label_smoothing=0.05)
    loss = criterion(logits, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_alpha_changes_loss_value():
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    targets = torch.tensor([0, 1, 2, 1, 0, 2])

    uniform = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0]), gamma=2.5, label_smoothing=0.05)
    weighted = FocalLoss(alpha=torch.tensor([1.0, 1.2, 10.0]), gamma=2.5, label_smoothing=0.05)

    assert not torch.isclose(uniform(logits, targets), weighted(logits, targets))


def test_focal_loss_gamma_changes_loss_value():
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    targets = torch.tensor([0, 1, 2, 1, 0, 2])
    alpha = torch.tensor([1.0, 1.2, 10.0])

    low_gamma = FocalLoss(alpha=alpha, gamma=0.0, label_smoothing=0.05)
    high_gamma = FocalLoss(alpha=alpha, gamma=2.5, label_smoothing=0.05)

    assert not torch.isclose(low_gamma(logits, targets), high_gamma(logits, targets))


def test_focal_loss_label_smoothing_changes_loss_value():
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    targets = torch.tensor([0, 1, 2, 1, 0, 2])
    alpha = torch.tensor([1.0, 1.2, 10.0])

    no_smoothing = FocalLoss(alpha=alpha, gamma=2.5, label_smoothing=0.0)
    smoothing = FocalLoss(alpha=alpha, gamma=2.5, label_smoothing=0.05)

    assert not torch.isclose(no_smoothing(logits, targets), smoothing(logits, targets))
