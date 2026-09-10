#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye disease severity classification system
Main entry point

Usage:
    python -m image_pipeline.src.main --mode train --data-dir <path> [--output-dir <path>] [--device auto|cpu|cuda] [--seed 42]
    python -m image_pipeline.src.main --mode evaluate --data-dir <path> [--output-dir <path>] [--checkpoint <path>]
    python -m image_pipeline.src.main --mode predict --image <path> [--output-dir <path>] [--checkpoint <path>]
"""

import argparse
import os
import pickle
import random
import numpy as np
import torch
from pathlib import Path

# Import custom modules
from .config import Config
from .data_utils import (
    load_and_preprocess_data,
    create_data_transforms,
    split_data,
    create_data_loaders
)
from .models import create_model, print_model_info
from .train import train_model, load_model
from .evaluate import (
    evaluate_model,
    plot_training_history,
    predict_single_image
)

# Directory of the image_pipeline package (parent of this src/ package).
# Used only to compute a safe default --output-dir; never a hardcoded personal path.
IMAGE_PIPELINE_DIR = Path(__file__).resolve().parent.parent


def build_arg_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Eye disease severity classification (M0_IMG ResNet-50 pipeline)"
    )
    parser.add_argument(
        '--mode', choices=['train', 'evaluate', 'predict'], default='train',
        help="Operation mode (default: train)"
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help="Path to a dataset directory containing 0/, 1/, 2/ class subfolders "
             "(required for --mode train and --mode evaluate)"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Directory to save/read models/ and results/ "
             "(default: <image_pipeline>/outputs)"
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help="Path to a model checkpoint (.pth) for evaluate/predict. "
             "Defaults to <output-dir>/models/best_model.pth"
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help="Path to a single image file (required for --mode predict)"
    )
    parser.add_argument(
        '--device', choices=['auto', 'cpu', 'cuda'], default='auto',
        help="Device selection: auto picks cuda if available else cpu; "
             "cpu forces CPU; cuda fails loudly if unavailable (default: auto)"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed, propagated to Config.RANDOM_STATE (default: 42)"
    )
    return parser


def set_global_seed(seed):
    """Seed all RNGs used by the pipeline (Python, NumPy, torch/CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_output_dir(output_dir_arg):
    """Resolve --output-dir, defaulting to <image_pipeline>/outputs (never hardcoded)."""
    if output_dir_arg is not None:
        return Path(output_dir_arg).resolve()
    return (IMAGE_PIPELINE_DIR / 'outputs').resolve()


def resolve_device(requested):
    """Resolve the torch device from the --device choice."""
    if requested == 'cpu':
        return torch.device('cpu')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but torch.cuda.is_available() is False "
                "on this machine. Use --device auto or --device cpu instead."
            )
        return torch.device('cuda')
    # auto
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def check_class_subfolders(data_dir):
    """Verify data_dir/0, data_dir/1, data_dir/2 exist. Does not read any files inside."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir not found or not a directory: {data_dir}")
    for class_name in ('0', '1', '2'):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class subfolder not found: {class_dir} "
                f"(--data-dir must contain 0/, 1/, 2/ subfolders)"
            )


def train_mode(device, data_dir, models_dir, results_dir):
    """Training mode workflow."""
    print("\n=== Training Mode ===")
    Config.print_config()

    # Load and preprocess data (with class balancing)
    print("\n1. Loading data...")
    img_paths, labels = load_and_preprocess_data(
        str(data_dir),
        balance_classes=Config.BALANCE_CLASSES,
        max_samples_per_class=Config.MAX_SAMPLES_PER_CLASS
    )

    # Split data
    print("\n2. Splitting data...")
    train_data, val_data, test_data = split_data(
        img_paths,
        labels,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE
    )

    # Create data transforms
    print("\n3. Creating data transforms...")
    train_transform, val_transform = create_data_transforms(Config.IMAGE_SIZE)

    # Create data loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        train_transform, val_transform,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )

    # Create model
    print("\n5. Creating model...")
    num_classes = 3  # Fixed to 3 classes: Mild, Moderate, Severe
    model = create_model(
        num_classes=num_classes,
        input_size=Config.IMAGE_SIZE,
        pretrained=Config.PRETRAINED
    )
    print_model_info(model)

    # Train model
    print("\n6. Starting training...")
    print(f"Using mixed precision: {device.type == 'cuda'}")
    train_history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        device=device,
        save_dir=str(models_dir),
        use_amp=True  # Enable mixed precision training
    )

    # Plot training history
    print("\n7. Generating training history plot...")
    plot_training_history(train_history, str(results_dir))

    # Save label mapping
    label_mapping = Config.LABEL_MAPPING
    with open(models_dir / 'label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)

    # Evaluate best model
    print("\n8. Evaluating best model...")
    model, checkpoint = load_model(model, best_model_path, device)
    accuracy, report, predictions, labels = evaluate_model(
        model, test_loader, label_mapping, device, str(results_dir)
    )

    print("\nTraining completed!")
    print(f"Best model: {best_model_path}")
    print(f"Test accuracy: {accuracy:.4f}")


def evaluate_mode(device, data_dir, checkpoint_path, results_dir):
    """Evaluation mode workflow."""
    print("\n=== Evaluation Mode ===")

    label_mapping_path = checkpoint_path.parent / 'label_mapping.pkl'
    if not label_mapping_path.is_file():
        raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")

    with open(label_mapping_path, 'rb') as f:
        label_mapping = pickle.load(f)

    # Load data (only for test set, no balancing needed)
    img_paths, labels = load_and_preprocess_data(
        str(data_dir),
        balance_classes=False  # Don't balance for evaluation
    )
    train_data, val_data, test_data = split_data(
        img_paths,
        labels,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE,
        apply_augmentation=False  # No sample augmentation in evaluation mode
    )

    # Create test data loader
    _, val_transform = create_data_transforms(Config.IMAGE_SIZE)
    _, _, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        val_transform, val_transform,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )

    # Create and load model
    num_classes = 3  # Fixed to 3 classes
    model = create_model(
        num_classes=num_classes,
        input_size=Config.IMAGE_SIZE,
        pretrained=False  # No need for pretrained weights during evaluation
    )

    model, checkpoint = load_model(model, str(checkpoint_path), device)

    # Evaluate model
    accuracy, report, predictions, labels = evaluate_model(
        model, test_loader, label_mapping, device, str(results_dir)
    )

    print(f"\nEvaluation completed! Test accuracy: {accuracy:.4f}")


def predict_mode(device, image_path, checkpoint_path):
    """Prediction mode workflow."""
    print("\n=== Prediction Mode ===")
    print(f"Image path: {image_path}")

    label_mapping_path = checkpoint_path.parent / 'label_mapping.pkl'
    if not label_mapping_path.is_file():
        raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")

    with open(label_mapping_path, 'rb') as f:
        label_mapping = pickle.load(f)

    # Create model
    num_classes = 3  # Fixed to 3 classes
    model = create_model(
        num_classes=num_classes,
        input_size=Config.IMAGE_SIZE,
        pretrained=False
    )

    # Load model weights
    model, checkpoint = load_model(model, str(checkpoint_path), device)

    # Create preprocessing transform
    _, transform = create_data_transforms(Config.IMAGE_SIZE)

    # Predict
    predicted_label, confidence, probabilities = predict_single_image(
        model, str(image_path), transform, label_mapping, device
    )

    print("\nPrediction result:")
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print("\nClass probabilities:")
    for i, prob in enumerate(probabilities):
        class_name = label_mapping[i]
        print(f"  {class_name}: {prob:.4f}")


def main():
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    Config.RANDOM_STATE = args.seed
    set_global_seed(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    models_dir = output_dir / 'models'
    results_dir = output_dir / 'results'
    Config.MODEL_SAVE_DIR = str(models_dir)
    Config.RESULTS_SAVE_DIR = str(results_dir)

    Config.validate_config()

    device = resolve_device(args.device)
    Config.DEVICE = device.type
    print(f"Using device: {device}")

    checkpoint_path = (
        Path(args.checkpoint).resolve() if args.checkpoint is not None
        else models_dir / 'best_model.pth'
    )

    if args.mode == 'train':
        if args.data_dir is None:
            parser.error("--data-dir is required when --mode train")
        data_dir = Path(args.data_dir).resolve()
        check_class_subfolders(data_dir)

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        train_mode(device, data_dir, models_dir, results_dir)

    elif args.mode == 'evaluate':
        if args.data_dir is None:
            parser.error("--data-dir is required when --mode evaluate")
        data_dir = Path(args.data_dir).resolve()
        check_class_subfolders(data_dir)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        os.makedirs(results_dir, exist_ok=True)

        evaluate_mode(device, data_dir, checkpoint_path, results_dir)

    elif args.mode == 'predict':
        if args.image is None:
            parser.error("--image is required when --mode predict")
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        predict_mode(device, image_path, checkpoint_path)


if __name__ == '__main__':
    main()
