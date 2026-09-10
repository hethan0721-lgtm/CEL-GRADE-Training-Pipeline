import os

class Config:
    """Configuration class containing all training and model parameters."""

    # Model configuration
    IMAGE_SIZE = 224  # Input image resolution (ResNet50 recommended size)
    PRETRAINED = True  # Use pretrained weights for better feature extraction (IMPORTANT for small datasets)

    # Training configuration
    BATCH_SIZE = 16  # Batch size (further reduced for better gradient estimates on small classes)
    NUM_EPOCHS = 200  # Number of training epochs (increased for better convergence)
    LEARNING_RATE = 0.00005  # Learning rate (further reduced for fine-tuning pretrained model)

    # Data split configuration
    TEST_SIZE = 0.2  # Proportion for test set (increased to get more severe samples)
    VAL_SIZE = 0.15   # Proportion for validation set
    RANDOM_STATE = 42  # Random seed for reproducibility
    # This gives: Train 65%, Val 15%, Test 20%
    # With 41 severe images: Train~27, Val~6, Test~8 (more reliable evaluation)

    # Data loading configuration
    NUM_WORKERS = 2  # Number of worker processes for data loading

    # Output path configuration
    MODEL_SAVE_DIR = 'models'  # Directory for model checkpoints
    RESULTS_SAVE_DIR = 'results'  # Directory for evaluation results

    # Device configuration
    DEVICE = 'cuda'  # Device to use ('cuda' or 'cpu')

    # Data augmentation configuration (train split only; matches actual
    # transforms.Compose pipeline built in data_utils.create_data_transforms)
    AUGMENTATION = {
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.3,
        'rotation_degrees': 15,
        'color_jitter': {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.2,
            'hue': 0.1
        },
        'affine': {
            'degrees': 0,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1)
        },
        'random_resized_crop': {
            'scale': (0.85, 1.0)
        }
    }

    # Sample augmentation multipliers (to address class imbalance)
    # NOTE: Disabled because we use undersampling to balance classes
    # Original distribution: 轻(0): 723, 中(1): 1865, 重(2): 810
    # After undersampling: ~766 samples per class
    SAMPLE_AUGMENTATION = {
        'enable': False,  # Disabled - using undersampling instead
        'multipliers': {
            0: 1,   # 轻 (class 0)
            1: 1,   # 中 (class 1)
            2: 1    # 重 (class 2)
        }
    }

    # Class balancing configuration (undersampling)
    BALANCE_CLASSES = True  # Enable class balancing by undersampling
    # Set max samples per class: None = no limit, int = limit for all, dict = per-class limit
    MAX_SAMPLES_PER_CLASS = {
        0: None,   # 轻: no limit (~855)
        1: 1200,   # 中: limit to 1200 (from 1865)
        2: None    # 重: no limit (~811)
    }

    # Learning rate scheduler configuration (matches ReduceLROnPlateau call in train.train_model)
    SCHEDULER = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 7,
        'verbose': True
    }

    # Image preprocessing configuration
    NORMALIZATION = {
        'mean': [0.485, 0.456, 0.406],  # ImageNet pretrained model normalization parameters
        'std': [0.229, 0.224, 0.225]
    }

    # Supported image formats (matched against extension after .lower())
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Label mapping configuration (Chinese labels stored for data compatibility)
    # 轻->0, 中->1, 重->2 (matching Excel '脱位程度' column)
    LABEL_MAPPING = {
        0: '轻',
        1: '中',
        2: '重'
    }

    # English label mapping for display
    ENGLISH_LABEL_MAPPING = {
        '轻': 'Mild',
        '中': 'Moderate',
        '重': 'Severe'
    }

    @classmethod
    def validate_config(cls):
        """Validate configuration values and raise if invalid."""
        errors = []

        # Validate proportions
        if not 0 < cls.TEST_SIZE < 1:
            errors.append("TEST_SIZE must be between 0 and 1")

        if not 0 < cls.VAL_SIZE < 1:
            errors.append("VAL_SIZE must be between 0 and 1")

        if cls.TEST_SIZE + cls.VAL_SIZE >= 1:
            errors.append("TEST_SIZE + VAL_SIZE must be less than 1")

        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be greater than 0")

        if cls.NUM_EPOCHS <= 0:
            errors.append("NUM_EPOCHS must be greater than 0")

        if cls.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be greater than 0")

        if cls.IMAGE_SIZE <= 0:
            errors.append("IMAGE_SIZE must be greater than 0")

        # Validate device
        if cls.DEVICE not in ['cuda', 'cpu']:
            errors.append("DEVICE must be 'cuda' or 'cpu'")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    @classmethod
    def print_config(cls):
        """Pretty-print the current configuration."""
        print("\n=== Current Configuration ===")
        print("Data configuration:")
        print(f"  Image size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")

        print("\nTraining configuration:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Num epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Device: {cls.DEVICE}")

        print("\nData split:")
        print(f"  Test size: {cls.TEST_SIZE}")
        print(f"  Validation size: {cls.VAL_SIZE}")
        print(f"  Random seed: {cls.RANDOM_STATE}")

        print("\nOutput paths:")
        print(f"  Model save: {cls.MODEL_SAVE_DIR}")
        print(f"  Results save: {cls.RESULTS_SAVE_DIR}")
        print("=" * 30)

# Create default configuration instance
config = Config()