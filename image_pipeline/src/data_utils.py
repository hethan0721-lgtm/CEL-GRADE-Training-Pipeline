import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from .config import Config

class EyeDataset(Dataset):
    """Custom Dataset class for eye disease images."""
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Ensure label is int64 (required by PyTorch CrossEntropyLoss)
        label = int(label)

        return image, label

def load_and_preprocess_data(dataset_dir, balance_classes=True, max_samples_per_class=None):
    """
    Load and preprocess data from folder structure (0/1/2 subfolders).

    Args:
        dataset_dir: Path to dataset directory containing 0/, 1/, 2/ subfolders
        balance_classes: Whether to balance classes by undersampling
        max_samples_per_class: Max samples per class. Can be:
            - None: no limit
            - int: same limit for all classes
            - dict: per-class limit, e.g. {0: None, 1: 1200, 2: None}
    """
    print("Loading data from folder structure...")

    img_paths = []
    labels = []

    # Get absolute dataset directory
    if not os.path.isabs(dataset_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, dataset_dir)

    # Label mapping: folder name -> label
    # 0=轻, 1=中, 2=重
    class_folders = ['0', '1', '2']
    class_data = {0: [], 1: [], 2: []}  # Store paths for each class

    for label, folder in enumerate(class_folders):
        folder_path = os.path.join(dataset_dir, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping")
            continue

        # Get all image files in this folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Check if it's an image file
                ext = os.path.splitext(filename)[1].lower()
                if ext in Config.SUPPORTED_EXTENSIONS:
                    class_data[label].append(file_path)

    # Print original distribution
    print("\nOriginal class distribution:")
    for label in class_data:
        label_name = Config.LABEL_MAPPING.get(label, str(label))
        print(f"  Class {label} ({label_name}): {len(class_data[label])} images")

    # Apply undersampling if enabled
    if balance_classes and max_samples_per_class is not None:
        print("\nApplying undersampling...")
        np.random.seed(Config.RANDOM_STATE)

        for label in class_data:
            # Get max samples for this class
            if isinstance(max_samples_per_class, dict):
                max_samples = max_samples_per_class.get(label, None)
            else:
                max_samples = max_samples_per_class

            # Apply undersampling if limit is set and current count exceeds it
            if max_samples is not None and len(class_data[label]) > max_samples:
                indices = np.random.choice(
                    len(class_data[label]),
                    size=max_samples,
                    replace=False
                )
                class_data[label] = [class_data[label][i] for i in indices]
                label_name = Config.LABEL_MAPPING.get(label, str(label))
                print(f"  Class {label} ({label_name}): undersampled to {len(class_data[label])} images")

    # Combine all classes
    for label in class_data:
        for path in class_data[label]:
            img_paths.append(path)
            labels.append(label)

    print(f"\nTotal images loaded: {len(img_paths)}")

    # Print final distribution
    if len(labels) > 0:
        unique, counts = np.unique(labels, return_counts=True)
        print("Final class distribution:")
        for label, count in zip(unique, counts):
            label_name = Config.LABEL_MAPPING.get(label, str(label))
            print(f"  Class {label} ({label_name}): {count} images")
    else:
        print("Warning: No valid images found!")

    # Ensure labels are Python int
    labels = [int(label) for label in labels]

    return img_paths, labels

def create_data_transforms(image_size=224):
    """Create data transformation pipelines for training and validation."""
    aug = Config.AUGMENTATION
    mean = Config.NORMALIZATION['mean']
    std = Config.NORMALIZATION['std']

    # Training augmentation - ENHANCED to improve generalization especially for rare classes
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=aug['horizontal_flip_prob']),  # Horizontal flip with 50% probability
        transforms.RandomVerticalFlip(p=aug['vertical_flip_prob']),  # Add vertical flip for more diversity
        transforms.RandomRotation(degrees=aug['rotation_degrees']),  # Increased rotation (±15 degrees)
        transforms.ColorJitter(**aug['color_jitter']),  # Enhanced color jittering
        transforms.RandomAffine(**aug['affine']),  # Add translation and scaling
        transforms.RandomResizedCrop(image_size, scale=aug['random_resized_crop']['scale']),  # Random crop for more variation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # ImageNet normalization
    ])

    # Validation/test transformation - NO augmentation, only normalization
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # ImageNet normalization
    ])

    return train_transform, val_transform

def split_data(img_paths, labels, test_size=0.2, val_size=0.2, random_state=42, apply_augmentation=True):
    """
    Split dataset into train, validation, and test sets.

    IMPORTANT: Splits FIRST on original data, THEN applies augmentation ONLY to training set.
    This prevents data leakage where identical copies appear in train/val/test splits.
    """
    # Convert to arrays
    img_paths = np.array(img_paths)
    labels = np.array(labels)

    print("\nOriginal dataset distribution (before split):")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} images")

    # STEP 1: Split original data first (NO augmentation yet)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        img_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Split train into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_size/(1-test_size),
        stratify=train_labels,
        random_state=random_state
    )

    print("\nData split (before augmentation):")
    print(f"Training set: {len(train_paths)} samples")
    print(f"Validation set: {len(val_paths)} samples")
    print(f"Test set: {len(test_paths)} samples")

    # Display class distribution for each split
    for dataset_name, dataset_labels in [("Training", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        unique_labels, counts = np.unique(dataset_labels, return_counts=True)
        print(f"{dataset_name} class distribution (original):")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} images")

    # STEP 2: Apply augmentation ONLY to training set
    if apply_augmentation:
        print("\n" + "="*60)
        print("Applying sample augmentation to TRAINING SET ONLY")
        print("(Validation and test sets remain unchanged to prevent data leakage)")
        print("="*60)
        train_paths, train_labels = augment_samples(train_paths, train_labels)

        print(f"\nFinal training set size after augmentation: {len(train_paths)} samples")
        print(f"Validation set size (unchanged): {len(val_paths)} samples")
        print(f"Test set size (unchanged): {len(test_paths)} samples")

    # Convert labels to Python int list to ensure int64 compatibility with PyTorch
    train_labels = [int(label) for label in train_labels]
    val_labels = [int(label) for label in val_labels]
    test_labels = [int(label) for label in test_labels]

    train_data = (train_paths, train_labels)
    val_data = (val_paths, val_labels)
    test_data = (test_paths, test_labels)

    return train_data, val_data, test_data

def augment_samples(img_paths, labels, augmentation_config=None):
    """
    Augment samples based on configuration to address class imbalance.

    Args:
        img_paths: List of image paths.
        labels: List of labels.
        augmentation_config: Augmentation config dict; defaults to ``Config.SAMPLE_AUGMENTATION``.

    Returns:
        augmented_img_paths: Augmented list of image paths.
        augmented_labels: Augmented list of labels.
    """
    if augmentation_config is None:
        augmentation_config = Config.SAMPLE_AUGMENTATION

    # If sample augmentation is disabled, return original data
    if not augmentation_config.get('enable', False):
        return img_paths, labels

    multipliers = augmentation_config.get('multipliers', {})

    # Convert to numpy arrays for easier manipulation
    img_paths = np.array(img_paths)
    labels = np.array(labels)

    augmented_paths = []
    augmented_labels = []

    # Show original distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nDistribution before sample augmentation:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} images")

    # Augment each class
    for label in unique_labels:
        # Get all samples from this class
        class_mask = labels == label
        class_paths = img_paths[class_mask]
        class_labels = labels[class_mask]

        # Get augmentation multiplier for this class
        multiplier = multipliers.get(label, 1)

        # Add original samples
        augmented_paths.extend(class_paths.tolist())
        augmented_labels.extend(class_labels.tolist())

        # Replicate samples if augmentation is needed
        if multiplier > 1:
            additional_copies = multiplier - 1
            for _ in range(additional_copies):
                augmented_paths.extend(class_paths.tolist())
                augmented_labels.extend(class_labels.tolist())

    # Show augmented distribution
    augmented_labels_array = np.array(augmented_labels)
    unique_labels, counts = np.unique(augmented_labels_array, return_counts=True)
    print("\nDistribution after sample augmentation:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} images")

    print(f"\nTotal samples: {len(augmented_paths)} images (original: {len(img_paths)} images)")

    # Convert labels to Python int to ensure compatibility
    augmented_labels = [int(label) for label in augmented_labels]

    return augmented_paths, augmented_labels

def collate_fn(batch):
    """Custom collate function that filters out None values."""
    # Filter None values
    batch = [item for item in batch if item[0] is not None]

    if len(batch) == 0:
        return None, None

    # Separate images and labels
    images, labels = zip(*batch)

    # Stack tensors
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

def _seed_worker(worker_id):
    """DataLoader worker_init_fn: seed NumPy and Python random per worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loaders(train_data, val_data, test_data, train_transform, val_transform,
                         batch_size=8, num_workers=2, random_state=None):
    """Create PyTorch DataLoader instances for train, validation, and test splits."""
    if random_state is None:
        random_state = Config.RANDOM_STATE

    # Unpack data
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    # Create datasets
    train_dataset = EyeDataset(train_paths, train_labels, train_transform)
    val_dataset = EyeDataset(val_paths, val_labels, val_transform)
    test_dataset = EyeDataset(test_paths, test_labels, val_transform)

    # Separate seeded generators (not one shared mutable Generator) so that
    # iterating one loader cannot change another loader's RNG state.
    train_generator = torch.Generator()
    train_generator.manual_seed(random_state)
    val_generator = torch.Generator()
    val_generator.manual_seed(random_state)
    test_generator = torch.Generator()
    test_generator.manual_seed(random_state)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=_seed_worker, generator=train_generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=_seed_worker, generator=val_generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=_seed_worker, generator=test_generator
    )

    return train_loader, val_loader, test_loader