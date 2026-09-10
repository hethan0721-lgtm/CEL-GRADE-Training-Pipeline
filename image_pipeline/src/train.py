import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time

from .config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples and down-weights easy examples.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.label_smoothing = label_smoothing  # Label smoothing factor
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                   label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir='models', use_amp=True):
    """Train the model with support for mixed precision and early stopping."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Define loss function - Using Focal Loss for extreme class imbalance
    # Focal Loss focuses on hard examples and down-weights easy examples
    class_weights = torch.FloatTensor([1.0, 1.2, 10.0]).to(device)  # [Mild, Moderate, Severe]
    print(f"Using Focal Loss with class weights: Mild=1.0, Moderate=1.2, Severe=10.0")
    print(f"Focal Loss gamma=2.5 (higher gamma = more focus on hard examples)")
    print(f"Label smoothing=0.05 (to prevent overconfidence)")

    criterion = FocalLoss(alpha=class_weights, gamma=2.5, label_smoothing=0.05)

    # Mixed precision training support
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    # Move model to device
    model = model.to(device)

    # Freeze the initial ResNet convolution (conv1); fine-tune the remaining
    # backbone and classifier.
    print("\nFreezing the initial ResNet convolution (conv1); fine-tuning the remaining backbone and classifier.")
    if hasattr(model, 'features'):
        for param in model.features[0].parameters():
            param.requires_grad = False

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Create optimizer AFTER freezing layers (only optimize trainable parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER['mode'],
        factor=Config.SCHEDULER['factor'],
        patience=Config.SCHEDULER['patience'],
        verbose=Config.SCHEDULER['verbose']
    )

    # Track training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')

    # Early stopping mechanism - increased patience for better convergence
    early_stopping_patience = 15
    early_stopping_counter = 0
    best_val_loss = float('inf')

    print(f"\nStarting training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Record history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_history': train_history
            }, best_model_path)
            print(f"✓ Saved best model (validation accuracy: {val_acc:.4f})")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] - {epoch_time:.1f}s")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Early stop counter: {early_stopping_counter}/{early_stopping_patience}")
        print("-" * 60)

        # Trigger early stopping if patience is exceeded
        if early_stopping_counter >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered! Val loss did not improve for {early_stopping_patience} consecutive epochs")
            break

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")

    return train_history, best_model_path


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc='Training')

    for batch_idx, (data, target) in enumerate(train_bar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Update progress bar
        train_bar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='Validating')

        for batch_idx, (data, target) in enumerate(val_bar):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            val_bar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def load_model(model, model_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ensure model is on the correct device
    model = model.to(device)

    print(f"Model loaded successfully: {model_path}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"Training epoch: {checkpoint['epoch'] + 1}")
    print(f"Model moved to device: {device}")

    return model, checkpoint
