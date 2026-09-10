import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50Classifier(nn.Module):
    """ResNet50-based image classifier for eye disease severity."""
    def __init__(self, num_classes, input_size=224, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        self.input_size = input_size

        # Load ResNet50 model. IMAGENET1K_V1 is used (not DEFAULT) to match
        # the historical pretrained=True weights this pipeline was trained
        # with; DEFAULT may resolve to a newer IMAGENET1K_V2 checkpoint.
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Extract ResNet50 feature layers (remove the final fully connected layer)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        # Optimize pooling layer for the given input size
        if input_size == 224:
            # Standard 7x7 pooling for ResNet50's default configuration
            pool_size = (7, 7)
            linear_input = 2048 * 7 * 7
        elif input_size <= 512:
            pool_size = (4, 4)  # Smaller pooling for smaller input
            linear_input = 2048 * 4 * 4
        else:
            pool_size = (7, 7)  # Larger images use original pooling
            linear_input = 2048 * 7 * 7

        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_size)

        # Classification head with moderate dropout (balanced for imbalanced dataset)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),  # Reduced dropout to allow more feature learning for rare classes
            nn.Linear(linear_input, 512),  # Increased capacity for better feature extraction
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Classification
        x = self.classifier(x)

        return x

    def get_model_info(self):
        """Return model metadata including parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_size': self.input_size
        }

def create_model(num_classes, input_size=224, pretrained=True):
    """Factory function to instantiate a ResNet50 classifier."""
    model = ResNet50Classifier(num_classes=num_classes, input_size=input_size, pretrained=pretrained)
    return model

def print_model_info(model):
    """Print a human-readable summary of the model."""
    info = model.get_model_info()
    print("\nModel information:")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Input image size: {info['input_size']}x{info['input_size']}")
    print("Model architecture: ResNet50 + custom classification head")