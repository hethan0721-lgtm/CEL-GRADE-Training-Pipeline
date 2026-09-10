import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import os
from .config import Config


def evaluate_model(model, test_loader, label_mapping, device, save_dir='results'):
    """Evaluate model performance on the test set."""
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_predictions = []
    all_labels = []

    print("\nStarting model evaluation...")

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Evaluating')

        for data, target in test_bar:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    # Generate classification report
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True
    )

    # Print results
    print(f"\nTest set accuracy: {accuracy:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Save classification report
    report_text = classification_report(all_labels, all_predictions, target_names=class_names)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Test set accuracy: {accuracy:.4f}\n\n")
        f.write("Detailed classification report:\n")
        f.write(report_text)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_dir)

    # Plot per-class performance
    plot_class_performance(report, class_names, save_dir)

    return accuracy, report, all_predictions, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    # Use English class labels from config
    display_class_names = [Config.ENGLISH_LABEL_MAPPING.get(name, name) for name in class_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_class_names, yticklabels=display_class_names,
                annot_kws={'size': 18})

    plt.title('Confusion Matrix', fontsize=26, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Confusion matrix saved to: {os.path.join(save_dir, 'confusion_matrix.png')}")

def plot_class_performance(report, class_names, save_dir):
    """Plot performance metrics by class"""
    # Extract precision, recall and F1-score for each class
    precision = [report[class_name]['precision'] for class_name in class_names]
    recall = [report[class_name]['recall'] for class_name in class_names]
    f1_score = [report[class_name]['f1-score'] for class_name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    # Use English class labels from config
    display_class_names = [Config.ENGLISH_LABEL_MAPPING.get(name, name) for name in class_names]

    plt.figure(figsize=(12, 8))

    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.title('Performance Metrics by Class', fontsize=26, fontweight='bold')

    plt.xticks(x, display_class_names, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.ylim(0, 1.1)

    # Add values on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1_score)):
        plt.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=14)
        plt.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=14)
        plt.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(save_dir, 'class_performance.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Class performance plot saved to: {os.path.join(save_dir, 'class_performance.png')}")

def plot_training_history(train_history, save_dir='results'):
    """Plot training history"""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_history['train_loss']) + 1)

    plt.figure(figsize=(15, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, train_history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss', fontsize=24, fontweight='bold')
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, train_history['val_acc'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=24, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Training history plot saved to: {os.path.join(save_dir, 'training_history.png')}")

def predict_single_image(model, image_path, transform, label_mapping, device):
    """Predict the class of a single image."""
    from PIL import Image

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Decode label
    predicted_label = label_mapping[predicted_class]

    return predicted_label, confidence, probabilities[0].cpu().numpy()