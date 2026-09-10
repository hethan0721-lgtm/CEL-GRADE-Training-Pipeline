#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module responsible for model assessment, metric computation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so plotting works headless (CI, servers without a display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
import os

from .config import Config

# Configure font for Chinese characters (optional; can be disabled if no Chinese needed in plots)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """Model evaluation coordinator for computing metrics and generating visualizations."""

    def __init__(self, config=None):
        """
        Initialize the evaluator.

        Args:
            config: Optional configuration object.
        """
        self.config = config if config is not None else Config()
        self.results = {}

    def evaluate(self, model, X_test, y_test, verbose=True):
        """
        Evaluate the model on the test dataset and compute comprehensive metrics.

        Args:
            model: Trained classifier instance.
            X_test: Test features.
            y_test: Test labels.
            verbose: Whether to print detailed results.

        Returns:
            dict: Dictionary containing all evaluation metrics and predictions.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" " * 26 + "Model Evaluation")
            print(f"{'='*70}")
            print(f"\nTest set size: {len(y_test)} samples")

        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Positive class probability

        # Compute metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=[self.config.LABEL_NAMES[0], self.config.LABEL_NAMES[1]],
                output_dict=True
            ),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        self.results = results

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results):
        """Print evaluation results in a human-readable format."""
        print(f"\n{'='*70}")
        print(f" " * 25 + "Evaluation Results")
        print(f"{'='*70}")

        print("\n[Overall Metrics]")
        print(f"  Accuracy:           {results['accuracy']:.4f}")
        print(f"  Precision:          {results['precision']:.4f}")
        print(f"  Recall:             {results['recall']:.4f}")
        print(f"  F1-Score:           {results['f1']:.4f}")
        print(f"  ROC-AUC:            {results['roc_auc']:.4f}")
        print(f"  Average Precision:  {results['average_precision']:.4f}")

        print("\n[Confusion Matrix]")
        cm = results['confusion_matrix']
        print(f"                 Predicted:{self.config.LABEL_NAMES[0]:^15s}  Predicted:{self.config.LABEL_NAMES[1]:^15s}")
        print(f"  Actual:{self.config.LABEL_NAMES[0]:<15s}  {cm[0,0]:6d}                {cm[0,1]:6d}")
        print(f"  Actual:{self.config.LABEL_NAMES[1]:<15s}  {cm[1,0]:6d}                {cm[1,1]:6d}")

        print("\n[Classification Report]")
        report = results['classification_report']
        for label_name in [self.config.LABEL_NAMES[0], self.config.LABEL_NAMES[1]]:
            if label_name in report:
                metrics = report[label_name]
                print(f"  {label_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-Score:  {metrics['f1-score']:.4f}")
                print(f"    Support:   {metrics['support']}")

        print(f"{'='*70}\n")

    def plot_confusion_matrix(self, save_dir=None, show=False):
        """
        Plot the confusion matrix as a heatmap.

        Args:
            save_dir: Optional directory to save the figure.
            show: Whether to display the figure interactively.
        """
        if not self.results:
            raise ValueError("Call evaluate() before plotting the confusion matrix.")

        cm = self.results['confusion_matrix']

        plt.figure(figsize=(10, 8))

        # Create heatmap with English labels
        ax = sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[self.config.LABEL_NAMES_EN[0], self.config.LABEL_NAMES_EN[1]],
            yticklabels=[self.config.LABEL_NAMES_EN[0], self.config.LABEL_NAMES_EN[1]],
            cbar_kws={'label': 'Sample Count'},
            annot_kws={'size': 28, 'weight': 'bold'}
        )

        plt.title('Confusion Matrix', fontsize=28, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=24, fontweight='bold')
        plt.ylabel('True Label', fontsize=24, fontweight='bold')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Sample Count', fontsize=22)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_roc_curve(self, save_dir=None, show=False):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.

        Args:
            save_dir: Optional directory to save the figure.
            show: Whether to display the figure interactively.
        """
        if not self.results:
            raise ValueError("Call evaluate() before plotting the ROC curve.")

        y_test = self.results['y_test']
        y_proba = self.results['y_proba']

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = self.results['roc_auc']

        plt.figure(figsize=(10, 8))

        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.4f})')

        # Plot diagonal (random guess baseline)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Guess (AUC = 0.5000)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=24, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=24, fontweight='bold')
        plt.title('ROC Curve', fontsize=28, fontweight='bold')
        plt.legend(loc="lower right", fontsize=20)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'roc_curve.png')
            plt.savefig(save_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ ROC curve saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_precision_recall_curve(self, save_dir=None, show=False):
        """
        Plot the Precision-Recall curve.

        Args:
            save_dir: Optional directory to save the figure.
            show: Whether to display the figure interactively.
        """
        if not self.results:
            raise ValueError("Call evaluate() before plotting the PR curve.")

        y_test = self.results['y_test']
        y_proba = self.results['y_proba']

        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        avg_precision = self.results['average_precision']

        plt.figure(figsize=(10, 8))

        # Plot PR curve
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.4f})')

        # Plot baseline (random guess)
        baseline = y_test.sum() / len(y_test)
        plt.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--',
                label=f'Random Guess (AP = {baseline:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=24, fontweight='bold')
        plt.ylabel('Precision', fontsize=24, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=28, fontweight='bold')
        plt.legend(loc="lower left", fontsize=18)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'precision_recall_curve.png')
            plt.savefig(save_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ PR curve saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_importance(self, model, top_n=15, save_dir=None, show=False):
        """
        Plot feature importance scores as a horizontal bar chart.

        Args:
            model: Trained model with feature importance.
            top_n: Number of top features to display.
            save_dir: Optional directory to save the figure.
            show: Whether to display the figure interactively.
        """
        feature_importance = model.get_feature_importance(top_n=top_n)

        if feature_importance is None:
            print("⚠ Model does not support feature importance")
            return

        # Convert to DataFrame for plotting with English feature names
        feature_names_en = [
            self.config.FEATURE_NAME_MAPPING.get(f, f) for f in feature_importance.keys()
        ]
        df = pd.DataFrame({
            'Feature': feature_names_en,
            'Importance': list(feature_importance.values())
        })

        plt.figure(figsize=(12, max(8, len(df) * 0.5)))

        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        bars = plt.barh(df['Feature'], df['Importance'], color=colors)

        plt.xlabel('Importance Score', fontsize=22, fontweight='bold')
        plt.ylabel('Feature', fontsize=24, fontweight='bold')
        plt.title(f'Feature Importance (Top {len(df)})', fontsize=28, fontweight='bold')
        plt.gca().invert_yaxis()  # Most important feature on top
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    ha='left', va='center', fontsize=16, fontweight='bold')

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.config.FEATURE_IMPORTANCE_NAME)
            plt.savefig(save_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ Feature importance plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_metrics_comparison(self, save_dir=None, show=False):
        """
        Plot a bar chart comparing all computed metrics.

        Args:
            save_dir: Optional directory to save the figure.
            show: Whether to display the figure interactively.
        """
        if not self.results:
            raise ValueError("Call evaluate() before plotting metrics comparison.")

        metrics = {
            'Accuracy': self.results['accuracy'],
            'Precision': self.results['precision'],
            'Recall': self.results['recall'],
            'F1-Score': self.results['f1'],
            'ROC-AUC': self.results['roc_auc']
        }

        plt.figure(figsize=(12, 7), dpi=100)

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars = plt.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        plt.ylabel('Score', fontsize=25, fontweight='bold')
        plt.title('Model Performance Metrics Comparison', fontsize=28, fontweight='bold', pad=20)
        plt.ylim([0, 1.1])
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'metrics_comparison.png')
            plt.savefig(save_path, dpi=self.config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ Metrics comparison plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_all_plots(self, model, save_dir=None, show=False):
        """
        Generate all evaluation plots (confusion matrix, ROC, PR, feature importance, metrics).

        Args:
            model: Trained model instance.
            save_dir: Optional directory to save figures.
            show: Whether to display figures interactively.
        """
        if save_dir is None:
            paths = self.config.get_absolute_paths()
            save_dir = paths['results_save_dir']

        print(f"\n{'='*70}")
        print(f" " * 24 + "Generating Evaluation Plots")
        print(f"{'='*70}\n")

        self.plot_confusion_matrix(save_dir=save_dir, show=show)
        self.plot_roc_curve(save_dir=save_dir, show=show)
        self.plot_precision_recall_curve(save_dir=save_dir, show=show)
        self.plot_feature_importance(model, top_n=15, save_dir=save_dir, show=show)
        self.plot_metrics_comparison(save_dir=save_dir, show=show)

        print(f"\n✓ All plots generated and saved to: {save_dir}")
        print(f"{'='*70}\n")

    def save_results(self, save_dir=None):
        """
        Save evaluation results to a text file.

        Args:
            save_dir: Optional directory to save the report.
        """
        if not self.results:
            raise ValueError("Call evaluate() before saving results.")

        if save_dir is None:
            paths = self.config.get_absolute_paths()
            save_dir = paths['results_save_dir']

        os.makedirs(save_dir, exist_ok=True)

        # Save classification report
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" " * 25 + "Classification Report\n")
            f.write("="*70 + "\n\n")

            f.write("[Overall Metrics]\n")
            f.write(f"  Accuracy:           {self.results['accuracy']:.4f}\n")
            f.write(f"  Precision:          {self.results['precision']:.4f}\n")
            f.write(f"  Recall:             {self.results['recall']:.4f}\n")
            f.write(f"  F1-Score:           {self.results['f1']:.4f}\n")
            f.write(f"  ROC-AUC:            {self.results['roc_auc']:.4f}\n")
            f.write(f"  Average Precision:  {self.results['average_precision']:.4f}\n\n")

            f.write("[Confusion Matrix]\n")
            cm = self.results['confusion_matrix']
            f.write(f"                 Predicted:{self.config.LABEL_NAMES[0]:^15s}  Predicted:{self.config.LABEL_NAMES[1]:^15s}\n")
            f.write(f"  Actual:{self.config.LABEL_NAMES[0]:<15s}  {cm[0,0]:6d}                {cm[0,1]:6d}\n")
            f.write(f"  Actual:{self.config.LABEL_NAMES[1]:<15s}  {cm[1,0]:6d}                {cm[1,1]:6d}\n\n")

            f.write("[Detailed Classification Report]\n")
            report = self.results['classification_report']
            for label_name in [self.config.LABEL_NAMES[0], self.config.LABEL_NAMES[1]]:
                if label_name in report:
                    metrics = report[label_name]
                    f.write(f"  {label_name}:\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall:    {metrics['recall']:.4f}\n")
                    f.write(f"    F1-Score:  {metrics['f1-score']:.4f}\n")
                    f.write(f"    Support:   {metrics['support']}\n\n")

        print(f"✓ Evaluation report saved to: {report_path}")

        return report_path


def evaluate_model(model, X_test, y_test, save_results=True,
                   generate_plots=True, show_plots=False, verbose=True):
    """
    Convenience helper to evaluate a model on test data.

    Args:
        model: Trained classifier instance.
        X_test: Test features.
        y_test: Test labels.
        save_results: Whether to persist the report to disk.
        generate_plots: Whether to generate plots.
        show_plots: Whether to display plots interactively.
        verbose: Whether to print detailed results.

    Returns:
        results: Dictionary containing all evaluation metrics.
        evaluator: ``ModelEvaluator`` instance for additional operations.
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(model, X_test, y_test, verbose=verbose)

    if generate_plots:
        evaluator.generate_all_plots(model, show=show_plots)

    if save_results:
        evaluator.save_results()

    return results, evaluator
