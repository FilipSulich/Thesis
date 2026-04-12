import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')


class BaseCNN(nn.Module, ABC):
    """
    Abstract base class for UTRP CNN classifiers.

    Provides:
      - Training loop with early stopping
      - Evaluation with multiple metrics
      - Model save/load
      - Visualization (training curves, ROC, confusion matrix, GradCAM)
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.dropout_p = dropout
        self.device = self._get_device()

    @property
    @abstractmethod
    def input_channels(self) -> int:
        """Number of input channels (1 for single-scale, 5 for multi-scale)."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, 128, 128)

        Returns:
            Logits (NOT probabilities) of shape (batch, 1)
        """
        pass

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            # I am using MacOS for training
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            (avg_loss, accuracy)
        """
        self.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc='Training', leave=False)
        for utrps, labels, _ in pbar:
            utrps = utrps.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = self(utrps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * utrps.size(0)

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate_epoch(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            loader: Validation data loader
            criterion: Loss function

        Returns:
            (avg_loss, accuracy)
        """
        self.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for utrps, labels, _ in tqdm(loader, desc='Validation', leave=False):
                utrps = utrps.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                logits = self(utrps)
                loss = criterion(logits, labels)

                total_loss += loss.item() * utrps.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, lr: float = 1e-3, patience: int = 10,
            class_weights: Optional[torch.Tensor] = None,
            weight_decay: float = 0.05,  # Strong regularization for small dataset
            ) -> Dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            class_weights: Optional class weights for loss function

        Returns:
            History dict with keys: train_loss, train_acc, val_loss, val_acc
        """
        self.to(self.device)
        print(weight_decay)

        # Loss function with optional class weights
        if class_weights is not None:
            # Convert to float32 for MPS compatibility
            pos_weight = class_weights.float().to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Optimizer and scheduler
        # Only optimize trainable parameters (important for frozen backbones)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        # History tracking
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nTraining on {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Scheduler step
            scheduler.step(epoch)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = self.state_dict().copy()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on a dataset.

        Args:
            loader: Data loader

        Returns:
            Dict with accuracy, precision, recall, F1, AUC, confusion_matrix
        """
        self.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for utrps, labels, _ in tqdm(loader, desc='Evaluating'):
                utrps = utrps.to(self.device)

                logits = self(utrps)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

        # Add sensitivity and specificity
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def save(self, path: Path, hyperparams: Optional[Dict] = None):
        """
        Save model state and hyperparameters.

        Args:
            path: Path to save .pt file
            hyperparams: Optional hyperparameter dict to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'state_dict': self.state_dict(),
            'class_name': self.__class__.__name__,
            'hyperparams': hyperparams or {}
        }

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path, **model_kwargs):
        """
        Load model from file.

        Args:
            path: Path to .pt file
            **model_kwargs: Arguments to pass to model constructor

        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')

        # Merge saved hyperparams with provided kwargs
        hyperparams = checkpoint.get('hyperparams', {})
        hyperparams.update(model_kwargs)

        # Instantiate model
        model = cls(**hyperparams)
        model.load_state_dict(checkpoint['state_dict'])

        print(f"Model loaded from {path}")
        return model

    def plot_training_curves(self, history: Dict, save_path: Optional[Path] = None):
        """
        Plot training and validation loss/accuracy curves.

        Args:
            history: History dict from fit()
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")

        plt.close()

    def plot_roc(self, loader: DataLoader, save_path: Optional[Path] = None):
        """
        Plot ROC curve.

        Args:
            loader: Data loader
            save_path: Optional path to save figure
        """
        self.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for utrps, labels, _ in loader:
                utrps = utrps.to(self.device)
                logits = self(utrps)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)

        # Plot
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.close()

    def plot_confusion_matrix(self, loader: DataLoader, save_path: Optional[Path] = None):
        """
        Plot confusion matrix.

        Args:
            loader: Data loader
            save_path: Optional path to save figure
        """
        metrics = self.evaluate(loader)
        cm = metrics['confusion_matrix']

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['NSR', 'Pre-AF'],
                    yticklabels=['NSR', 'Pre-AF'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.close()

    def grad_cam(self, x: torch.Tensor, target_layer: nn.Module) -> np.ndarray:
        """
        Generate GradCAM heatmap for a single input.

        Args:
            x: Input tensor of shape (1, channels, H, W)
            target_layer: Target convolutional layer for GradCAM

        Returns:
            Heatmap array of shape (H, W)
        """
        self.eval()

        # Forward pass
        x = x.to(self.device).requires_grad_(True)

        # Hook for gradients and activations
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        forward_handle = target_layer.register_forward_hook(forward_hook)

        # Forward + backward
        logits = self(x)
        score = torch.sigmoid(logits)

        self.zero_grad()
        score.backward()

        # Remove hooks
        backward_handle.remove()
        forward_handle.remove()

        # Compute GradCAM
        grads = gradients[0].cpu().data.numpy()[0]  # (C, H, W)
        acts = activations[0].cpu().data.numpy()[0]  # (C, H, W)

        weights = np.mean(grads, axis=(1, 2))  # (C,)
        cam = np.sum(weights[:, None, None] * acts, axis=0)  # (H, W)

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        from scipy.ndimage import zoom
        cam_resized = zoom(cam, (x.shape[2] / cam.shape[0], x.shape[3] / cam.shape[1]))

        return cam_resized