"""
Model Evaluator - Load trained model and compute test metrics
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.CNN.dataset.dataset import UTRPDataset
from src.CNN.models.models import ZhuResNetSingleScale
from src.CNN.models.complexity_test import LateFusionWithAttention, HybridFusionWithAttention, SingleScaleCNN


class ModelEvaluator:
    """
    Evaluator for trained UTRP classification models.
    
    Loads a trained model checkpoint and evaluates it on test data
    with comprehensive metrics including AUPRC.
    """
    
    def __init__(self, model_path: Path, data_dir: Path, model_arch: str = 'single'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            data_dir: Path to dataset root (contains test.h5)
            model_arch: 'single', 'multi', or 'both'
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.model_arch = model_arch
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        
        # Will be populated
        self.model = None
        self.test_dataset = None
        self.test_loader = None
        self.predictions = {}
        
        print(f"✓ ModelEvaluator initialized")
        print(f"  Device: {self.device}")
        print(f"  Model path: {self.model_path}")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Architecture: {self.model_arch}")
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Ensure deterministic behavior on GPU/MPS
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # MPS determinism (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            os.environ['PYTHONHASHSEED'] = str(seed)

    def load_model(self, dropout: float = 0.5, strict: bool = True, attention_mode: str = 'trained') -> None:
        """
        Load trained model from checkpoint.

        Args:
            dropout:          Dropout rate used during training
            strict:           If False, ignores missing/mismatched keys
            attention_mode:   'trained'  — use learned attention weights (default)
                              'uniform'  — zero all attention layers so sigmoid
                                           outputs 0.5 for all branches (equal weighting)
                                           Gives best generalisation if attention overfit.
        """
        assert attention_mode in ('trained', 'uniform'), \
            "attention_mode must be 'trained' or 'uniform'"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"\n✓ Loading model from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            if 'model_arch' in checkpoint:
                self.model_arch = checkpoint['model_arch']
                print(f"  Detected architecture: {self.model_arch}")
            if 'epoch' in checkpoint: print(f"  Epoch:   {checkpoint['epoch']}")
            if 'val_auc' in checkpoint: print(f"  Val AUC: {checkpoint['val_auc']:.4f}")
        else:
            state_dict = checkpoint

        # Infer n_subbands from checkpoint attention layer
        n_subbands = 7  # default
        attn_key = 'attention.0.weight'
        if attn_key in state_dict:
            # attention = Linear(64 * n_subbands, n_subbands * 4)
            # output_dim = n_subbands * 4
            n_subbands = state_dict[attn_key].shape[0] // 4
            print(f"  Inferred n_subbands={n_subbands} from checkpoint")

        if self.model_arch == 'single':
            self.model = SingleScaleCNN(dropout=dropout).to(self.device)
        elif self.model_arch == 'multi':
            self.model = LateFusionWithAttention(
                n_subbands=n_subbands, dropout=dropout).to(self.device)
        else:
            self.model = HybridFusionWithAttention(
                n_subbands=n_subbands, dropout=dropout).to(self.device)

        self.model.load_state_dict(state_dict, strict=strict)
        print(f"  ✓ Weights loaded (strict={strict})")

        # Apply attention mode override
        if attention_mode == 'uniform' and hasattr(self.model, 'attention'):
            with torch.no_grad():
                for layer in self.model.attention:
                    if hasattr(layer, 'weight'):
                        layer.weight.zero_()
                    if hasattr(layer, 'bias'):
                        layer.bias.zero_()
            print(f"  ✓ Attention overridden → uniform (sigmoid=0.5 for all branches)")
        else:
            print(f"  ✓ Attention mode: {attention_mode}")

        self.model.eval()
        print(f"  Parameters: {self.model.count_params():,}")
    
    def _load_state_dict_flexible(self, state_dict: Dict) -> None:
        """
        Load state dict with flexible shape matching.
        Only loads weights that match the current model's shape.
        """
        model_state_dict = self.model.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        shape_mismatches = []
        
        for key, param in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == param.shape:
                    # Shapes match, load the weight
                    model_state_dict[key].copy_(param)
                    loaded_keys.append(key)
                else:
                    # Shape mismatch, skip
                    shape_mismatches.append((key, param.shape, model_state_dict[key].shape))
                    skipped_keys.append(key)
            else:
                # Key not in current model, skip
                skipped_keys.append(key)
        
        # Print summary
        print(f"    Loaded {len(loaded_keys)} weights")
        if skipped_keys:
            print(f"    Skipped {len(skipped_keys)} incompatible weights")
        if shape_mismatches:
            print(f"    Shape mismatches for {len(shape_mismatches)} params:")
            for key, old_shape, new_shape in shape_mismatches[:3]:  # Show first 3
                print(f"      {key}: {old_shape} → {new_shape}")
            if len(shape_mismatches) > 3:
                print(f"      ... and {len(shape_mismatches) - 3} more")

    def load_test_data(self, batch_size: int = 32) -> None:
        ds_mode = 'both' if self.model_arch == 'hybrid' else self.model_arch
        self.test_dataset = UTRPDataset(self.data_dir, split='test',
                                        mode=ds_mode, augment=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=0)
        dist = self.test_dataset.get_class_distribution()
        print(f"  Test: {len(self.test_dataset)} samples  "
              f"NSR={dist[0]}  Pre-AF={dist[1]}")

    def evaluate(self) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.test_loader is None:
            raise RuntimeError("Test data not loaded. Call load_test_data() first.")

        print(f"\n{'=' * 70}\nEVALUATING MODEL\n{'=' * 70}\n")

        all_preds, all_probs, all_labels = [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating', unit='batch'):
                if self.model_arch == 'hybrid':
                    x_single, x_multi, batch_y = batch
                    logits = self.model(x_single.to(self.device),
                                        x_multi.to(self.device))
                else:
                    batch_x, batch_y = batch
                    logits = self.model(batch_x.to(self.device))

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        self.predictions = {'preds': all_preds, 'probs': all_probs, 'labels': all_labels}
        return self._compute_metrics(all_preds, all_probs, all_labels)
    
    def _compute_metrics(self, preds: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        # ROC-AUC
        if len(np.unique(labels)) > 1:
            roc_auc = roc_auc_score(labels, probs)
        else:
            roc_auc = 0.5
        
        # AUPRC (Area Under Precision-Recall Curve)
        if len(np.unique(labels)) > 1:
            auprc = average_precision_score(labels, probs)
        else:
            auprc = 0.5
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        # Sensitivity & Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate & False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Positive & Negative Predictive Value
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Balanced Accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
        
        # Youden's Index
        youden = sensitivity + specificity - 1
        
        # Compile comprehensive metrics
        metrics = {
            # Basic Classification
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            
            # Sensitivity/Specificity
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'true_positive_rate': float(sensitivity),
            'true_negative_rate': float(specificity),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            
            # Predictive Values
            'positive_predictive_value': float(ppv),
            'negative_predictive_value': float(npv),
            
            # Discrimination Metrics
            'auc_roc': float(roc_auc),
            'auprc': float(auprc),
            
            # Additional Metrics
            'mcc': float(mcc),  # Matthews Correlation Coefficient
            'youden_index': float(youden),
            
            # Confusion Matrix
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            
            # Dataset Statistics
            'n_samples': len(labels),
            'n_positive': int(np.sum(labels)),
            'n_negative': int(len(labels) - np.sum(labels)),
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print evaluation metrics in formatted table."""
        print(f"\n{'='*70}")
        print("TEST METRICS - COMPREHENSIVE EVALUATION")
        print(f"{'='*70}\n")
        
        print("Classification Metrics:")
        print(f"  Accuracy:               {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy:      {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (PPV):        {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity):   {metrics['recall']:.4f}")
        print(f"  F1-Score:               {metrics['f1']:.4f}")
        
        print("\nSensitivity/Specificity:")
        print(f"  Sensitivity (TPR):      {metrics['sensitivity']:.4f}")
        print(f"  Specificity (TNR):      {metrics['specificity']:.4f}")
        print(f"  False Positive Rate:    {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate:    {metrics['false_negative_rate']:.4f}")
        
        print("\nPredictive Values:")
        print(f"  Positive Pred Value:    {metrics['positive_predictive_value']:.4f}")
        print(f"  Negative Pred Value:    {metrics['negative_predictive_value']:.4f}")
        
        print("\nDiscrimination Metrics:")
        print(f"  AUC-ROC:                {metrics['auc_roc']:.4f}")
        print(f"  AUPRC:                  {metrics['auprc']:.4f}")
        
        print("\nAdditional Metrics:")
        print(f"  Matthews Corr Coeff:    {metrics['mcc']:.4f}")
        print(f"  Youden's Index:         {metrics['youden_index']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  True Negatives (TN):    {metrics['tn']}")
        print(f"  False Positives (FP):   {metrics['fp']}")
        print(f"  False Negatives (FN):   {metrics['fn']}")
        print(f"  True Positives (TP):    {metrics['tp']}")
        
        print("\nDataset Statistics:")
        print(f"  Total samples:          {metrics['n_samples']}")
        print(f"  Positive (Pre-AF):      {metrics['n_positive']} ({100*metrics['n_positive']/metrics['n_samples']:.1f}%)")
        print(f"  Negative (NSR):         {metrics['n_negative']} ({100*metrics['n_negative']/metrics['n_samples']:.1f}%)")
        
        print(f"\n{'='*70}\n")
    
    def save_metrics_json(self, metrics: Dict[str, float], save_path: Path) -> None:
        """Save metrics to JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'model_arch': self.model_arch,
            'metrics': metrics
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved metrics to {save_path}")
    
    def get_predictions_with_confidence(self) -> np.ndarray:
        """
        Get predictions with confidence scores.
        
        Returns:
            Array of shape (n_samples, 3) with [label, prediction, confidence]
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Call evaluate() first.")
        
        probs = self.predictions['probs']
        preds = self.predictions['preds']
        labels = self.predictions['labels']
        
        # Confidence is the probability of predicted class
        confidence = np.where(preds == 1, probs, 1 - probs)
        
        return np.column_stack([labels, preds, confidence])
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve points.
        
        Returns:
            (fpr, tpr, thresholds)
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Call evaluate() first.")
        
        fpr, tpr, thresholds = roc_curve(self.predictions['labels'], self.predictions['probs'])
        return fpr, tpr, thresholds
    
    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve points.
        
        Returns:
            (precision, recall, thresholds)
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Call evaluate() first.")
        
        precision, recall, thresholds = precision_recall_curve(
            self.predictions['labels'],
            self.predictions['probs']
        )
        return precision, recall, thresholds
    
    @staticmethod
    def plot_comparison(evaluators_dict: Dict[str, 'ModelEvaluator'], save_path: Path = None) -> None:
        """
        Plot AUC and AUPRC comparison for multiple models on the same figure.
        
        Args:
            evaluators_dict: Dictionary with model names as keys and ModelEvaluator instances as values
                            Example: {'Single': evaluator_single, 'Multi': evaluator_multi}
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Define colors for different models
        colors = {
            'single': '#1f77b4',  # blue
            'multi': '#ff7f0e',   # orange
            'both': '#2ca02c',    # green
            'Single': '#1f77b4',
            'Multi Uniform': '#1f77b4',
            'Multi Trained': '#2ca02c',
            'Multi': '#ff7f0e',
            'Both': '#2ca02c',
        }
        
        # Plot ROC curves (AUC)
        for name, evaluator in evaluators_dict.items():
            fpr, tpr, _ = evaluator.get_roc_curve()
            auc_score = roc_auc_score(evaluator.predictions['labels'], evaluator.predictions['probs'])
            color = colors.get(name, '#1f77b4')
            ax1.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc_score:.4f})', color=color)
        
        # Diagonal reference line for ROC
        ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax1.set_title('ROC Curves - AUC Comparison', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Plot Precision-Recall curves (AUPRC)
        for name, evaluator in evaluators_dict.items():
            precision, recall, _ = evaluator.get_pr_curve()
            auprc_score = average_precision_score(evaluator.predictions['labels'], evaluator.predictions['probs'])
            color = colors.get(name, '#1f77b4')
            ax2.plot(recall, precision, lw=2.5, label=f'{name} (AUPRC={auprc_score:.4f})', color=color)
        
        # Baseline for precision-recall (random classifier)
        if evaluators_dict:
            first_evaluator = next(iter(evaluators_dict.values()))
            pos_ratio = np.sum(first_evaluator.predictions['labels']) / len(first_evaluator.predictions['labels'])
            ax2.axhline(y=pos_ratio, color='k', linestyle='--', lw=1.5, label='Random Classifier')
        
        ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax2.set_title('Precision-Recall Curves - AUPRC Comparison', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {save_path}")
        
        plt.show()

def evaluate_model(
    model_path: Path,
    data_dir: Path,
    model_arch: str = None,
    batch_size: int = 32,
    save_metrics_path: Path = None,
    seed: int = 42,
    attention_mode: str = 'trained',   # ← new
) -> Dict[str, float]:
    ModelEvaluator.set_seed(seed)

    if model_arch is None:
        checkpoint = torch.load(model_path, map_location='cpu')
        model_arch = (checkpoint.get('model_arch', 'single')
                      if isinstance(checkpoint, dict) else 'single')
        print(f"✓ Auto-detected architecture: {model_arch}")

    evaluator = ModelEvaluator(model_path, data_dir, model_arch)
    evaluator.load_model(strict=True, attention_mode=attention_mode)
    evaluator.load_test_data(batch_size=batch_size)
    metrics = evaluator.evaluate()
    evaluator.print_metrics(metrics)

    if save_metrics_path:
        evaluator.save_metrics_json(metrics, save_metrics_path)

    return metrics


if __name__ == '__main__':
    model_path = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_new_single/single/final_checkpoint.pt')
    # model_path = Path('/results_new/multi/final_checkpoint.pt')
    data_dir = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_final/utrp_dataset_db8_no_denoise')
    
    metrics = evaluate_model(
        model_path,
        data_dir,
        batch_size=32,
        save_metrics_path=Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_new/metrics/test_metrics_single_new.json'),
        attention_mode='uniform'
    )
    
    print("\n✓ Evaluation complete!")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
