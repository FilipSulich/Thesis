import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.CNN.dataset.dataset import UTRPDataset
from src.CNN.models.models import MultiScaleCNN, HybridCNN, SingleScaleCNN


class ModelEvaluator:

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

        # Infer n_subbands from checkpoint
        # First, check if explicitly stored in checkpoint
        if isinstance(checkpoint, dict) and 'n_subbands' in checkpoint:
            n_subbands = checkpoint['n_subbands']
            print(f"  Using n_subbands={n_subbands} from checkpoint metadata")
        else:
            # Infer n_subbands from checkpoint attention layer
            n_subbands = 7  # default
            attn_key = 'attention.0.weight'
            if attn_key in state_dict:
                # attention = Linear(64 * n_branches, n_branches * 4)
                # For LateFusionWithAttention: n_branches = n_subbands
                # For HybridFusionWithAttention: n_branches = 1 + n_subbands
                output_dim = state_dict[attn_key].shape[0]
                
                if self.model_arch == 'hybrid':
                    # n_branches = 1 + n_subbands, so output = (1 + n_subbands) * 4
                    n_branches = output_dim // 4
                    n_subbands = n_branches - 1
                else:  # 'multi'
                    # n_branches = n_subbands, so output = n_subbands * 4
                    n_subbands = output_dim // 4
                
                print(f"  Inferred n_subbands={n_subbands} from checkpoint (arch={self.model_arch})")

        if self.model_arch == 'single':
            # self.model = SingleScalePaperCNN(dropout=dropout).to(self.device)
            self.model = SingleScaleCNN(dropout=dropout).to(self.device)
        elif self.model_arch == 'multi':
            self.model = MultiScaleCNN(
                n_subbands=n_subbands, dropout=dropout).to(self.device)
        else:
            self.model = HybridCNN(
                n_subbands=n_subbands, dropout=dropout).to(self.device)

        try:
            self.model.load_state_dict(state_dict, strict=strict)
            print(f"  ✓ Weights loaded (strict={strict})")
        except RuntimeError as e:
            if strict:
                print(f"  ⚠ Strict loading failed: {str(e)[:100]}...")
                print(f"  Retrying with strict=False...")
                self.model.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Weights loaded (strict=False)")
            else:
                raise

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
        self.test_dataset = UTRPDataset(self.data_dir, split='test',mode=ds_mode, augment=False)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        dist = self.test_dataset.get_class_distribution()

        print(f"  Test: {len(self.test_dataset)} samples  "
              f"NSR={dist[0]}  Pre-AF={dist[1]}")

    def evaluate(self) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.test_loader is None:
            raise RuntimeError("Test data_old not loaded. Call load_test_data() first.")

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

        # self.model.get_attention_weights(all_preds)

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

    def evaluate_patient_level(self) -> Dict[str, any]:
        """
        Evaluate patient-level performance by aggregating segment predictions.

        Groups predictions by patient and aggregates using:
        - Majority vote: most common prediction across segments
        - Mean probability: average probability across segments

        Returns:
            Dictionary with patient-level metrics and detailed analysis
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Call evaluate() first.")
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not loaded.")

        print(f"\n{'='*70}")
        print("PATIENT-LEVEL PERFORMANCE ANALYSIS")
        print(f"{'='*70}\n")

        # Collect predictions with metadata
        patient_data = {}  # {patient_id: {'segments': [], 'probs': [], 'preds': [], 'label': int}}

        for idx in range(len(self.test_dataset)):
            metadata = self.test_dataset.get_metadata(idx)
            patient_id = metadata['record_id']
            segment_idx = metadata['segment_index']
            true_label = metadata['label']

            pred = self.predictions['preds'][idx]
            prob = self.predictions['probs'][idx]

            if patient_id not in patient_data:
                patient_data[patient_id] = {
                    'segments': [],
                    'probs': [],
                    'preds': [],
                    'label': true_label
                }

            patient_data[patient_id]['segments'].append(segment_idx)
            patient_data[patient_id]['probs'].append(prob)
            patient_data[patient_id]['preds'].append(pred)

        # Aggregate predictions per patient
        patient_results = {}
        for patient_id, data in patient_data.items():
            # Majority vote
            pred_counts = np.bincount(data['preds'])
            majority_pred = np.argmax(pred_counts)

            # Mean probability
            mean_prob = np.mean(data['probs'])
            mean_prob_pred = 1 if mean_prob >= 0.5 else 0

            # Count misclassified segments
            segment_errors = [i for i, (seg_idx, pred) in enumerate(zip(data['segments'], data['preds']))
                            if pred != data['label']]

            patient_results[patient_id] = {
                'true_label': data['label'],
                'n_segments': len(data['segments']),
                'segment_indices': data['segments'],
                'segment_probs': data['probs'],
                'segment_preds': data['preds'],
                'majority_vote_pred': majority_pred,
                'mean_prob': mean_prob,
                'mean_prob_pred': mean_prob_pred,
                'n_errors': len(segment_errors),
                'error_segments': [data['segments'][i] for i in segment_errors],
            }

        # Calculate patient-level metrics
        true_labels_majority = []
        pred_labels_majority = []
        true_labels_mean = []
        pred_labels_mean = []
        mean_probs = []

        for patient_id, result in patient_results.items():
            true_labels_majority.append(result['true_label'])
            pred_labels_majority.append(result['majority_vote_pred'])
            true_labels_mean.append(result['true_label'])
            pred_labels_mean.append(result['mean_prob_pred'])
            mean_probs.append(result['mean_prob'])

        true_labels_majority = np.array(true_labels_majority)
        pred_labels_majority = np.array(pred_labels_majority)
        true_labels_mean = np.array(true_labels_mean)
        pred_labels_mean = np.array(pred_labels_mean)
        mean_probs = np.array(mean_probs)
        mean_probs_mean = np.mean(mean_probs)

        # Compute metrics for both aggregation methods
        metrics_majority = self._compute_metrics(pred_labels_majority, mean_probs, true_labels_majority)
        metrics_mean = self._compute_metrics(pred_labels_mean, mean_probs, true_labels_mean)

        # Print summary
        print("Aggregation Method: MAJORITY VOTE")
        print(f"  Patient-level Accuracy:  {metrics_majority['accuracy']:.4f}")
        print(f"  Patients classified:     {len(patient_results)}")
        print(f"  Correct:                 {metrics_majority['tp'] + metrics_majority['tn']}")
        print(f"  Incorrect:               {metrics_majority['fp'] + metrics_majority['fn']}")
        print(f"  Sensitivity:             {metrics_majority['sensitivity']:.4f}")
        print(f"  Specificity:             {metrics_majority['specificity']:.4f}")
        print(f"  AUC-ROC:                 {metrics_majority['auc_roc']:.4f}")
        print(f"Mean pred. probability:   {mean_probs_mean:.2f}")

        print("\nAggregation Method: MEAN PROBABILITY")
        print(f"  Patient-level Accuracy:  {metrics_mean['accuracy']:.4f}")
        print(f"  Correct:                 {metrics_mean['tp'] + metrics_mean['tn']}")
        print(f"  Incorrect:               {metrics_mean['fp'] + metrics_mean['fn']}")
        print(f"  Sensitivity:             {metrics_mean['sensitivity']:.4f}")
        print(f"  Specificity:             {metrics_mean['specificity']:.4f}")
        print(f"  AUC-ROC:                 {metrics_mean['auc_roc']:.4f}")

        # Analyze misclassified patients
        print(f"\n{'─'*70}")
        print("MISCLASSIFIED PATIENTS (Majority Vote):")
        print(f"{'─'*70}")

        misclassified_patients = [
            (pid, result) for pid, result in patient_results.items()
            if result['majority_vote_pred'] != result['true_label']
        ]

        if misclassified_patients:
            for patient_id, result in sorted(misclassified_patients):
                true_class = "Pre-AF" if result['true_label'] == 1 else "NSR"
                pred_class = "Pre-AF" if result['majority_vote_pred'] == 1 else "NSR"
                print(f"\nPatient: {patient_id}")
                print(f"  True: {true_class}, Predicted: {pred_class}")
                print(f"  Segments with errors: {result['n_errors']}/{result['n_segments']}")
                print(f"  Error segment indices: {result['error_segments']}")
                print(f"  Mean probability: {result['mean_prob']:.4f}")
        else:
            print("  None - all patients correctly classified!")

        # Find patients with most segment errors
        print(f"\n{'─'*70}")
        print("PATIENTS WITH MOST SEGMENT ERRORS:")
        print(f"{'─'*70}")

        sorted_by_errors = sorted(patient_results.items(),
                                 key=lambda x: x[1]['n_errors'],
                                 reverse=True)

        print("\nTop 10 patients with most misclassified segments:")
        for i, (patient_id, result) in enumerate(sorted_by_errors[:10], 1):
            if result['n_errors'] == 0:
                break
            true_class = "Pre-AF" if result['true_label'] == 1 else "NSR"
            print(f"{i:2d}. {patient_id}: {result['n_errors']}/{result['n_segments']} errors "
                  f"(True: {true_class})")

        # Segment-level error distribution
        print(f"\n{'─'*70}")
        print("SEGMENT ERROR DISTRIBUTION:")
        print(f"{'─'*70}")

        segment_error_counter = {}
        for patient_id, result in patient_results.items():
            for seg_idx in result['error_segments']:
                segment_error_counter[seg_idx] = segment_error_counter.get(seg_idx, 0) + 1

        if segment_error_counter:
            print("\nSegment indices with most errors across all patients:")
            sorted_segments = sorted(segment_error_counter.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
            for seg_idx, count in sorted_segments[:5]:
                print(f"  Segment {seg_idx}: {count} patients had errors")
        else:
            print("  No segment errors detected!")

        print(f"\n{'='*70}\n")

        return {
            'patient_results': patient_results,
            'metrics_majority': metrics_majority,
            'metrics_mean': metrics_mean,
            'n_patients': len(patient_results),
            'misclassified_patients': misclassified_patients,
        }

    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        save_path: Path = None,
        title: str = None,
        cmap: str = 'Blues',
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrix for the model.

        Args:
            normalize: If True, normalize the confusion matrix
            save_path: Optional path to save the figure
            title: Custom title (default: auto-generated)
            cmap: Colormap for the heatmap (default: 'Blues')
            show: Whether to display the figure

        Returns:
            matplotlib Figure
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Call evaluate() first.")

        labels = self.predictions['labels']
        preds = self.predictions['preds']

        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
            ax=ax,
            vmin=0,
            vmax=1 if normalize else None,
        )

        # Labels
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['NSR', 'Pre-AF'], fontsize=11)
        ax.set_yticklabels(['NSR', 'Pre-AF'], fontsize=11, rotation=0)

        # Title
        if title is None:
            title = f'Confusion Matrix - {self.model_arch.capitalize()}'
            if normalize:
                title += ' (Normalized)'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    @staticmethod
    def plot_confusion_matrices_comparison(
        evaluators_dict: Dict[str, 'ModelEvaluator'],
        normalize: bool = False,
        save_path: Path = None,
        title: str = "Confusion Matrices Comparison",
        cmap: str = 'Blues',
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrices for multiple models side by side.

        Args:
            evaluators_dict: Dictionary with model names as keys and ModelEvaluator instances as values
            normalize: If True, normalize the confusion matrices
            save_path: Optional path to save the figure
            title: Figure title
            cmap: Colormap for the heatmaps
            show: Whether to display the figure

        Returns:
            matplotlib Figure
        """
        n_models = len(evaluators_dict)

        if n_models == 0:
            raise ValueError("evaluators_dict cannot be empty")

        # Create subplots
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))

        if n_models == 1:
            axes = [axes]

        for ax, (name, evaluator) in zip(axes, evaluators_dict.items()):
            if not evaluator.predictions:
                raise RuntimeError(f"No predictions for {name}. Call evaluate() first.")

            labels = evaluator.predictions['labels']
            preds = evaluator.predictions['preds']

            # Compute confusion matrix
            cm = confusion_matrix(labels, preds)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap=cmap,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Normalized' if normalize else 'Count'},
                ax=ax,
                vmin=0,
                vmax=1 if normalize else None,
            )

            # Labels
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_xticklabels(['NSR', 'Pre-AF'], fontsize=10)
            ax.set_yticklabels(['NSR', 'Pre-AF'], fontsize=10, rotation=0)

            # Subplot title
            ax.set_title(name, fontsize=12, fontweight='bold', pad=10)

        # Main title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrices comparison saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

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
            'Hybrid': '#2ca02c',    # green
            'Single-scale': '#ff7f0e',
            'Multi-scale Uniform': '#1f77b4',
            'Multi-scale  Trained': '#2ca02c',
            'Multi-scale': '#1f77b4',
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
    evaluator.evaluate_patient_level()

    if save_metrics_path:
        evaluator.save_metrics_json(metrics, save_metrics_path)

    return metrics


if __name__ == '__main__':

    model = 'multi'
    attention_mode = 'trained'

    if model == 'multi':
        model_path = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/results_new/multi/final_checkpoint.pt')
    elif model == 'single':
        model_path = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/results_new2/single/final_checkpoint.pt')
    else:
        model_path = Path('/data/results_final/hybrid/best_checkpoint.pt')

    data_dir = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_final/utrp_dataset_db8')
    results_dir = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new_new/results')

    metrics = evaluate_model(
        model_path,
        data_dir,
        batch_size=32,
        save_metrics_path=results_dir / f'test_metrics_{model}_new_{attention_mode}.json',
        attention_mode=attention_mode
    )

    # ModelEvaluator.set_seed(42)
    # evaluator = ModelEvaluator(model_path, data_dir, model_arch=model)
    # evaluator.load_model(attention_mode=attention_mode)
    # evaluator.load_test_data(batch_size=32)
    # evaluator.evaluate()


    # evaluator.plot_confusion_matrix(
    #     normalize=True,
    #     save_path=results_dir / f'confusion_matrix_{model}_{attention_mode}_normalized.png',
    #     show=False
    # )
    #
    # evaluator.plot_confusion_matrix(
    #     normalize=False,
    #     save_path=results_dir / f'confusion_matrix_{model}_{attention_mode}.png',
    #     show=False
    # )
