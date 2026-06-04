import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve, auc as sklearn_auc
from typing import List, Dict, Optional

from src.CNN.models.full_band_cnn import FullBandCNN
from src.CNN.models.multi_scale_cnn import MultiScaleCNN
from src.CNN.training.data_loaders import FullDataset, AugmentedSubset
from src.CNN.utils.utils import DEVICE, get_subject_group, set_seed


class CrossValidator:
    """
    cross-validation trainer supporting:
    - full-band models
    - multi-scale models with configurable subbands (ablation)
    - denoised or non-denoised dataset
    """

    def __init__(
        self,
        h5_dir: Path,
        output_dir: Path,
        model_type: str = 'multi',
        subband_keys: Optional[List[str]] = None,
        denoise: bool = False,
        n_folds: int = 10,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-4,
        dropout: float = 0.5,
        patience: int = 10,
        scheduler_patience: int = 5,
        grad_clip: float = 1.0,
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            h5_dir: Directory containing HDF5 dataset files
            output_dir: Directory to save results and checkpoints
            model_type: 'single' for full-band, 'multi' for multi-scale
            subband_keys: List of subbands to use (e.g., ['D2', 'D3', 'D4']).
            denoise: Whether to use denoised data
            n_folds: Number of cross-validation folds
            epochs: Maximum training epochs per fold
            batch_size: Training batch size
            lr: Learning rate
            dropout: Dropout rate
            patience: Early stopping patience (epochs)
            scheduler_patience: LR scheduler patience (epochs)
            grad_clip: Gradient clipping norm
            seed: Random seed
            device: torch device (defaults to DEVICE from utils)
        """
        self.h5_dir = Path(h5_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type
        self.subband_keys = subband_keys
        self.denoise = denoise

        self.n_folds = n_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.grad_clip = grad_clip
        self.seed = seed
        self.device = device or DEVICE

        set_seed(self.seed)

    def _get_dataset_mode(self) -> str:
        """determine dataset loading mode based on configuration."""
        if self.model_type == 'single':
            return 'single'
        elif self.denoise:
            return 'multi_denoise'
        else:
            return 'multi'

    def _create_model(self) -> nn.Module:
        """create model based on configuration"""
        if self.model_type == 'single':
            return FullBandCNN(dropout=self.dropout).to(self.device)
        else:
            n_subbands = len(self.subband_keys) if self.subband_keys else 7
            return MultiScaleCNN(n_subbands=n_subbands, dropout=self.dropout).to(self.device)

    def _train_epoch(self, model, loader, criterion, optimizer):
        """train for one epoch"""
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        return total_loss / n, correct / n

    @torch.no_grad()
    def _eval_epoch(self, model, loader, criterion):
        """evaluate for one epoch"""
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        all_probs, all_labels = [], []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        # compute metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc_roc = sklearn_auc(fpr, tpr)
        auprc = average_precision_score(all_labels, all_probs)

        # use Youden's Index for threshold
        youden_idx = np.argmax(tpr - fpr)
        threshold = _[youden_idx] if len(_) > youden_idx else 0.5
        preds = (all_probs >= threshold).astype(int)

        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'auc_roc': float(auc_roc),
            'auprc': float(auprc),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        }

        return total_loss / n, correct / n, metrics, all_probs, all_labels

    @torch.no_grad()
    def _get_attention_weights(self, model, loader):
        """extract attention weights for multi-scale models"""
        if self.model_type == 'single':
            return None

        model.eval()
        all_weights = []

        for x, _ in loader:
            x = x.to(self.device)
            weights = model.get_attention_weights(x)
            all_weights.append(weights.cpu().numpy())

        all_weights = np.concatenate(all_weights, axis=0)
        return np.mean(all_weights, axis=0)

    def run(self) -> Dict:
        """
        run complete cross-validation training.

        Returns:
            dictionary with aggregated results across folds
        """
        print(f"model: {self.model_type}")
        if self.model_type == 'multi':
            subbands = self.subband_keys if self.subband_keys else ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
            print(f"subbands: {subbands}")
            print(f"denoise: {self.denoise}")

        dataset_mode = self._get_dataset_mode()
        dataset = FullDataset(
            self.h5_dir,
            mode=dataset_mode,
            subband_keys=self.subband_keys
        )

        groups = np.array([get_subject_group(r) for r in dataset.record_ids])

        gkf = GroupKFold(n_splits=self.n_folds)
        dummy_X = np.zeros(len(dataset))

        fold_results = []
        fold_probs = []
        fold_labels = []
        fold_attn_weights = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(dummy_X, dataset.labels, groups), start=1):
            print(f"fold {fold}/{self.n_folds}")

            tr_groups = groups[tr_idx]
            unique_tr_groups = np.unique(tr_groups)
            n_tr_nsr_subjects = int((unique_tr_groups >= 200).sum())  # NSR groups are >= 200
            n_tr_paf_subjects = int((unique_tr_groups < 200).sum())   # Pre-AF groups are < 200
            w_preaf = n_tr_nsr_subjects / n_tr_paf_subjects if n_tr_paf_subjects > 0 else 1.0
            class_weights = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(self.device)

            tr_subset = AugmentedSubset(dataset, tr_idx, augment=True)
            va_subset = Subset(dataset, va_idx)

            tr_loader = DataLoader(tr_subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            va_loader = DataLoader(va_subset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            # initialize model
            set_seed(self.seed + fold)
            model = self._create_model()

            if fold == 1:
                print(f"model parameters: {model.count_params():,}\n")

            # optimizer and scheduler
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=self.scheduler_patience
            )

            # training loop
            best_auc = 0.0
            patience_counter = 0

            for epoch in range(1, self.epochs + 1):
                tr_loss, tr_acc = self._train_epoch(model, tr_loader, criterion, optimizer)
                va_loss, va_acc, va_metrics, va_probs, va_labels = self._eval_epoch(model, va_loader, criterion)

                scheduler.step(va_metrics['auc_roc'])

                # early stopping and checkpointing
                if va_metrics['auc_roc'] > best_auc:
                    best_auc = va_metrics['auc_roc']
                    patience_counter = 0

                    checkpoint = {
                        'fold': fold,
                        'epoch': epoch,
                        'model_type': self.model_type,
                        'state_dict': model.state_dict(),
                        'val_indices': va_idx.tolist(),
                        'config': {
                            'n_subbands': len(self.subband_keys) if self.subband_keys else 7,
                            'subband_keys': self.subband_keys,
                            'denoise': self.denoise,
                            'dropout': self.dropout,
                            'lr': self.lr,
                            'seed': self.seed,
                        },
                    }
                    torch.save(checkpoint, self.output_dir / f'fold_{fold}_best.pt')
                else:
                    patience_counter += 1

                if (epoch % 10 == 0) or (epoch == 1):
                    print(f"epoch {epoch:3d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_auc={va_metrics['auc_roc']:.4f} (best={best_auc:.4f})")

                if patience_counter >= self.patience:
                    print(f"early stopping at epoch {epoch}")
                    break

            # load best model and save final results
            checkpoint = torch.load(self.output_dir / f'fold_{fold}_best.pt', map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])

            _, _, final_metrics, final_probs, final_labels = self._eval_epoch(model, va_loader, criterion)

            # get attention weights for multi-scale models
            if self.model_type == 'multi':
                attn_weights = self._get_attention_weights(model, va_loader)
                fold_attn_weights.append(attn_weights.tolist())

            fold_results.append(final_metrics)
            fold_probs.append(final_probs)
            fold_labels.append(final_labels)

            print(f"\nfold {fold} final results:")
            print(f"AUC: {final_metrics['auc_roc']:.4f}")
            print(f"AUPRC: {final_metrics['auprc']:.4f}")
            print(f"sensitivity: {final_metrics['sensitivity']:.4f}")
            print(f"specificity: {final_metrics['specificity']:.4f}")

        def aggregate(fold_results):
            keys = fold_results[0].keys()
            return {
                key: {
                    'mean': float(np.mean([f[key] for f in fold_results])),
                    'std': float(np.std([f[key] for f in fold_results])),
                    'median': float(np.median([f[key] for f in fold_results])),
                    'min': float(np.min([f[key] for f in fold_results])),
                    'max': float(np.max([f[key] for f in fold_results])),
                    'values': [float(f[key]) for f in fold_results],
                }
                for key in keys if isinstance(fold_results[0][key], (int, float))
            }

        aggregated = aggregate(fold_results)

        results = {
            'config': {
                'model_type': self.model_type,
                'subband_keys': self.subband_keys,
                'denoise': self.denoise,
                'n_folds': self.n_folds,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'dropout': self.dropout,
                'patience': self.patience,
                'seed': self.seed,
            },
            'aggregated': aggregated,
            'per_fold': fold_results,
        }

        if self.model_type == 'multi' and fold_attn_weights:
            results['attention_weights'] = {
                'per_fold': fold_attn_weights,
                'mean': np.mean(fold_attn_weights, axis=0).tolist(),
            }

        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results
