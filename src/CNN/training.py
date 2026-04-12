"""
Training utilities and cross-validation runner.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Type, Any, Tuple
from itertools import product
from tqdm import tqdm
import random

from base_cnn import BaseCNN
from dataset import UTRPDataset, create_dataloaders, compute_class_weights
from cross_validation import PatientStratifiedKFold, get_remaining_labels
import re

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrossValidationRunner:
    """
    Runs K-fold cross-validation with hyperparameter tuning.

    Workflow:
      1. Split patients into train/val folds (stratified)
      2. For each hyperparameter combination:
           - Train on each fold
           - Record validation metrics
      3. Select best hyperparameters by mean val AUC
      4. Retrain on full train+val set with best config
      5. Evaluate on held-out test set
      6. Save all results
    """

    def __init__(
        self,
        model_class: Type[BaseCNN],
        hdf5_path: Path,
        mode: str,
        results_dir: Path,
        n_folds: int = 5,
        seed: int = 42
    ):
        """
        Args:
            model_class: CNN class (SingleScaleCNN or MultiScaleCNN)
            hdf5_path: Path to UTRP HDF5 dataset
            mode: 'single' or 'multi'
            results_dir: Directory to save results
            n_folds: Number of CV folds
            seed: Random seed
        """
        self.model_class = model_class
        self.hdf5_path = Path(hdf5_path)
        self.mode = mode
        self.results_dir = Path(results_dir)
        self.n_folds = n_folds
        self.seed = seed

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        set_seed(seed)

        print(f"\nCrossValidationRunner initialized")
        print(f"  Model: {model_class.__name__}")
        print(f"  Mode: {mode}")
        print(f"  Folds: {n_folds}")
        print(f"  Results dir: {results_dir}")

    def get_subject_id(self, record_id: str) -> str:
        """
        Maps record IDs to unique human subjects for cross-validation splitting.

        Dataset structure:
        - n01-n50: Healthy controls (each is a unique subject)
        - p01/p02 are from same PAF patient (p01=Pre-AF, p02=NSR)
        - p03/p04 are from same PAF patient (p03=Pre-AF, p04=NSR), etc.

        Returns subject ID for grouping in CV splits.
        """
        match = re.search(r'([pn])(\d+)', record_id)
        if match:
            prefix = match.group(1)
            num = int(match.group(2))

            if prefix == 'p':
                # PAF patients: group odd/even pairs (p01/p02 -> subject_p01)
                subject_num = ((num + 1) // 2) * 2 - 1  # Maps to odd: 1,2->1, 3,4->3, etc.
                return f"subject_p{subject_num:02d}"
            else:
                # Healthy controls: each n* is a unique subject
                return f"subject_n{num:02d}"

        return str(record_id).strip().lower()

    def run(
        self,
        hyperparam_grid: Dict[str, List[Any]],
        train_patients: List[str],
        test_patients: List[str],
        epochs: int = 100,
        patience: int = 10,
        num_workers: int = 4
    ) -> Dict:
        """
        Run full cross-validation workflow.

        Args:
            hyperparam_grid: Dict of hyperparameter lists to search
              e.g., {'lr': [1e-3, 1e-4], 'dropout': [0.3, 0.5], 'batch_size': [16, 32]}
            train_patients: RECORD IDs for training/validation (not test)
            test_patients: RECORD IDs for held-out test set
            epochs: Max epochs per training run
            patience: Early stopping patience
            num_workers: DataLoader workers

        Returns:
            Results dict with best config, cv results, test results
        """
        print("\n" + "="*80)
        print("CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
        print("="*80)

        # train_patients is a list of patient IDs
        # Split by patients (stratified by label)
        cv_splitter = PatientStratifiedKFold(n_splits=self.n_folds, random_state=self.seed)
        folds = cv_splitter.split(train_patients)

        # Generate hyperparameter combinations
        param_names = list(hyperparam_grid.keys())
        param_values = list(hyperparam_grid.values())
        param_combinations = list(product(*param_values))

        print(f"\nHyperparameter grid:")
        for name, values in hyperparam_grid.items():
            print(f"  {name}: {values}")
        print(f"Total combinations: {len(param_combinations)}")

        # Store results for all combinations
        all_results = []

        # Search over hyperparameters
        for combo_idx, param_combo in enumerate(param_combinations, 1):
            params = dict(zip(param_names, param_combo))

            print(f"\n{'='*80}")
            print(f"Combination {combo_idx}/{len(param_combinations)}: {params}")
            print(f"{'='*80}")

            fold_results = []

            # Train on each fold
            for fold_idx, (train_pids, val_pids) in enumerate(folds, 1):
                print(f"\n--- Fold {fold_idx}/{self.n_folds} ---")

                # Create datasets
                train_dataset = UTRPDataset(
                    self.hdf5_path, mode=self.mode, augment=True, patient_ids=train_pids
                )
                val_dataset = UTRPDataset(
                    self.hdf5_path, mode=self.mode, augment=False, patient_ids=val_pids
                )

                # Create dataloaders
                batch_size = params.get('batch_size', 32)
                train_loader, val_loader = create_dataloaders(
                    train_dataset, val_dataset, batch_size=batch_size, num_workers=num_workers
                )

                # Compute class weights
                class_weights = compute_class_weights(train_dataset)

                # Create model
                model_params = {k: v for k, v in params.items() if k not in ['lr', 'batch_size']}

                model = self.model_class(**model_params)

                # Train
                lr = params.get('lr', 1e-3)
                history = model.fit(
                    train_loader, val_loader,
                    epochs=epochs, lr=lr, patience=patience,
                    class_weights=class_weights,
                    weight_decay=params.get('weight_decay', 0.1)
                )

                # Evaluate on validation set
                val_metrics = model.evaluate(val_loader)

                # Save fold results
                fold_result = {
                    'fold': fold_idx,
                    **params,
                    **{f'val_{k}': v for k, v in val_metrics.items() if k != 'confusion_matrix'},
                    'n_epochs': len(history['train_loss']),
                    'best_train_loss': min(history['train_loss']),
                    'best_val_loss': min(history['val_loss'])
                }
                fold_results.append(fold_result)

                # Plot training curves for this fold
                curve_path = self.results_dir / f"training_curves_fold{fold_idx}_combo{combo_idx}.png"
                model.plot_training_curves(history, save_path=curve_path)

                print(f"\nFold {fold_idx} Results:")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}")

            # Aggregate fold results
            mean_val_auc = np.mean([r['val_auc'] for r in fold_results])
            std_val_auc = np.std([r['val_auc'] for r in fold_results])

            combo_summary = {
                **params,
                'mean_val_auc': mean_val_auc,
                'std_val_auc': std_val_auc,
                'mean_val_accuracy': np.mean([r['val_accuracy'] for r in fold_results]),
                'mean_val_f1': np.mean([r['val_f1'] for r in fold_results]),
                'fold_results': fold_results
            }

            all_results.append(combo_summary)

            print(f"\nCombination Summary:")
            print(f"  Mean Val AUC: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
            print(f"  Mean Val Accuracy: {combo_summary['mean_val_accuracy']:.4f}")

        # Select best hyperparameters
        best_combo = max(all_results, key=lambda x: x['mean_val_auc'])

        print(f"\n{'='*80}")
        print("BEST HYPERPARAMETERS (by mean val AUC)")
        print(f"{'='*80}")
        for key, value in best_combo.items():
            if key not in ['fold_results', 'mean_val_auc', 'std_val_auc']:
                print(f"  {key}: {value}")
        print(f"\nMean Val AUC: {best_combo['mean_val_auc']:.4f} ± {best_combo['std_val_auc']:.4f}")

        # Retrain on full train+val set with best config
        print(f"\n{'='*80}")
        print("RETRAINING ON FULL TRAIN+VAL SET")
        print(f"{'='*80}")

        full_train_dataset = UTRPDataset(
            self.hdf5_path, mode=self.mode, augment=True, patient_ids=train_patients
        )
        test_dataset = UTRPDataset(
            self.hdf5_path, mode=self.mode, augment=False, patient_ids=test_patients
        )

        # Create dummy validation set for early stopping (10% of train)
        n_val = max(1, len(train_patients) // 10)
        val_pids = train_patients[:n_val]
        train_pids = train_patients[n_val:]

        retrain_dataset = UTRPDataset(
            self.hdf5_path, mode=self.mode, augment=True, patient_ids=train_pids
        )
        retrain_val_dataset = UTRPDataset(
            self.hdf5_path, mode=self.mode, augment=False, patient_ids=val_pids
        )

        batch_size = best_combo.get('batch_size', 32)
        train_loader, val_loader = create_dataloaders(
            retrain_dataset, retrain_val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader, _ = create_dataloaders(
            test_dataset, test_dataset, batch_size=batch_size, num_workers=0
        )

        # Create final model
        final_model_params = {k: v for k, v in best_combo.items()
                              if k not in ['lr', 'batch_size', 'mean_val_auc', 'std_val_auc',
                                          'mean_val_accuracy', 'mean_val_f1', 'fold_results']}

        final_model = self.model_class(**final_model_params)

        # Train
        lr = best_combo.get('lr', 1e-3)
        class_weights = compute_class_weights(retrain_dataset)
        final_history = final_model.fit(
            train_loader, val_loader,
            epochs=epochs, lr=lr, patience=patience,
            class_weights=class_weights
        )

        # Evaluate on test set
        print(f"\n{'='*80}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*80}")

        test_metrics = final_model.evaluate(test_loader)

        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")

        # Save final model
        model_name = f"best_{self.mode}_scale.pt"
        model_path = self.results_dir / model_name
        final_model.save(model_path, hyperparams=final_model_params)

        # Save CV results to CSV
        cv_results_df = pd.DataFrame([
            {**combo, **fold_res}
            for combo in all_results
            for fold_res in combo['fold_results']
        ])
        cv_path = self.results_dir / f"cv_results_{self.mode}.csv"
        cv_results_df.to_csv(cv_path, index=False)
        print(f"\nCV results saved to {cv_path}")

        # Save test metrics
        test_results_df = pd.DataFrame([test_metrics])
        test_path = self.results_dir / f"test_results_{self.mode}.csv"
        test_results_df.to_csv(test_path, index=False)
        print(f"Test results saved to {test_path}")

        # Generate visualizations
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        # Training curves
        final_model.plot_training_curves(
            final_history,
            save_path=self.results_dir / f"final_training_curves_{self.mode}.png"
        )

        # ROC curve
        final_model.plot_roc(
            test_loader,
            save_path=self.results_dir / f"roc_curve_{self.mode}.png"
        )

        # Confusion matrix
        final_model.plot_confusion_matrix(
            test_loader,
            save_path=self.results_dir / f"confusion_matrix_{self.mode}.png"
        )

        print(f"\n{'='*80}")
        print("CROSS-VALIDATION COMPLETE")
        print(f"{'='*80}")

        return {
            'best_config': best_combo,
            'cv_results': all_results,
            'test_metrics': test_metrics,
            'final_model': final_model
        }