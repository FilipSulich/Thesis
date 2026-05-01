#!/usr/bin/env python3
"""
Extract attention weights from already-trained cross-validation models.
This script loads each fold's best checkpoint and collects attention weights
from the corresponding validation set.
"""

import json
import re
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.CNN.models.models import MultiScaleCNN
from src.CNN.training.cross_validation import (
    CombinedPNGDataset, get_subject_group,
    SUBBAND_KEYS, N_FOLDS, SEED, DEVICE, N_SUBBANDS
)

# Paths
PNG_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2')
RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/multi')
BATCH_SIZE = 32


@torch.no_grad()
def collect_attention_weights(model, loader, device) -> np.ndarray:
    """
    Collect mean attention weights across all batches in loader.
    Returns (n_subbands,) mean weight vector.
    """
    model.eval()
    all_weights = []
    for x, _ in tqdm(loader, desc="Collecting weights", leave=False):
        x = x.to(device)
        # Extract branch features and compute attention
        feats = [model.branches[i](x[:, i*3:(i+1)*3])
                 for i in range(model.n_subbands)]
        stacked = torch.stack(feats, dim=1)
        flat = stacked.reshape(stacked.size(0), -1)
        weights = model.attention(flat)  # (B, n_subbands)
        all_weights.append(weights.cpu().numpy())
    return np.concatenate(all_weights, axis=0).mean(axis=0)  # (n_subbands,)


def extract_weights_from_checkpoints():
    """
    Load each fold's checkpoint and extract attention weights from
    its validation set by recreating the CV splits.
    """
    print("Loading combined dataset...")
    dataset = CombinedPNGDataset(PNG_DIR, mode='multi', cache=True)

    # Recreate the exact same splits as during training
    groups = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf = GroupKFold(n_splits=N_FOLDS)
    dummy_X = np.zeros(len(dataset))

    fold_weights = []

    print(f"\nExtracting attention weights from {N_FOLDS} folds...")
    print(f"Device: {DEVICE}")
    print(f"Results dir: {RESULTS_DIR}\n")

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(dummy_X, dataset.labels, groups), start=1):

        print(f"{'='*60}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*60}")

        # Load checkpoint
        ckpt_path = RESULTS_DIR / f'fold_{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            print(f"   Skipping fold {fold}")
            continue

        print(f"Loading checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

        # Create model and load weights
        model = MultiScaleCNN(N_SUBBANDS, dropout=0.5).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        print(f"Val AUC from training: {checkpoint['val_auc']:.4f}")
        print(f"Validation set: {len(va_idx)} samples")

        # Create validation loader
        va_subset = Subset(dataset, va_idx)
        va_loader = DataLoader(
            va_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=DEVICE.type == 'cuda'
        )

        # Collect attention weights
        weights = collect_attention_weights(model, va_loader, DEVICE)
        fold_weights.append(weights)

        print(f"\nAttention weights for fold {fold}:")
        for name, w in zip(SUBBAND_KEYS[:N_SUBBANDS], weights):
            print(f"  {name}: {w:.4f}")
        print()

    if not fold_weights:
        print("\n❌ No checkpoints found! Make sure you've run cross_validation.py first.")
        return

    # Convert to array and save
    fold_weights_array = np.array(fold_weights)  # (n_folds, n_subbands)

    output_path = RESULTS_DIR / 'fold_attention_weights.npy'
    np.save(output_path, fold_weights_array)
    print(f"\n✅ Saved attention weights → {output_path}")
    print(f"   Shape: {fold_weights_array.shape}")

    # Print summary
    means = fold_weights_array.mean(axis=0)
    stds = fold_weights_array.std(axis=0)

    print(f"\n{'='*60}")
    print("SUMMARY: Mean Attention Weights Across Folds")
    print(f"{'='*60}")
    print(f"{'Subband':<10} {'Mean':>10} {'Std':>10}")
    print('-' * 60)
    for name, m, s in zip(SUBBAND_KEYS[:N_SUBBANDS], means, stds):
        print(f"{name:<10} {m:>10.4f} {s:>10.4f}")
    print(f"{'='*60}\n")

    return fold_weights_array


if __name__ == '__main__':
    weights = extract_weights_from_checkpoints()