"""
Attention analysis script.

Tests three attention configurations on the held-out test set
using the same trained branch backbones:

  1. Trained attention     — weights as saved in checkpoint
  2. Uniform attention     — all sigmoid outputs fixed to 0.5 (equal weighting)
  3. Oracle attention      — manually set weights based on LR AUC ranking
                             (D5 highest, D7 lowest — from wavelet selection)

If (2) beats (1): attention overfit to val patients.
If (1) beats (2): attention genuinely learned useful weighting.
If (3) beats both: there is headroom from better subband selection.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from src.CNN.dataset.dataset import UTRPDataset
from src.CNN.models.complexity_test import LateFusionWithAttention

# ── Config ─────────────────────────────────────────────────────────────────────
CKPT      = Path('/results_new/multi/final_checkpoint.pt')
DATA_DIR  = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_final/utrp_dataset_db8')
SAVE_PATH = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_new/attention_analysis.json')

SUBBAND_NAMES = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
DEVICE        = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# LR AUC from wavelet selection experiment (db8, used for oracle weights)
# Higher AUC → higher oracle weight
LR_AUC = {
    'D2': 0.485, 'D3': 0.782, 'D4': 0.685,
    'D5': 0.737, 'D6': 0.584, 'D7': 0.356, 'D8': 0.697
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_base_model(ckpt_path: Path, n_subbands: int = 7,
                     dropout: float = 0.5) -> LateFusionWithAttention:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = LateFusionWithAttention(n_subbands=n_subbands,
                                    dropout=dropout).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model


def clone_model(model: LateFusionWithAttention) -> LateFusionWithAttention:
    clone = deepcopy(model)
    clone.eval()
    return clone


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             label: str = '') -> dict:
    """Run inference and return metrics dict."""
    all_probs, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    auc   = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens  = tp / (tp + fn) if tp + fn > 0 else 0.0
    spec  = tn / (tn + fp) if tn + fp > 0 else 0.0

    result = {
        'auc':         float(auc),
        'f1':          float(f1),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }

    print(f"\n  [{label}]")
    print(f"    AUC={auc:.4f}  F1={f1:.4f}  "
          f"Sens={sens:.3f}  Spec={spec:.3f}  "
          f"TP={tp} TN={tn} FP={fp} FN={fn}")
    return result


def get_attention_weights(model: LateFusionWithAttention,
                           loader: DataLoader) -> np.ndarray:
    """Collect mean attention weights across the test set."""
    all_w = []
    with torch.no_grad():
        for x, _ in loader:
            w = model.get_attention_weights(x.to(DEVICE))
            all_w.append(w.cpu().numpy())
    return np.concatenate(all_w).mean(axis=0)


# ── Attention override helpers ─────────────────────────────────────────────────

def set_uniform_attention(model: LateFusionWithAttention) -> None:
    """
    Override attention to produce sigmoid(0) = 0.5 for all subbands.
    Zero all attention weights and biases → uniform weighting.
    """
    with torch.no_grad():
        for layer in model.attention:
            if hasattr(layer, 'weight'):
                layer.weight.zero_()
            if hasattr(layer, 'bias'):
                layer.bias.zero_()
    print("  Attention set to uniform (sigmoid(0)=0.5 for all branches)")


def set_oracle_attention(model: LateFusionWithAttention,
                          lr_auc: dict, subband_names: list) -> None:
    """
    Override attention to fixed values proportional to LR AUC scores.
    Uses inverse sigmoid (logit) to set the final bias of the attention
    so that sigmoid output matches the desired weights.

    This simulates what the attention would look like if it perfectly
    followed the RQA discriminability ranking.
    """
    # Normalise LR AUC scores to [0.1, 0.9] range for stable logit
    aucs   = np.array([lr_auc[n] for n in subband_names], dtype=np.float32)
    lo, hi = aucs.min(), aucs.max()
    scaled = 0.1 + 0.8 * (aucs - lo) / (hi - lo + 1e-9)

    # logit = log(p / (1-p))
    logits = np.log(scaled / (1.0 - scaled))

    with torch.no_grad():
        # Zero all attention layers
        for layer in model.attention:
            if hasattr(layer, 'weight'):
                layer.weight.zero_()
            if hasattr(layer, 'bias'):
                layer.bias.zero_()
        # Set the final bias to desired logit values
        # attention[-2] is the last Linear layer (before Sigmoid)
        final_linear = model.attention[-2]
        if hasattr(final_linear, 'bias'):
            final_linear.bias.copy_(
                torch.tensor(logits, dtype=torch.float32).to(DEVICE)
            )

    weights = torch.sigmoid(torch.tensor(logits)).numpy()
    print("  Oracle attention weights (from LR AUC):")
    for name, w in zip(subband_names, weights):
        bar = '█' * int(w * 20)
        print(f"    {name}: {w:.4f}  {bar}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print("ATTENTION ANALYSIS")
    print(f"Checkpoint: {CKPT}")
    print(f"{'='*60}")

    # Load checkpoint info
    ckpt = torch.load(CKPT, map_location=DEVICE)
    print(f"\nCheckpoint: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")
    print(f"Config: {ckpt['config']}")

    # Load test data
    test_ds     = UTRPDataset(DATA_DIR, split='test', mode='multi', augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"\nTest set: {len(test_ds)} samples  "
          f"(NSR={test_ds.get_class_distribution()[0]}  "
          f"PreAF={test_ds.get_class_distribution()[1]})")

    results = {}

    # ── 1. Trained attention ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("1. TRAINED ATTENTION (as saved in checkpoint)")
    model_trained = load_base_model(CKPT)
    results['trained'] = evaluate(model_trained, test_loader, 'Trained attention')

    mean_w = get_attention_weights(model_trained, test_loader)
    print("  Mean attention weights (test set):")
    for name, w in zip(SUBBAND_NAMES, mean_w):
        bar = '█' * int(w * 30)
        print(f"    {name}: {w:.4f}  {bar}")
    results['trained']['mean_attention'] = {
        n: float(w) for n, w in zip(SUBBAND_NAMES, mean_w)
    }

    # ── 2. Uniform attention ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("2. UNIFORM ATTENTION (all branches weighted equally)")
    model_uniform = load_base_model(CKPT)
    set_uniform_attention(model_uniform)
    results['uniform'] = evaluate(model_uniform, test_loader, 'Uniform attention')

    # ── 3. Oracle attention (LR AUC ranking) ──────────────────────────────────
    print(f"\n{'─'*50}")
    print("3. ORACLE ATTENTION (weights ∝ LR AUC from wavelet selection)")
    model_oracle = load_base_model(CKPT)
    set_oracle_attention(model_oracle, LR_AUC, SUBBAND_NAMES)
    results['oracle'] = evaluate(model_oracle, test_loader, 'Oracle attention')

    # ── 4. Single best subband (D3 — highest trained attention weight) ─────────
    print(f"\n{'─'*50}")
    print("4. D3-ONLY ATTENTION (model's most attended subband)")
    best_idx = int(np.argmax(mean_w))
    best_name = SUBBAND_NAMES[best_idx]
    print(f"  Setting only {best_name} (idx={best_idx}) to 1.0, rest to 0.0")
    model_best = load_base_model(CKPT)
    with torch.no_grad():
        for layer in model_best.attention:
            if hasattr(layer, 'weight'):
                layer.weight.zero_()
            if hasattr(layer, 'bias'):
                layer.bias.zero_()
        final_linear = model_best.attention[-2]
        if hasattr(final_linear, 'bias'):
            bias = torch.full((7,), -10.0)   # sigmoid(-10) ≈ 0
            bias[best_idx] = 10.0             # sigmoid(10)  ≈ 1
            final_linear.bias.copy_(bias.to(DEVICE))
    results[f'{best_name}_only'] = evaluate(
        model_best, test_loader, f'{best_name}-only attention'
    )

    # ── 5. D5-only (most discriminative per RQA) ───────────────────────────────
    print(f"\n{'─'*50}")
    print("5. D5-ONLY ATTENTION (most discriminative per LR AUC)")
    d5_idx = SUBBAND_NAMES.index('D5')
    model_d5 = load_base_model(CKPT)
    with torch.no_grad():
        for layer in model_d5.attention:
            if hasattr(layer, 'weight'):
                layer.weight.zero_()
            if hasattr(layer, 'bias'):
                layer.bias.zero_()
        final_linear = model_d5.attention[-2]
        if hasattr(final_linear, 'bias'):
            bias = torch.full((7,), -10.0)
            bias[d5_idx] = 10.0
            final_linear.bias.copy_(bias.to(DEVICE))
    results['D5_only'] = evaluate(model_d5, test_loader, 'D5-only attention')

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Configuration':<25} {'AUC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}")
    print(f"  {'-'*57}")
    for name, m in results.items():
        if 'auc' in m:
            print(f"  {name:<25} {m['auc']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['sensitivity']:>8.3f} {m['specificity']:>8.3f}")

    # Interpretation
    print(f"\n{'─'*60}")
    t_auc = results['trained']['auc']
    u_auc = results['uniform']['auc']
    o_auc = results['oracle']['auc']

    if u_auc > t_auc + 0.01:
        print("→ Uniform > Trained: attention OVERFIT to val patients.")
        print("  Equal weighting of trained backbones generalises better.")
    elif t_auc > u_auc + 0.01:
        print("→ Trained > Uniform: attention learned GENUINE subband weighting.")
        print("  The model correctly identifies which subbands are most informative.")
    else:
        print("→ Trained ≈ Uniform: attention makes little difference.")
        print("  Subband selection is what matters, not their relative weighting.")

    if o_auc > t_auc + 0.01:
        print(f"\n→ Oracle > Trained (Δ={o_auc-t_auc:+.3f}): RQA-based weighting")
        print("  outperforms learned attention. Consider retraining with")
        print("  attention regularised toward discriminability-based weights.")

    # Save results
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {SAVE_PATH}")


if __name__ == '__main__':
    main()