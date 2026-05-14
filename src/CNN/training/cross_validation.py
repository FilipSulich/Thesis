import json
import re
import shutil
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve,
)
from scipy import stats
from PIL import Image
from tqdm import tqdm

from src.CNN.models.models import SingleScaleCNN, MultiScaleCNN

H5_DIR      = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_final/utrp_dataset_db8_full_denoise_seed_42')
PNG_DIR     = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2_denoise')
SAVE_DIR    = Path('/results_final/results_kfold2_denoise')

MODEL_TYPE   = 'multi'

N_FOLDS      = 10
N_SUBBANDS   = 7
SUBBAND_KEYS = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-4
DROPOUT      = 0.5
PATIENCE     = 20         # early stopping patience
SCHED_PAT    = 5           # LR scheduler patience
GRAD_CLIP    = 1.0
SEED         = 42

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cpu')
)

# Update save directory based on model type
SAVE_DIR = SAVE_DIR / MODEL_TYPE
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Model:    {MODEL_TYPE.upper()}")
print(f"Device:   {DEVICE}")
print(f"H5 dir:   {H5_DIR}")
print(f"PNG dir:  {PNG_DIR}")
print(f"Save dir: {SAVE_DIR}")


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

set_seed(SEED)


def extract_h5_to_png(h5_dir: Path, png_dir: Path,
                       splits=('train', 'val')) -> None:
    """
    Extract HDF5 images to PNG files on local disk.
    Layout:
        png_dir/
          train/
            000000_D2.png   (sample_idx_subband_key.png)
            000000_D3.png
            ...
            000000_single.png
          val/
            ...
    Also saves metadata.json per split with labels, record_ids, segment_indices.
    """
    png_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        h5_path  = h5_dir / f'{split}.h5'
        out_dir  = png_dir / split
        meta_path = out_dir / 'metadata.json'

        if meta_path.exists():
            print(f"  ✓ {split} already extracted — skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting {split}...")

        with h5py.File(h5_path, 'r') as f:
            labels   = f['metadata/labels'][:]
            rids     = np.array([r.decode() for r in f['metadata/record_ids'][:]])
            sidxs    = f['metadata/segment_indices'][:]
            n        = len(labels)

            # Collect all available image keys
            img_keys = [k for k in SUBBAND_KEYS + ['single'] if k in f]

            for key in img_keys:
                data = f[key][:]    # (N, 3, H, W) float32
                print(f"    {key}: {data.shape}")

                for i in range(n):
                    img_np  = (data[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np, mode='RGB')
                    img_pil.save(out_dir / f'{i:06d}_{key}.png', optimize=False)

                if (i + 1) % 500 == 0:
                    print(f"      {i+1}/{n}")

        # Save metadata
        meta = {
            'labels':          labels.tolist(),
            'record_ids':      rids.tolist(),
            'segment_indices': sidxs.tolist(),
            'img_keys':        img_keys,
            'n_samples':       int(n),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        print(f"  ✓ {split}: {n} samples → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Dataset (reads PNGs from local disk)
# ══════════════════════════════════════════════════════════════════════════════

class PNGSubbandDataset(Dataset):
    """
    Loads per-subband UTRP images from PNG files.

    For multi-scale: Returns (x, label) where x is (N_SUBBANDS*3, 256, 256) float32.
    For single-scale: Returns (x, label) where x is (3, 256, 256) float32.

    Images are cached in RAM on first access if cache=True.
    For Colab with ~12GB RAM and ~2250 images × 8 keys × 256×256×3 bytes
    that is ~3.5GB — feasible.
    """

    def __init__(self, png_dir: Path, split: str,
                 mode: str = 'multi',
                 subband_keys=None, augment: bool = False,
                 cache: bool = True):
        self.split        = split
        self.mode         = mode  # 'single' or 'multi'
        self.subband_keys = subband_keys or SUBBAND_KEYS if mode == 'multi' else []
        self.augment      = augment
        self.cache        = cache
        self.split_dir    = Path(png_dir) / split

        with open(self.split_dir / 'metadata.json') as f:
            meta = json.load(f)

        self.labels      = np.array(meta['labels'],          dtype=np.int64)
        self.record_ids  = np.array(meta['record_ids'])
        self.seg_indices = np.array(meta['segment_indices'], dtype=np.int16)
        self.n           = len(self.labels)

        # Optional RAM cache: {(idx, key): np.ndarray (3,H,W)}
        self._cache: dict = {} if cache else None

        n_paf = int((self.labels == 1).sum())
        n_nsr = int((self.labels == 0).sum())
        aug_s = ' +aug' if augment else ''
        print(f"PNGSubbandDataset [{mode}-{split}{aug_s}]: "
              f"{self.n} samples  NSR={n_nsr}  Pre-AF={n_paf}")

    def _load_img(self, idx: int, key: str) -> np.ndarray:
        """Load one PNG → (3, H, W) float32 in [0,1]."""
        cache_key = (idx, key)
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.split_dir / f'{idx:06d}_{key}.png'
        img  = np.array(Image.open(path), dtype=np.float32) / 255.0
        arr  = img.transpose(2, 0, 1)   # (3, H, W)

        if self._cache is not None:
            self._cache[cache_key] = arr
        return arr

    def _augment(self, imgs: list) -> list:
        """
        Consistent augmentation across all subband images.
        Only transpose (time reversal) + tiny Gaussian noise.
        Both operations preserve recurrence structure.
        """
        do_T = np.random.rand() < 0.5
        do_N = np.random.rand() < 0.3
        out  = []
        for img in imgs:
            if do_T:
                img = img.transpose(0, 2, 1).copy()
            if do_N:
                img = np.clip(img + np.random.normal(0, 0.01, img.shape).astype(np.float32),
                              0.0, 1.0)
            out.append(img)
        return out

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        if self.mode == 'single':
            # Load single-scale image
            img = self._load_img(idx, 'single')
            if self.augment:
                img = self._augment([img])[0]
            return torch.from_numpy(img), int(self.labels[idx])
        else:
            # Load multi-scale images
            imgs = [self._load_img(idx, k) for k in self.subband_keys]
            if self.augment:
                imgs = self._augment(imgs)
            x = np.concatenate(imgs, axis=0).astype(np.float32)  # (21, H, W)
            return torch.from_numpy(x), int(self.labels[idx])

    def get_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class CombinedPNGDataset(Dataset):
    """
    Combines train + val PNG datasets into one pool for k-fold CV.
    Test split is deliberately excluded.
    """

    def __init__(self, png_dir: Path, mode: str = 'multi',
                 subband_keys=None, cache: bool = True):
        self.mode         = mode
        self.subband_keys = subband_keys or SUBBAND_KEYS if mode == 'multi' else []
        self.png_dir      = Path(png_dir)

        all_labels, all_rids, all_maps = [], [], []
        self.split_datasets = {}
        offset = 0

        for split in ('train', 'val'):    # ← test is NEVER included
            ds = PNGSubbandDataset(
                png_dir, split,
                mode=mode,
                subband_keys=self.subband_keys,
                augment=False,   # augment applied per-fold in training
                cache=cache,
            )
            self.split_datasets[split] = ds
            all_labels.extend(ds.labels.tolist())
            all_rids.extend(ds.record_ids.tolist())
            for i in range(len(ds)):
                all_maps.append((split, i))
            offset += len(ds)

        self.labels     = np.array(all_labels, dtype=np.int64)
        self.record_ids = np.array(all_rids)
        self.sample_map = all_maps

        n_paf = int((self.labels == 1).sum())
        n_nsr = int((self.labels == 0).sum())
        print(f"\nCombinedPNGDataset ({mode}, train+val): "
              f"{len(self.labels)} samples  NSR={n_nsr}  Pre-AF={n_paf}  "
              f"Patients={len(np.unique(self.record_ids))}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        split, local_idx = self.sample_map[idx]
        return self.split_datasets[split][local_idx]


def train_epoch(model, loader, criterion, optimizer, device) -> tuple:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n



# ── Add to imports ─────────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc as sklearn_auc
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Attention weight collection
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_attention_weights(model, loader, device) -> np.ndarray:
    """
    Collect mean attention weights across all batches in loader.
    Returns (n_subbands,) mean weight vector.
    Only call for multi-scale models.
    """
    model.eval()
    all_weights = []
    for x, _ in loader:
        x = x.to(device)
        # Extract branch features and compute attention
        feats   = [model.branches[i](x[:, i*3:(i+1)*3])
                   for i in range(model.n_subbands)]
        stacked = torch.stack(feats, dim=1)
        flat    = stacked.reshape(stacked.size(0), -1)
        weights = model.attention(flat)   # (B, n_subbands)
        all_weights.append(weights.cpu().numpy())
    return np.concatenate(all_weights, axis=0).mean(axis=0)   # (n_subbands,)


# ══════════════════════════════════════════════════════════════════════════════
# ROC curve plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(fold_probs_list: list, fold_labels_list: list,
                    title: str, save_path: Path,
                    colour: str = '#1f77b4') -> float:
    """
    Plot per-fold ROC curves with ±1 std shading and bold mean curve.

    Args:
        fold_probs_list:  list of (N,) prob arrays, one per fold
        fold_labels_list: list of (N,) label arrays, one per fold
        title:            plot title
        save_path:        where to save the figure
        colour:           line colour

    Returns:
        mean AUC across folds
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    fig, ax = plt.subplots(figsize=(6, 6))

    for probs, labels in zip(fold_probs_list, fold_labels_list):
        fpr, tpr, _ = roc_curve(labels, probs)
        fold_auc    = sklearn_auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr    = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, color=colour, alpha=0.15, lw=0.8)

    mean_tpr      = np.mean(tprs, axis=0)
    mean_tpr[-1]  = 1.0
    std_tpr       = np.std(tprs, axis=0)
    mean_auc      = float(np.mean(aucs))
    std_auc       = float(np.std(aucs))

    ax.plot(mean_fpr, mean_tpr, color=colour, lw=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color=colour, label='±1 std')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC saved → {save_path}")
    return mean_auc


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc  = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    auprc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    # Find optimal threshold using Youden's J statistic
    if len(np.unique(all_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        youdens_j = j_scores[optimal_idx]

        # Convert probabilities to predictions using optimal threshold
        all_preds_optimal = (all_probs >= optimal_threshold).astype(int)
    else:
        optimal_threshold = 0.5
        youdens_j = 0.0
        all_preds_optimal = all_preds

    # Compute confusion matrix using optimal threshold
    cm = confusion_matrix(all_labels, all_preds_optimal)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'auc_roc':     float(auc),
        'auprc':       float(auprc),
        'accuracy':    float(accuracy_score(all_labels, all_preds)),
        'f1':          float(f1_score(all_labels, all_preds, zero_division=0)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if tn + fp > 0 else 0.0,
        'precision':   float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall':      float(recall_score(all_labels, all_preds, zero_division=0)),
        'optimal_threshold': float(optimal_threshold),
        'youdens_j': float(youdens_j),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

def aggregate(fold_results: list) -> dict:
    keys = fold_results[0].keys()
    return {
        k: {
            'mean':   float(np.mean([f[k] for f in fold_results])),
            'std':    float(np.std( [f[k] for f in fold_results])),
            'values': [float(f[k]) for f in fold_results],
        }
        for k in keys
    }

def get_subject_group(record_id: str) -> int:
    """Map record → subject group ID (paired records share a group)."""
    m = re.match(r'n(\d+)', record_id)
    if m:
        return int(m.group(1))
    m = re.match(r'p(\d+)', record_id)
    if m:
        num = int(m.group(1))
        return 100 + num // 2   # p01/p02 → same group
    return -1

def run_kfold_cv(dataset: CombinedPNGDataset, model_type: str = 'multi') -> dict:
    groups   = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf      = GroupKFold(n_splits=N_FOLDS)
    dummy_X  = np.zeros(len(dataset))

    trained_results = []
    uniform_results = [] if model_type == 'multi' else None
    fold_probs_trained = []
    fold_labels_trained = []
    fold_probs_uniform = []
    fold_labels_uniform = []
    fold_attn_weights = []

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(dummy_X, dataset.labels, groups), start=1):

        print(f"\n{'='*65}")
        print(f"FOLD {fold}/{N_FOLDS}")
        print(f"{'='*65}")

        # Verify no patient overlap
        tr_pats = set(dataset.record_ids[tr_idx])
        va_pats = set(dataset.record_ids[va_idx])
        overlap = tr_pats & va_pats
        if overlap:
            raise RuntimeError(f"Patient overlap in fold {fold}: {overlap}")

        n_tr_paf = int((dataset.labels[tr_idx] == 1).sum())
        n_va_paf = int((dataset.labels[va_idx] == 1).sum())
        print(f"Train: {len(tr_idx)} segs  "
              f"({len(tr_pats)} patients, {n_tr_paf} Pre-AF segs)")
        print(f"Val:   {len(va_idx)} segs  "
              f"({len(va_pats)} patients, {n_va_paf} Pre-AF segs)")
        print("✓ No patient overlap")

        # Class-weighted loss — consistent with main training
        n_nsr   = int((dataset.labels[tr_idx] == 0).sum())
        n_preaf = n_tr_paf
        w_preaf = min(n_nsr / max(1, n_preaf), 2.0)
        cw      = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(DEVICE)
        print(f"Class weights: NSR=1.00  Pre-AF={w_preaf:.3f}")

        # Create subsets with augmentation only on training subset
        tr_subset = AugmentedSubset(dataset, tr_idx, augment=True)
        va_subset = Subset(dataset, va_idx)

        tr_loader = DataLoader(tr_subset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0,
                               pin_memory=DEVICE.type == 'cuda',
                               persistent_workers=False)
        va_loader = DataLoader(va_subset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0,
                               pin_memory=DEVICE.type == 'cuda',
                               persistent_workers=False)

        # Model — fresh initialisation every fold
        set_seed(SEED + fold)

        if model_type == 'single':
            model = SingleScaleCNN(DROPOUT).to(DEVICE)
        else:
            model = MultiScaleCNN(N_SUBBANDS, DROPOUT).to(DEVICE)

        if fold == 1:
            print(f"Parameters: {model.count_params():,}")

        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=SCHED_PAT, min_lr=1e-7,
        )

        best_auc    = 0.0
        best_state  = None
        no_improve  = 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, criterion, optimizer, DEVICE)
            va_met = evaluate(model, va_loader, DEVICE)
            va_auc = va_met['auc_roc']

            scheduler.step(va_auc)
            lr_now = optimizer.param_groups[0]['lr']

            print(f"  Epoch {epoch:3d}  "
                  f"tr={tr_loss:.4f}/{tr_acc:.3f}  "
                  f"va_auc={va_auc:.4f}  lr={lr_now:.1e}")

            if va_auc > best_auc:
                best_auc   = va_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
                ckpt_path  = SAVE_DIR / f'fold_{fold}_best.pt'
                torch.save({
                    'fold':        fold,
                    'epoch':       epoch,
                    'model_type':  model_type,
                    'state_dict':  best_state,
                    'val_auc':     best_auc,
                    'val_metrics': va_met,
                    'config': {
                        'n_subbands': N_SUBBANDS if model_type == 'multi' else None,
                        'dropout':    DROPOUT,
                        'lr':         LR,
                        'seed':       SEED,
                        'w_preaf':    float(w_preaf),
                    },
                }, ckpt_path)
                print(f"    → Best val AUC {best_auc:.4f}  (saved)")
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        model.eval()

        va_probs_trained, va_labels = [], []
        with torch.no_grad():
            for x, y in va_loader:
                logits = model(x.to(DEVICE))
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                va_probs_trained.extend(probs)
                va_labels.extend(y.numpy())

        va_probs_trained = np.array(va_probs_trained)
        va_labels = np.array(va_labels)

        m_trained = evaluate(model, va_loader, DEVICE)
        trained_results.append(m_trained)
        fold_probs_trained.append(va_probs_trained)
        fold_labels_trained.append(va_labels)

        # Attention weights (multi only)
        if model_type == 'multi':
            fold_attn_weights.append(
                collect_attention_weights(model, va_loader, DEVICE)
            )

            # Uniform attention
            model.set_uniform_attention()
            va_probs_uniform = []
            with torch.no_grad():
                for x, y in va_loader:
                    logits = model(x.to(DEVICE))
                    probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                    va_probs_uniform.extend(probs)

            va_probs_uniform = np.array(va_probs_uniform)
            m_uniform = evaluate(model, va_loader, DEVICE)
            uniform_results.append(m_uniform)
            fold_probs_uniform.append(va_probs_uniform)
            fold_labels_uniform.append(va_labels)

            print(f"\n  Fold {fold} results_harm:")
            print(f"    Trained  AUC={m_trained['auc_roc']:.4f}  "
                  f"Sens={m_trained['sensitivity']:.3f}  "
                  f"Spec={m_trained['specificity']:.3f}")
            print(f"    Uniform  AUC={m_uniform['auc_roc']:.4f}  "
                  f"Sens={m_uniform['sensitivity']:.3f}  "
                  f"Spec={m_uniform['specificity']:.3f}")
        else:
            print(f"\n  Fold {fold} AUC={m_trained['auc_roc']:.4f}")

    # ── Aggregate across folds ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("K-FOLD CV SUMMARY")
    print(f"{'='*65}")

    agg_trained = aggregate(trained_results)

    results = {
        'config': {
            'model_type': model_type,
            'n_folds': N_FOLDS,
            'n_subbands': N_SUBBANDS if model_type == 'multi' else None,
            'epochs': EPOCHS, 'patience': PATIENCE,
            'lr': LR, 'dropout': DROPOUT, 'seed': SEED,
        },
        'trained_attention' if model_type == 'multi' else 'results_harm': {
            'aggregated': agg_trained,
            'per_fold':   trained_results,
        },
    }

    if model_type == 'multi':
        agg_uniform = aggregate(uniform_results)
        results['uniform_attention'] = {
            'aggregated': agg_uniform,
            'per_fold':   uniform_results,
        }

        for mode, agg in [('Trained', agg_trained), ('Uniform', agg_uniform)]:
            print(f"\n{mode} attention:")
            for k in ('auc_roc', 'auprc', 'sensitivity', 'specificity', 'f1'):
                print(f"  {k:<15} {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
    else:
        print(f"\n{model_type.upper()} results_harm:")
        for k in ('auc_roc', 'auprc', 'sensitivity', 'specificity', 'f1'):
            print(f"  {k:<15} {agg_trained[k]['mean']:.4f} ± {agg_trained[k]['std']:.4f}")

    out_path = SAVE_DIR / 'kfold_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")

    subband_names = SUBBAND_KEYS[:N_SUBBANDS]

    if model_type == 'multi':
        plot_roc_curves(
            fold_probs_trained, fold_labels_trained,
            title=f'Multi-scale CNN — Trained Attention\n10-fold CV ROC',
            save_path=SAVE_DIR / 'roc_trained.png',
            colour='#d62728',
        )
        plot_roc_curves(
            fold_probs_uniform, fold_labels_uniform,
            title=f'Multi-scale CNN — Uniform Attention\n10-fold CV ROC',
            save_path=SAVE_DIR / 'roc_uniform.png',
            colour='#ff7f0e',
        )

        # Save raw attention weights for ablation subband selection
        attn_array = np.array(fold_attn_weights)  # (n_folds, n_subbands)
        np.save(SAVE_DIR / 'fold_attention_weights.npy', attn_array)
        print(f"\nAttention weights saved → "
              f"{SAVE_DIR / 'fold_attention_weights.npy'}")
    else:
        plot_roc_curves(
            fold_probs_trained, fold_labels_trained,
            title=f'Single-scale CNN\n10-fold CV ROC',
            save_path=SAVE_DIR / 'roc_single.png',
            colour='#1f77b4',
        )
    return results

class AugmentedSubset(Dataset):
    def __init__(self, dataset: CombinedPNGDataset,
                 indices: np.ndarray, augment: bool = True):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        split, local_idx = self.dataset.sample_map[self.indices[idx]]
        split_ds = self.dataset.split_datasets[split]

        if split_ds.mode == 'single':
            # Load single-scale image
            img = split_ds._load_img(local_idx, 'single')
            if self.augment:
                img = split_ds._augment([img])[0]
            x = img
        else:
            # Load multi-scale images
            imgs = [split_ds._load_img(local_idx, k)
                    for k in split_ds.subband_keys]
            if self.augment:
                imgs = split_ds._augment(imgs)
            x = np.concatenate(imgs, axis=0).astype(np.float32)

        label = int(self.dataset.labels[self.indices[idx]])
        return torch.from_numpy(x.astype(np.float32)), label


def run_ablation_study(png_dir: Path, base_save_dir: Path) -> dict:
    """
    Run ablation study on multi-scale CNN with three different subband configurations.

    Uses existing PNG files (e.g., 000000_D2.png, 000000_D3.png, etc.) and filters
    which subbands are loaded for each configuration:

    1. Lower frequencies: D2, D3, D4
    2. Higher frequencies: D5, D6, D7, D8
    3. Selected frequencies: D2, D4, D7

    Each configuration is trained with 10-fold cross-validation and saved to a separate folder.

    Args:
        png_dir: Directory containing existing PNG images (000000_D2.png, etc.)
        base_save_dir: Base directory for saving results (e.g., results_kfold2/ablation)

    Returns:
        dict: Combined results from all three configurations
    """

    # Define three ablation configurations
    ablation_configs = {
        'lower_freq': {
            'subbands': ['D2', 'D3', 'D4'],
            'description': 'Lower frequencies (D2, D3, D4)',
            'folder': 'lower_freq_d234'
        },
        'higher_freq': {
            'subbands': ['D5', 'D6', 'D7', 'D8'],
            'description': 'Higher frequencies (D5, D6, D7, D8)',
            'folder': 'higher_freq_d5678'
        },
        'selected_freq': {
            'subbands': ['D2', 'D4', 'D7'],
            'description': 'Selected frequencies (D2, D4, D7)',
            'folder': 'selected_freq_d247'
        }
    }

    all_results = {}

    print("\n" + "="*80)
    print("ABLATION STUDY - Multi-scale CNN with Different Subband Configurations")
    print("="*80)
    print(f"Using existing PNG files from: {png_dir}")

    for config_name, config in ablation_configs.items():
        print(f"\n{'='*80}")
        print(f"Configuration: {config['description']}")
        print(f"Subbands: {config['subbands']}")
        print(f"{'='*80}")

        # Filter existing PNG files to load only specified subbands
        # E.g., for ['D2', 'D3', 'D4'], only loads 000000_D2.png, 000000_D3.png, 000000_D4.png
        dataset = CombinedPNGDataset(
            png_dir,
            mode='multi',
            subband_keys=config['subbands'],  # Filters which PNGs to load
            cache=True
        )

        # Set up save directory for this configuration
        config_save_dir = base_save_dir / config['folder']
        config_save_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily override global SAVE_DIR
        global SAVE_DIR
        original_save_dir = SAVE_DIR
        SAVE_DIR = config_save_dir

        # Update N_SUBBANDS for the model
        global N_SUBBANDS
        original_n_subbands = N_SUBBANDS
        N_SUBBANDS = len(config['subbands'])

        print(f"\nSave directory: {config_save_dir}")
        print(f"Number of subbands: {N_SUBBANDS}")
        print(f"Total samples: {len(dataset)}")

        # Run k-fold cross-validation
        results = run_kfold_cv(dataset, model_type='multi')

        # Add configuration metadata to results
        results['ablation_config'] = {
            'name': config_name,
            'description': config['description'],
            'subbands': config['subbands'],
            'n_subbands': len(config['subbands'])
        }

        # Save configuration-specific results
        config_results_path = config_save_dir / 'ablation_results.json'
        with open(config_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Configuration results saved → {config_results_path}")

        # Store in combined results
        all_results[config_name] = results

        # Restore global settings
        SAVE_DIR = original_save_dir
        N_SUBBANDS = original_n_subbands

    # Save combined results
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)

    combined_results_path = base_save_dir / 'ablation_study_combined.json'
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined ablation results saved → {combined_results_path}")

    # Print summary comparison
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    for config_name, config in ablation_configs.items():
        results = all_results[config_name]
        agg = results['trained_attention']['aggregated']

        print(f"\n{config['description']}")
        print(f"  Subbands: {', '.join(config['subbands'])}")
        print(f"  AUC-ROC:  {agg['auc_roc']['mean']:.4f} ± {agg['auc_roc']['std']:.4f}")
        print(f"  AUPRC:    {agg['auprc']['mean']:.4f} ± {agg['auprc']['std']:.4f}")
        print(f"  Sens:     {agg['sensitivity']['mean']:.4f} ± {agg['sensitivity']['std']:.4f}")
        print(f"  Spec:     {agg['specificity']['mean']:.4f} ± {agg['specificity']['std']:.4f}")
        print(f"  F1:       {agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}")

    return all_results


def plot_ablation_comparison(ablation_base_dir: Path, full_model_dir: Path,
                             metric: str = 'auc_roc', save_path: Path = None):
    """
    Create a bar chart comparing ablation configurations with the full multi-scale model.

    Args:
        ablation_base_dir: Directory containing ablation study results
        full_model_dir: Directory containing full multi-scale model results
        metric: Metric to plot ('auc_roc', 'auprc', 'sensitivity', 'specificity', 'f1')
        save_path: Path to save the plot (if None, saves to ablation_base_dir)

    Returns:
        dict: Comparison data
    """
    import matplotlib.pyplot as plt

    # Metric display names
    metric_names = {
        'auc_roc': 'AUC-ROC',
        'auprc': 'AUPRC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'f1': 'F1-Score'
    }

    # Load ablation results
    ablation_file = ablation_base_dir / 'ablation_study_combined.json'
    if not ablation_file.exists():
        raise FileNotFoundError(f"Ablation results not found: {ablation_file}")

    with open(ablation_file, 'r') as f:
        ablation_data = json.load(f)

    # Load full model results
    full_model_file = full_model_dir / 'kfold_results.json'
    if not full_model_file.exists():
        raise FileNotFoundError(f"Full model results not found: {full_model_file}")

    with open(full_model_file, 'r') as f:
        full_model_data = json.load(f)

    # Extract metric values for each configuration
    configs = {
        'Lower Freq\n(D2, D3, D4)': {
            'data': ablation_data['lower_freq'],
            'color': '#1f77b4'
        },
        'Higher Freq\n(D5, D6, D7, D8)': {
            'data': ablation_data['higher_freq'],
            'color': '#ff7f0e'
        },
        'Selected Freq\n(D2, D4, D7)': {
            'data': ablation_data['selected_freq'],
            'color': '#2ca02c'
        },
        'Full Multi-scale\n(All 7 subbands)': {
            'data': full_model_data,
            'color': '#d62728'
        }
    }

    # Extract means and stds
    config_names = []
    means = []
    stds = []
    colors = []

    for name, info in configs.items():
        config_names.append(name)
        colors.append(info['color'])

        # Get the aggregated results (trained attention)
        agg = info['data']['trained_attention']['aggregated']
        means.append(agg[metric]['mean'])
        stds.append(agg[metric]['std'])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(config_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.8,
                   color=colors, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric_names.get(metric, metric)}', fontsize=14, fontweight='bold')
    ax.set_title(f'Ablation Study: {metric_names.get(metric, metric)} Comparison\n10-Fold Cross-Validation',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add horizontal line at baseline (full model)
    full_model_mean = means[-1]
    ax.axhline(y=full_model_mean, color='red', linestyle='--', linewidth=2,
               alpha=0.5, label=f'Full Model Baseline ({full_model_mean:.4f})')
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()

    # Save plot
    if save_path is None:
        save_path = ablation_base_dir / f'ablation_comparison_{metric}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Comparison plot saved → {save_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY COMPARISON - {metric_names.get(metric, metric)}")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Mean':<12} {'Std':<12} {'Diff from Full':<15}")
    print(f"{'-'*80}")

    for i, (name, mean, std) in enumerate(zip(config_names, means, stds)):
        diff = mean - means[-1]  # Difference from full model
        diff_str = f"{diff:+.4f}" if i < len(config_names) - 1 else "baseline"
        name_clean = name.replace('\n', ' ')
        print(f"{name_clean:<30} {mean:.4f}      ±{std:.4f}      {diff_str:<15}")

    # Return comparison data
    comparison_data = {
        'metric': metric,
        'configurations': {
            name.replace('\n', ' '): {
                'mean': float(mean),
                'std': float(std),
                'diff_from_full': float(mean - means[-1]) if i < len(config_names) - 1 else 0.0
            }
            for i, (name, mean, std) in enumerate(zip(config_names, means, stds))
        }
    }

    return comparison_data


def plot_all_ablation_metrics(ablation_base_dir: Path, full_model_dir: Path):
    """
    Generate comparison plots for all metrics (AUC-ROC, AUPRC, Sensitivity, Specificity, F1).

    Args:
        ablation_base_dir: Directory containing ablation study results
        full_model_dir: Directory containing full multi-scale model results

    Returns:
        dict: Comparison data for all metrics
    """
    metrics = ['auc_roc', 'auprc', 'sensitivity', 'specificity', 'f1']
    all_comparisons = {}

    print("\n" + "="*80)
    print("GENERATING ABLATION COMPARISON PLOTS FOR ALL METRICS")
    print("="*80)

    for metric in metrics:
        print(f"\nGenerating plot for {metric}...")
        comparison = plot_ablation_comparison(ablation_base_dir, full_model_dir, metric)
        all_comparisons[metric] = comparison

    # Save combined comparison data
    output_path = ablation_base_dir / 'all_metrics_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\n✓ All metrics comparison data saved → {output_path}")

    return all_comparisons


def plot_ablation_boxplot(ablation_base_dir: Path, full_model_dir: Path,
                          single_model_dir: Path, metric: str = 'auc_roc'):
    """
    Create boxplot showing per-fold metric distribution for all 5 configurations.

    Configurations (left to right):
    1. Single-scale (anchor/baseline)
    2. Lower freq (D2, D3, D4)
    3. Higher freq (D5, D6, D7, D8)
    4. Selected freq (D2, D4, D7)
    5. Full multi-scale (all 7 subbands - anchor/best)

    Args:
        ablation_base_dir: Directory containing ablation study results
        full_model_dir: Directory containing full multi-scale model results
        single_model_dir: Directory containing single-scale model results
        metric: Metric to plot (default: 'auc_roc')

    Returns:
        dict: Boxplot data for all configurations
    """
    import matplotlib.pyplot as plt

    # Metric display names
    metric_names = {
        'auc_roc': 'AUC-ROC',
        'auprc': 'AUPRC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'f1': 'F1-Score'
    }

    print(f"\n{'='*80}")
    print(f"GENERATING ABLATION BOXPLOT FOR {metric_names.get(metric, metric)}")
    print(f"{'='*80}")

    # Load ablation results
    ablation_file = ablation_base_dir / 'ablation_study_combined.json'
    if not ablation_file.exists():
        raise FileNotFoundError(f"Ablation results not found: {ablation_file}")

    with open(ablation_file, 'r') as f:
        ablation_data = json.load(f)

    # Load full multi-scale model results
    full_model_file = full_model_dir / 'kfold_results.json'
    if not full_model_file.exists():
        raise FileNotFoundError(f"Full model results not found: {full_model_file}")

    with open(full_model_file, 'r') as f:
        full_model_data = json.load(f)

    # Load single-scale model results
    single_model_file = single_model_dir / 'kfold_results.json'
    if not single_model_file.exists():
        raise FileNotFoundError(f"Single model results not found: {single_model_file}")

    with open(single_model_file, 'r') as f:
        single_model_data = json.load(f)

    # Extract per-fold values for each configuration (trained attention only)
    configs = [
        {
            'name': 'Single-scale',
            'label': 'Single-scale\n(baseline)',
            'values': single_model_data['results']['aggregated'][metric]['values'],
            'color': '#7f7f7f'  # Gray
        },
        {
            'name': 'Higher Freq (D2-D4)',
            'label': 'Higher Freq\n(D2, D3, D4)',
            'values': ablation_data['lower_freq']['trained_attention']['aggregated'][metric]['values'],
            'color': '#1f77b4'  # Blue
        },
        {
            'name': 'Lower Freq (D5-D8)',
            'label': 'Lower Freq\n(D5, D6, D7, D8)',
            'values': ablation_data['higher_freq']['trained_attention']['aggregated'][metric]['values'],
            'color': '#ff7f0e'  # Orange
        },
        {
            'name': 'Selected Freq (D2,D4,D7)',
            'label': 'Selected Freq\n(D2, D4, D7)',
            'values': ablation_data['selected_freq']['trained_attention']['aggregated'][metric]['values'],
            'color': '#2ca02c'  # Green
        },
        {
            'name': 'Full Multi-scale',
            'label': 'Full Multi-scale\n(All 7 subbands)',
            'values': full_model_data['trained_attention']['aggregated'][metric]['values'],
            'color': '#d62728'  # Red
        }
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data for boxplot
    data_to_plot = [config['values'] for config in configs]
    labels = [config['label'] for config in configs]
    colors = [config['color'] for config in configs]

    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                  markeredgecolor='black', markersize=8),
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_ylabel(f'{metric_names.get(metric, metric)}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
    ax.set_title(f'Ablation Study: Per-Fold {metric_names.get(metric, metric)} Distribution\n'
                f'10-Fold Cross-Validation (Trained Attention)',
                fontsize=15, fontweight='bold', pad=20)

    # Rotate x-axis labels for better readability
    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11)

    # Add legend explaining boxplot elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Box: IQR (25th-75th percentile)'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Line: Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
                  markeredgecolor='black', markersize=8, label='Diamond: Mean')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

    plt.tight_layout()

    # Save plot
    output_path_pdf = ablation_base_dir / f'ablation_boxplot_{metric}.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"\n✓ Boxplot saved (PDF) → {output_path_pdf}")

    output_path_png = ablation_base_dir / f'ablation_boxplot_{metric}.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✓ Boxplot saved (PNG) → {output_path_png}")

    plt.close()

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS - {metric_names.get(metric, metric)}")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*80}")

    for config in configs:
        values = np.array(config['values'])
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        name = config['name']
        print(f"{name:<30} {mean_val:<10.4f} {median_val:<10.4f} {std_val:<10.4f} "
              f"{min_val:<10.4f} {max_val:<10.4f}")

    print(f"{'='*80}\n")

    # Return data
    boxplot_data = {
        'metric': metric,
        'configurations': {
            config['name']: {
                'values': config['values'],
                'mean': float(np.mean(config['values'])),
                'median': float(np.median(config['values'])),
                'std': float(np.std(config['values'])),
                'min': float(np.min(config['values'])),
                'max': float(np.max(config['values']))
            }
            for config in configs
        }
    }

    # Save boxplot data
    output_data_path = ablation_base_dir / f'ablation_boxplot_{metric}_data.json'
    with open(output_data_path, 'w') as f:
        json.dump(boxplot_data, f, indent=2)
    print(f"✓ Boxplot data saved → {output_data_path}\n")

    return boxplot_data


def perform_ablation_statistical_tests(ablation_base_dir: Path, full_model_dir: Path,
                                       metric: str = 'auc_roc'):
    """
    Perform pairwise Wilcoxon signed-rank tests between the four multi-scale configurations:
    - Full (all 7 subbands: D2-D8)
    - D2-D4 (higher frequency)
    - D5-D8 (lower frequency)
    - D2, D4, D7 (selected frequency)

    Args:
        ablation_base_dir: Directory containing ablation study results
        full_model_dir: Directory containing full multi-scale model results
        metric: Metric to test (default: 'auc_roc')

    Returns:
        dict: Statistical test results with p-values and effect sizes
    """
    # Metric display names
    metric_names = {
        'auc_roc': 'AUC-ROC',
        'auprc': 'AUPRC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'f1': 'F1-Score'
    }

    print(f"\n{'='*80}")
    print(f"ABLATION STUDY: STATISTICAL COMPARISON - {metric_names.get(metric, metric)}")
    print(f"{'='*80}")

    # Load ablation results
    ablation_file = ablation_base_dir / 'ablation_study_combined.json'
    if not ablation_file.exists():
        raise FileNotFoundError(f"Ablation results not found: {ablation_file}")

    with open(ablation_file, 'r') as f:
        ablation_data = json.load(f)

    # Load full multi-scale model results
    full_model_file = full_model_dir / 'kfold_results.json'
    if not full_model_file.exists():
        raise FileNotFoundError(f"Full model results not found: {full_model_file}")

    with open(full_model_file, 'r') as f:
        full_model_data = json.load(f)

    # Extract per-fold values for the four configurations
    configs = {
        'Full (D2-D8)': np.array(full_model_data['trained_attention']['aggregated'][metric]['values']),
        'D2-D4': np.array(ablation_data['lower_freq']['trained_attention']['aggregated'][metric]['values']),
        'D5-D8': np.array(ablation_data['higher_freq']['trained_attention']['aggregated'][metric]['values']),
        'D2,D4,D7': np.array(ablation_data['selected_freq']['trained_attention']['aggregated'][metric]['values'])
    }

    # Helper function to compute Cohen's d
    def cohens_d(group1, group2):
        """Compute Cohen's d effect size."""
        diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        return diff / pooled_std if pooled_std > 0 else 0.0

    # Perform pairwise comparisons
    config_names = list(configs.keys())
    results = {
        'metric': metric,
        'display_name': metric_names.get(metric, metric),
        'configurations': {
            name: {
                'values': configs[name].tolist(),
                'mean': float(np.mean(configs[name])),
                'std': float(np.std(configs[name], ddof=1)),
                'median': float(np.median(configs[name]))
            }
            for name in config_names
        },
        'pairwise_comparisons': {}
    }

    print(f"\nConfiguration Summary:")
    print(f"{'Configuration':<20} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print(f"{'-'*60}")
    for name in config_names:
        mean_val = results['configurations'][name]['mean']
        std_val = results['configurations'][name]['std']
        median_val = results['configurations'][name]['median']
        print(f"{name:<20} {mean_val:<12.4f} {std_val:<12.4f} {median_val:<12.4f}")

    print(f"\n{'='*80}")
    print("PAIRWISE STATISTICAL TESTS (Wilcoxon Signed-Rank Test)")
    print(f"{'='*80}")
    print(f"{'Comparison':<35} {'Δ Mean':<12} {'p-value':<12} {'Cohen d':<12} {'Sig.'}")
    print(f"{'-'*80}")

    # Perform all pairwise comparisons
    for i, name1 in enumerate(config_names):
        for name2 in config_names[i+1:]:
            values1 = configs[name1]
            values2 = configs[name2]

            # Wilcoxon signed-rank test (paired, non-parametric)
            statistic, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')

            # Cohen's d effect size
            effect_size = cohens_d(values1, values2)

            # Mean difference
            mean_diff = np.mean(values1) - np.mean(values2)

            # Determine significance
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'

            # Store results
            comparison_key = f"{name1} vs {name2}"
            results['pairwise_comparisons'][comparison_key] = {
                'config1': name1,
                'config2': name2,
                'config1_mean': float(np.mean(values1)),
                'config2_mean': float(np.mean(values2)),
                'mean_difference': float(mean_diff),
                'wilcoxon': {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significance': significance
                },
                'cohens_d': float(effect_size),
                'interpretation': (
                    'negligible' if abs(effect_size) < 0.2 else
                    'small' if abs(effect_size) < 0.5 else
                    'medium' if abs(effect_size) < 0.8 else
                    'large'
                )
            }

            # Print row
            print(f"{comparison_key:<35} {mean_diff:<+12.4f} {p_value:<12.4f} "
                  f"{effect_size:<+12.4f} {significance}")

    print(f"{'-'*80}")
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Cohen's d interpretation: <0.2 (negligible), 0.2-0.5 (small), "
          "0.5-0.8 (medium), >0.8 (large)")
    print(f"{'='*80}\n")

    # Save results to JSON
    output_path = ablation_base_dir / f'ablation_statistical_tests_{metric}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Statistical test results saved → {output_path}\n")

    return results


if __name__ == '__main__':
    extract_h5_to_png(H5_DIR, PNG_DIR, splits=('train', 'val'))
    dataset = CombinedPNGDataset(PNG_DIR, mode=MODEL_TYPE, cache=True)
    results = run_kfold_cv(dataset, model_type=MODEL_TYPE)

    # Run ablation study
    # extract_h5_to_png(H5_DIR, PNG_DIR, splits=('train', 'val'))
    # ablation_base_dir = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/ablation')
    # ablation_results = run_ablation_study(PNG_DIR, ablation_base_dir)

    # Generate comparison plots
    full_model_dir = Path('/results_final/results_kfold2/multi')
    single_model_dir = Path('/results_final/results_kfold2/single')

    # Bar chart comparisons for all metrics
    # plot_all_ablation_metrics(ablation_base_dir, full_model_dir)

    # Boxplot showing per-fold distribution (AUC-ROC by default)
    # plot_ablation_boxplot(ablation_base_dir, full_model_dir, single_model_dir, metric='auc_roc')

    # Perform statistical tests between the four multi-scale configurations for all metrics
    # metrics_to_test = ['auc_roc', 'auprc', 'sensitivity', 'specificity']
    # for metric in metrics_to_test:
    #     perform_ablation_statistical_tests(ablation_base_dir, full_model_dir, metric=metric)