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
    confusion_matrix,
)
from PIL import Image
from tqdm import tqdm

from src.CNN.models.models import SingleScaleCNN, MultiScaleCNN

H5_DIR      = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_final/utrp_dataset_db8')
PNG_DIR     = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2')
SAVE_DIR    = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2')

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


def plot_attention_weights(fold_weights: list,
                            subband_names: list,
                            save_path: Path) -> None:
    """
    Plot mean ± std attention weights across all folds as a bar chart.

    Args:
        fold_weights:   list of (n_subbands,) arrays, one per fold
        subband_names:  ['D2','D3',...,'D8']
        save_path:      where to save
    """
    weights  = np.array(fold_weights)        # (n_folds, n_subbands)
    means    = weights.mean(axis=0)
    stds     = weights.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(subband_names))
    ax.bar(x, means, yerr=stds, capsize=5,
           color='steelblue', alpha=0.8, edgecolor='navy')
    ax.set_xticks(x)
    ax.set_xticklabels(subband_names, fontsize=11)
    ax.set_xlabel('Subband', fontsize=12)
    ax.set_ylabel('Mean attention weight (sigmoid)', fontsize=12)
    ax.set_title('Mean attention weights across CV folds\n'
                 '(mean ± std, 10 folds)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.2,
               label='Uniform baseline (0.5)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Attention weights saved → {save_path}")

    print("\nMean attention weights across folds:")
    print(f"  {'Subband':<8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*26}")
    for name, m, s in zip(subband_names, means, stds):
        print(f"  {name:<8} {m:>8.4f} {s:>8.4f}")

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
    cm   = confusion_matrix(all_labels, all_preds)
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


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — K-Fold CV
# ══════════════════════════════════════════════════════════════════════════════

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

            print(f"\n  Fold {fold} results:")
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
        'trained_attention' if model_type == 'multi' else 'results': {
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
        print(f"\n{model_type.upper()} results:")
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
        plot_attention_weights(
            fold_attn_weights, subband_names,
            save_path=SAVE_DIR / 'attention_weights_cv.png',
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


if __name__ == '__main__':
    extract_h5_to_png(H5_DIR, PNG_DIR, splits=('train', 'val'))
    dataset = CombinedPNGDataset(PNG_DIR, mode=MODEL_TYPE, cache=True)
    results = run_kfold_cv(dataset, model_type=MODEL_TYPE)