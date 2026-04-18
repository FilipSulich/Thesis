"""
Pre-AF classification training script.

Two modes:
  CNN    — SingleScaleNet / MultiScaleAttentionNet / HybridAttentionNet
  XGBoost — RQA-inspired features from UTRPs (single / multi / hybrid)

Usage:
  python train.py --mode cnn --arch single --hdf5 /path/to/data.h5 --split /path/to/split.json --results_dir /path/out
  python train.py --mode xgb --arch hybrid --hdf5 /path/to/data.h5 --split /path/to/split.json --results_dir /path/out
"""

import argparse
import json
import warnings
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — XGB mode unavailable")

warnings.filterwarnings('ignore')

DEVICE = (torch.device('mps') if torch.backends.mps.is_available()
          else torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class UTRPDataset(Dataset):
    def __init__(self, X_single, y, X_multi=None, augment=False):
        self.X_single = X_single.astype(np.float32)
        self.X_multi  = X_multi.astype(np.float32) if X_multi is not None else None
        self.y        = y.astype(np.int64)
        self.augment  = augment

    def __len__(self):
        return len(self.y)

    def _augment(self, x):
        C, H, W = x.shape
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1].copy()
        if np.random.rand() > 0.5:
            x = x.transpose(0, 2, 1).copy()
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.75, 1.0)
            crop  = int(H * scale)
            px    = np.random.randint(0, H - crop + 1)
            py    = np.random.randint(0, W - crop + 1)
            x = np.stack([cv2.resize(x[c, px:px+crop, py:py+crop], (H, W),
                           interpolation=cv2.INTER_LINEAR) for c in range(C)]).astype(np.float32)
        if np.random.rand() > 0.7:
            x = np.clip(x + np.random.normal(0, 0.01, x.shape).astype(np.float32), 0., 1.)
        return x

    def __getitem__(self, idx):
        ss = self.X_single[idx].copy()
        if self.augment:
            ss = self._augment(ss)
        ss = torch.from_numpy(ss)
        y  = torch.tensor(self.y[idx], dtype=torch.long)
        if self.X_multi is not None:
            ms = self.X_multi[idx].copy()
            if self.augment:
                ms = self._augment(ms)
            return ss, torch.from_numpy(ms), y
        return ss, y
"""
True Recurrence Quantification Analysis (RQA) feature extraction.

Standard RQA measures from Webber & Zbilut (1994) and Marwan et al. (2007).

Features:
  RR    Recurrence Rate
  DET   Determinism
  Lmean Mean diagonal line length
  Lmax  Longest diagonal line
  DIV   Divergence = 1/Lmax
  ENT   Shannon entropy of diagonal line lengths
  LAM   Laminarity
  TT    Trapping Time (mean vertical line length)
  Vmax  Longest vertical line
  W_ENT Entropy of white vertical lines (recurrence time entropy)
  T1    Mean recurrence time (type 1)
  T2    Mean recurrence time (type 2 - first return)
"""

import numpy as np
from tqdm import tqdm

FEATURE_NAMES = ['RR', 'DET', 'Lmean', 'Lmax', 'DIV', 'ENT',
                 'LAM', 'TT', 'Vmax', 'W_ENT', 'T1', 'T2']


def _diagonal_line_lengths(rp, lmin=2):
    """Lengths of diagonal lines (parallel to LOI) in upper triangle, lmin+ only."""
    N = rp.shape[0]
    lengths = []
    for k in range(1, N):
        run = 0
        for val in np.diag(rp, k):
            if val:
                run += 1
            else:
                if run >= lmin:
                    lengths.append(run)
                run = 0
        if run >= lmin:
            lengths.append(run)
    return np.array(lengths, dtype=np.int32)


def _vertical_line_lengths(rp, vmin=2):
    """Lengths of vertical lines (consecutive 1s in each column), vmin+ only."""
    N = rp.shape[0]
    lengths = []
    for j in range(N):
        run = 0
        for val in rp[:, j]:
            if val:
                run += 1
            else:
                if run >= vmin:
                    lengths.append(run)
                run = 0
        if run >= vmin:
            lengths.append(run)
    return np.array(lengths, dtype=np.int32)


def _white_vertical_line_lengths(rp, wmin=1):
    """Lengths of white vertical lines (consecutive 0s between 1s in each column)."""
    N = rp.shape[0]
    lengths = []
    for j in range(N):
        col = rp[:, j]
        run = 0
        in_white = False
        for val in col:
            if not val:
                run += 1
                in_white = True
            else:
                if in_white and run >= wmin:
                    lengths.append(run)
                run = 0
                in_white = False
    return np.array(lengths, dtype=np.int32)


def extract_rqa(rp, lmin=2, vmin=2):
    """
    Compute 12 RQA measures from a binary recurrence plot.

    Args:
        rp:   (N, N) binary matrix (0/1 or bool)
        lmin: minimum diagonal line length (Webber & Zbilut recommend 2)
        vmin: minimum vertical line length

    Returns:
        (12,) float64 — [RR, DET, Lmean, Lmax, DIV, ENT, LAM, TT, Vmax, W_ENT, T1, T2]
    """
    rp = (rp > 0.5).astype(np.int32)
    N  = rp.shape[0]

    # Recurrence Rate — upper triangle only, exclude LOI
    idx   = np.triu_indices(N, k=1)
    upper = rp[idx]
    RR    = float(upper.mean()) if len(upper) > 0 else 0.0

    # Diagonal lines
    d_len = _diagonal_line_lengths(rp, lmin)
    n_d   = int(d_len.sum()) if len(d_len) > 0 else 0
    total = int(upper.sum())

    DET   = float(n_d / total) if total > 0 else 0.0
    DET   = min(DET, 1.0)
    Lmean = float(d_len.mean()) if len(d_len) > 0 else 0.0
    Lmax  = float(d_len.max())  if len(d_len) > 0 else 0.0
    DIV   = 1.0 / Lmax if Lmax > 0 else 1.0

    if len(d_len) > 0:
        vals, counts = np.unique(d_len, return_counts=True)
        p   = counts / counts.sum()
        ENT = float(-np.sum(p * np.log2(p + 1e-12)))
    else:
        ENT = 0.0

    # Vertical lines
    v_len  = _vertical_line_lengths(rp, vmin)
    n_v    = int(v_len.sum()) if len(v_len) > 0 else 0
    rp_nd  = rp.copy(); np.fill_diagonal(rp_nd, 0)
    total_all = int(rp_nd.sum())

    LAM  = float(n_v / total_all) if total_all > 0 else 0.0
    LAM  = min(LAM, 1.0)
    TT   = float(v_len.mean()) if len(v_len) > 0 else 0.0
    Vmax = float(v_len.max())  if len(v_len) > 0 else 0.0

    # White vertical lines (recurrence times)
    w_len = _white_vertical_line_lengths(rp, wmin=1)
    if len(w_len) > 0:
        vals, counts = np.unique(w_len, return_counts=True)
        p     = counts / counts.sum()
        W_ENT = float(-np.sum(p * np.log2(p + 1e-12)))
    else:
        W_ENT = 0.0
    T1 = float(w_len.mean()) if len(w_len) > 0 else 0.0

    # T2 — first recurrence time per column
    T2_vals = []
    for j in range(N):
        run = 0
        for val in rp[:, j]:
            if not val:
                run += 1
            else:
                if run > 0:
                    T2_vals.append(run)
                    break
                run = 0
    T2 = float(np.mean(T2_vals)) if T2_vals else 0.0

    return np.array([RR, DET, Lmean, Lmax, DIV, ENT, LAM, TT, Vmax, W_ENT, T1, T2],
                    dtype=np.float64)

def rp_from_stored_utrp(utrp_rgb, epsilon_frac=0.1):
    if utrp_rgb.ndim == 3:
        blue = utrp_rgb[2]   # (H, W) — high where distance is low
    else:
        blue = utrp_rgb

    # Recurrent points have blue channel above the (1 - epsilon_frac) quantile
    # i.e. the top epsilon_frac fraction of blue values are recurrent
    threshold = np.percentile(blue[blue > 0], (1 - epsilon_frac) * 100)
    return (blue >= threshold).astype(np.float32)


# def extract_rqa_from_rgb(utrp_rgb, epsilon_frac=0.1, lmin=2, vmin=2):
#     """Extract 12 RQA features from a stored (3, H, W) UTRP RGB image."""
#     binary = rp_from_stored_utrp(utrp_rgb, epsilon_frac)
#     return extract_rqa(binary, lmin=lmin, vmin=vmin)


def build_rqa_feature_matrix(ss, ms=None, arch='single',
                              subband_names=None,
                              epsilon_frac=0.1, lmin=2, vmin=2):
    """
    Build RQA feature matrix from stored UTRP arrays.

    Each channel produces 12 RQA features.
    Total features: 12 (single) / 12*n_sub (multi) / 12*(1+n_sub) (hybrid)

    Args:
        ss:   (N, 3, H, W) single-scale RGB
        ms:   (N, n_sub*3, H, W) multi-scale RGB
        arch: 'single', 'multi', or 'hybrid'
    """
    N = ss.shape[0]

    def batch_rqa(data, desc):
        return np.array([extract_rqa_from_rgb(data[i], epsilon_frac, lmin, vmin)for i in tqdm(range(N), desc=desc)])
        # return np.array([extract_rqa(data[i], lmin, vmin)for i in tqdm(range(N), desc=desc)])

    print(f"\nRQA feature extraction: arch={arch}  ε={epsilon_frac}  lmin={lmin}")
    print(f"12 features per channel: {FEATURE_NAMES}")

    if arch == 'single':
        X = batch_rqa(ss, 'SS')
        print(f"Feature matrix: {X.shape}")
        return X

    n_sub = ms.shape[1] // 3
    ms_blocks = []
    for s in range(n_sub):
        name = subband_names[s] if subband_names else f'sub{s}'
        ms_blocks.append(batch_rqa(ms[:, s*3:(s+1)*3], name))
    ms_matrix = np.concatenate(ms_blocks, axis=1)

    if arch == 'multi':
        print(f"Feature matrix: {ms_matrix.shape}")
        return ms_matrix

    ss_feats = batch_rqa(ss, 'SS')
    X = np.concatenate([ss_feats, ms_matrix], axis=1)
    print(f"Feature matrix: {X.shape}")
    return X


def load_data(hdf5_path, split_json, arch):
    with h5py.File(hdf5_path, 'r') as f:
        ss     = f['single_scale/utrps'][:]
        ms     = f['multi_scale/utrps'][:]
        labels = f['metadata/labels'][:]
        rids   = np.array([r.decode() for r in f['metadata/record_ids'][:]])
    with open(split_json) as f:
        splits = json.load(f)
    def idx(s):
        return np.where(np.isin(rids, splits[s]))[0]
    tr, va, te = idx('train'), idx('val'), idx('test')
    need_multi = arch in ('multi', 'hybrid')
    def make_ds(ix, aug):
        return UTRPDataset(ss[ix], labels[ix],
                           X_multi=ms[ix] if need_multi else None,
                           augment=aug)
    return make_ds(tr, True), make_ds(va, False), make_ds(te, False), labels, tr, va, te


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        if kernel_size == 2:
            self.pad = nn.ZeroPad2d((0, 1, 0, 1)); pad = 0
        else:
            self.pad = nn.Identity(); pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1,      padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                     nn.BatchNorm2d(out_ch))
                      if stride != 1 or in_ch != out_ch else nn.Identity())
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(self.pad(x))))
        out = self.bn2(self.conv2(self.pad(out)))
        return torch.relu(out + self.skip(x))


def _make_backbone(in_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ResBlock(64, 64, kernel_size=2),
        ResBlock(64, 128, kernel_size=3, stride=2),
        ResBlock(128, 256, kernel_size=3, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
    )


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)


class SingleScaleNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.backbone   = _make_backbone(3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        _init_weights(self)
    def forward(self, x):
        return self.classifier(self.backbone(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class MobileNetV2SingleScale(nn.Module):
    """MobileNet-V2 backbone with custom classifier for single-scale."""
    def __init__(self, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove classifier
        self.backbone.classifier = nn.Identity()

        # Feature dimension from MobileNet-V2

        self.feature_dim = 1280
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        _init_weights(self.classifier)
    
    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class MultiScaleAttentionNet(nn.Module):
    """Late fusion with sigmoid attention — each subband independently weighted."""
    def __init__(self, n_subbands, dropout=0.5):
        super().__init__()
        self.n_subbands = n_subbands
        self.branches   = nn.ModuleList([_make_backbone(3) for _ in range(n_subbands)])
        self.attention  = nn.Sequential(
            nn.Linear(256 * n_subbands, n_subbands * 4), nn.ReLU(inplace=True),
            nn.Linear(n_subbands * 4, n_subbands), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 2),
        )
        _init_weights(self)

    def _feats(self, x):
        return torch.stack([self.branches[i](x[:, i*3:(i+1)*3]) for i in range(self.n_subbands)], dim=1)

    def forward(self, x):
        s = self._feats(x)
        w = self.attention(s.reshape(s.size(0), -1)).unsqueeze(-1)
        return self.classifier((s * w).sum(dim=1))

    def get_attention_weights(self, x):
        with torch.no_grad():
            s = self._feats(x)
            return self.attention(s.reshape(s.size(0), -1)).cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class HybridAttentionNet(nn.Module):
    """Single-scale branch + subband branches with sigmoid attention."""
    def __init__(self, n_subbands, dropout=0.5):
        super().__init__()
        self.n_subbands  = n_subbands
        n_branches       = 1 + n_subbands
        self.ss_branch   = _make_backbone(3)
        self.ms_branches = nn.ModuleList([_make_backbone(3) for _ in range(n_subbands)])
        self.attention   = nn.Sequential(
            nn.Linear(256 * n_branches, n_branches * 4), nn.ReLU(inplace=True),
            nn.Linear(n_branches * 4, n_branches), nn.Sigmoid(),
        )
        self.classifier  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 2),
        )
        _init_weights(self)

    def _feats(self, x_ss, x_ms):
        ss = self.ss_branch(x_ss)
        ms = [self.ms_branches[i](x_ms[:, i*3:(i+1)*3]) for i in range(self.n_subbands)]
        return torch.stack([ss] + ms, dim=1)

    def forward(self, x_ss, x_ms):
        s = self._feats(x_ss, x_ms)
        w = self.attention(s.reshape(s.size(0), -1)).unsqueeze(-1)
        return self.classifier((s * w).sum(dim=1))

    def get_attention_weights(self, x_ss, x_ms):
        with torch.no_grad():
            s = self._feats(x_ss, x_ms)
            return self.attention(s.reshape(s.size(0), -1)).cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# RQA features for XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def extract_rqa_from_rgb(utrp_rgb, epsilon_frac=0.1, lmin=2, vmin=2, size=64):
    import cv2
    # if utrp_rgb.ndim == 3:
    #     gray = utrp_rgb.mean(axis=0)
    # else:
    #     gray = utrp_rgb
    # if gray.shape[0] != size:
    #     gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    # epsilon = epsilon_frac * gray.max()
    # binary  = (gray <= epsilon).astype(np.float32)
    if utrp_rgb.ndim == 3:
        blue = utrp_rgb[2]  # (H, W) — high where distance is low
    else:
        blue = utrp_rgb

    # Recurrent points have blue channel above the (1 - epsilon_frac) quantile
    # i.e. the top epsilon_frac fraction of blue values are recurrent
    threshold = np.percentile(blue[blue > 0], (1 - epsilon_frac) * 100)
    binary = (blue >= threshold).astype(np.float32)
    return extract_rqa(binary, lmin=lmin, vmin=vmin)

def rp_from_stored_utrp(utrp_rgb, epsilon_frac=0.1):
    pass



def extract_rqa_features(utrp):
    """
    ~65 RQA-inspired features from a (H, W) float32 distance matrix.

    Captures:
      - Global distribution of distances (recurrence density)
      - Diagonal profile (determinism proxy — diagonal lines = repeated trajectories)
      - Spatial gradients (laminarity proxy)
      - Quadrant statistics (non-stationarity)
      - Recurrence rate at multiple thresholds
    """
    idx   = np.triu_indices(utrp.shape[0], k=1)
    upper = utrp[idx]

    # Global statistics
    global_stats = np.array([upper.mean(), upper.std(),
                               *np.percentile(upper, [10, 25, 50, 75, 90])])

    # Diagonal profile — mean distance along k-th off-diagonal (k=1..50)
    # Low diagonal values = high determinism = recurrent dynamics
    n = utrp.shape[0]
    diag_means = np.array([np.mean(np.diag(utrp, k)) for k in range(1, min(51, n))])

    # Spatial gradients
    grad_h = np.abs(np.diff(utrp, axis=1)).mean()
    grad_v = np.abs(np.diff(utrp, axis=0)).mean()

    # Quadrant means (non-stationarity across time)
    h, w = utrp.shape
    quads = np.array([utrp[:h//2, :w//2].mean(), utrp[:h//2, w//2:].mean(),
                      utrp[h//2:, :w//2].mean(), utrp[h//2:, w//2:].mean()])

    # Recurrence rate proxies
    p25, p50 = np.percentile(upper, 25), np.percentile(upper, 50)
    rr = np.array([(utrp < t).mean() for t in [p25, p50, p25*2, p50*2]])

    return np.concatenate([global_stats, diag_means, [grad_h, grad_v], quads, rr])


def build_xgb_features(ss, ms, arch, subband_names=None):
    """Build feature matrix from UTRP arrays using RQA features."""
    N = ss.shape[0]

    def to_gray(x):
        return x.mean(axis=0) if x.ndim == 3 else x

    print("Extracting RQA features from UTRPs...")
    ss_feats = np.array([extract_rqa_features(to_gray(ss[i]))
                          for i in tqdm(range(N), desc='SS')])

    if arch == 'single' or ms is None:
        return ss_feats

    n_sub = ms.shape[1] // 3
    ms_feats_list = []
    for s in range(n_sub):
        name = subband_names[s] if subband_names else f'sub{s}'
        gray = ms[:, s*3:(s+1)*3].mean(axis=1)
        feats = np.array([extract_rqa_features(gray[i])
                           for i in tqdm(range(N), desc=name, leave=False)])
        ms_feats_list.append(feats)
    ms_feats = np.concatenate(ms_feats_list, axis=1)

    if arch == 'multi':
        return ms_feats
    return np.concatenate([ss_feats, ms_feats], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _forward(model, batch, arch, device):
    if arch == 'hybrid':
        x_ss, x_ms, labels = batch
        return model(x_ss.to(device), x_ms.to(device)), labels.to(device)
    elif arch == 'multi':
        # Batch returns (x_ss, x_ms, labels) but we only use x_ms for multi-scale model
        _, x_ms, labels = batch
        return model(x_ms.to(device)), labels.to(device)
    else:  # arch == 'single'
        x, labels = batch
        return model(x.to(device)), labels.to(device)


def train_epoch(model, loader, opt, criterion, arch, device):
    model.train()
    loss_sum = correct = n = 0
    for batch in loader:
        logits, labels = _forward(model, batch, arch, device)
        opt.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * len(labels)
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += len(labels)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, arch, device):
    model.eval()
    loss_sum = correct = n = 0
    probs_all, labels_all = [], []
    for batch in loader:
        logits, labels = _forward(model, batch, arch, device)
        loss_sum += criterion(logits, labels).item() * len(labels)
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += len(labels)
        probs_all.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.5
    return loss_sum / n, correct / n, auc, np.array(probs_all), np.array(labels_all)


def threshold_analysis(va_probs, va_true, te_probs, te_true):
    """Find optimal thresholds on val set, report results on test set."""
    # F1-optimal
    threshs = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(va_true, (va_probs >= t).astype(int), zero_division=0) for t in threshs]
    f1_thresh = float(threshs[np.argmax(f1s)])

    # Youden (max sensitivity + specificity)
    fpr, tpr, roc_t = roc_curve(va_true, va_probs)
    youden_thresh = float(roc_t[np.argmax(tpr - fpr)])

    results = {}
    for name, thresh in [('default_0.50', 0.5),
                          (f'f1_optimal_{f1_thresh:.2f}', f1_thresh),
                          (f'youden_{youden_thresh:.2f}', youden_thresh)]:
        preds = (te_probs >= thresh).astype(int)
        cm    = confusion_matrix(te_true, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results[name] = {
            'threshold':   thresh,
            'accuracy':    float(accuracy_score(te_true, preds)),
            'precision':   float(precision_score(te_true, preds, zero_division=0)),
            'recall':      float(recall_score(te_true, preds, zero_division=0)),
            'f1':          float(f1_score(te_true, preds, zero_division=0)),
            'sensitivity': float(tp / (tp + fn)) if tp + fn > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if tn + fp > 0 else 0.0,
            'cm':          cm.tolist(),
        }
    return results


def print_results(results, auc):
    print(f"\n{'='*60}\nTest AUC: {auc:.4f}")
    for name, m in results.items():
        print(f"\n--- {name} ---")
        print(f"  Acc={m['accuracy']:.3f}  Prec={m['precision']:.3f}  "
              f"Rec={m['recall']:.3f}  F1={m['f1']:.3f}")
        print(f"  Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}")
        cm = np.array(m['cm'])
        print(f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")


def save_outputs(out_dir, metrics, history=None, te_probs=None, te_true=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    if history:
        ep = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(ep, history['train_loss'], label='Train')
        axes[0].plot(ep, history['val_loss'],   label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(0.3)
        axes[1].plot(ep, history['train_acc'], label='Train')
        axes[1].plot(ep, history['val_acc'],   label='Val')
        axes[1].set_title('Accuracy'); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(0.3)
        axes[2].plot(ep, history['val_auc'], color='green')
        axes[2].set_title('Val AUC'); axes[2].set_ylim(0, 1); axes[2].grid(0.3)
        plt.tight_layout(); plt.savefig(out_dir / 'training_curves.png', dpi=150); plt.close()

    if te_probs is not None:
        fpr, tpr, _ = roc_curve(te_true, te_probs)
        auc = roc_auc_score(te_true, te_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f'AUC={auc:.4f}')
        plt.plot([0, 1], [0, 1], '--', color='grey')
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title('ROC Curve'); plt.legend(); plt.grid(0.3)
        plt.tight_layout(); plt.savefig(out_dir / 'roc_curve.png', dpi=150); plt.close()

    print(f"Saved → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CNN training
# ══════════════════════════════════════════════════════════════════════════════

def train_cnn(hdf5_path, split_json, arch, results_dir,
              epochs=150, patience=25, lr=0.01,
              dropout=0.5, batch_size=32, seed=42,
              label_smoothing=0.1, backbone='resnet', pretrained=False):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\nDevice: {DEVICE}  Arch: {arch}  Backbone: {backbone}  Pretrained: {pretrained}  LR: {lr}  Seed: {seed}")

    train_ds, val_ds, test_ds, all_labels, tr_idx, _, _ = \
        load_data(hdf5_path, split_json, arch)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    if arch == 'single':
        if backbone == 'mobilenetv2':
            model = MobileNetV2SingleScale(pretrained=pretrained, dropout=dropout).to(DEVICE)
        else:
            model = SingleScaleNet(dropout).to(DEVICE)
    elif arch == 'multi':
        n_sub = train_ds.X_multi.shape[1] // 3
        model = MultiScaleAttentionNet(n_sub, dropout).to(DEVICE)
    else:
        n_sub = train_ds.X_multi.shape[1] // 3
        model = HybridAttentionNet(n_sub, dropout).to(DEVICE)
    print(f"Model: {model.__class__.__name__}  Params: {model.count_params():,}")

    # Class-weighted loss
    n_nsr   = int((all_labels[tr_idx] == 0).sum())
    n_preaf = int((all_labels[tr_idx] == 1).sum())
    w_preaf = min(n_nsr / max(1, n_preaf), 2.0)
    print(f"Class weights: NSR=1.00  Pre-AF={w_preaf:.3f}  ({n_nsr}/{n_preaf})")
    cw        = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # SGD + OneCycleLR — best combination for small datasets
    # OneCycleLR: warm up lr, peak, then anneal. Prevents flat SGD convergence.

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9, weight_decay=1e-3, nesterov=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # optimizer = torch.optim.Adam([
    #     {'params': backbone.features.parameters(), 'lr': 1e-5},
    #     {'params': backbone.classifier.parameters(), 'lr': 1e-4},
    # ], weight_decay=1e-4)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=lr,
    #     steps_per_epoch=len(tr_loader),
    #     epochs=epochs,
    #     pct_start=0.3,        # 30% warm-up
    #     div_factor=10,        # initial lr = max_lr / 25
    #     final_div_factor=100, # final lr = initial / 1e4
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_auc   = 0.0
    best_state = None
    no_improve = 0

    print(f"\n{'='*60}\nTraining\n{'='*60}")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, arch, DEVICE)
        scheduler.step()
        va_loss, va_acc, va_auc, _, _ = eval_epoch(model, va_loader, criterion, arch, DEVICE)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_auc'].append(va_auc)

        print(f"Epoch {epoch:3d}  tr={tr_loss:.4f}/{tr_acc:.3f}  "
              f"va={va_loss:.4f}/{va_acc:.3f}  auc={va_auc:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if va_auc > best_auc:
            best_auc   = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  → Best val AUC: {best_auc:.4f}")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    _, _, _, va_probs, va_true = eval_epoch(model, va_loader, criterion, arch, DEVICE)
    _, _, _, te_probs, te_true = eval_epoch(model, te_loader, criterion, arch, DEVICE)
    te_auc   = roc_auc_score(te_true, te_probs)
    thresh_r = threshold_analysis(va_probs, va_true, te_probs, te_true)
    print_results(thresh_r, te_auc)

    # Attention weights
    attn_info = {}
    if hasattr(model, 'get_attention_weights'):
        all_w = []
        model.eval()
        with torch.no_grad():
            for batch in te_loader:
                if arch == 'hybrid':
                    x_ss, x_ms, _ = batch
                    w = model.get_attention_weights(x_ss.to(DEVICE), x_ms.to(DEVICE))
                else:
                    x, _ = batch
                    w = model.get_attention_weights(x.to(DEVICE))
                all_w.append(w)
        mean_w = np.concatenate(all_w).mean(0)
        print("\nMean attention weights (test set):")
        for i, w in enumerate(mean_w):
            print(f"  branch {i}: {w:.4f}")
        attn_info['mean_attention'] = mean_w.tolist()

    metrics = {'arch': arch, 'seed': seed, 'test_auc': float(te_auc),
               'best_val_auc': float(best_auc), 'thresholds': thresh_r, **attn_info}
    torch.save({'state_dict': best_state, 'metrics': metrics, 'history': history},
               Path(results_dir) / 'checkpoint.pt')
    save_outputs(results_dir, metrics, history, te_probs, te_true)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost training
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb(hdf5_path, split_json, arch, results_dir, seed=42):
    assert HAS_XGB, "pip install xgboost"
    np.random.seed(seed)
    print(f"\nXGBoost  arch={arch}")

    with h5py.File(hdf5_path, 'r') as f:
        ss     = f['single_scale/utrps'][:]
        ms     = f['multi_scale/utrps'][:]
        labels = f['metadata/labels'][:]
        rids   = np.array([r.decode() for r in f['metadata/record_ids'][:]])
        attr   = f.attrs.get('subband_names', '')
        subband_names = attr.split(',') if attr else None

    with open(split_json) as f:
        splits = json.load(f)
    def idx(s):
        return np.where(np.isin(rids, splits[s]))[0]
    tr_idx, va_idx, te_idx = idx('train'), idx('val'), idx('test')
    #
    # X_train = build_xgb_features(ss[tr_idx], ms[tr_idx] if arch != 'single' else None,
    #                               arch, subband_names)
    # X_val   = build_xgb_features(ss[va_idx], ms[va_idx] if arch != 'single' else None,
    #                               arch, subband_names)
    # X_test  = build_xgb_features(ss[te_idx], ms[te_idx] if arch != 'single' else None,
    #                               arch, subband_names)

    X_train = build_rqa_feature_matrix(ss[tr_idx], ms[tr_idx] if arch != 'single' else None,
                                       arch, subband_names, epsilon_frac=0.1)
    X_val = build_rqa_feature_matrix(ss[va_idx], ms[va_idx] if arch != 'single' else None,
                                     arch, subband_names, epsilon_frac=0.1)
    X_test = build_rqa_feature_matrix(ss[te_idx], ms[te_idx] if arch != 'single' else None,
                                      arch, subband_names, epsilon_frac=0.1)
    y_train, y_val, y_test = labels[tr_idx], labels[va_idx], labels[te_idx]

    print(f"Features: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_test)

    n_nsr   = int((y_train == 0).sum())
    n_preaf = int((y_train == 1).sum())
    spw     = min(n_nsr / max(1, n_preaf), 3.0)

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=seed,
        n_jobs=-1, eval_metric='auc', early_stopping_rounds=30,
    )
    model.fit(X_tr_sc, y_train, eval_set=[(X_va_sc, y_val)], verbose=50)

    te_probs = model.predict_proba(X_te_sc)[:, 1]
    va_probs = model.predict_proba(X_va_sc)[:, 1]
    te_auc   = roc_auc_score(y_test, te_probs)
    thresh_r = threshold_analysis(va_probs, y_val, te_probs, y_test)
    print_results(thresh_r, te_auc)

    metrics = {'arch': arch, 'seed': seed, 'test_auc': float(te_auc),
               'n_features': X_train.shape[1], 'thresholds': thresh_r}

    import pickle
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'xgb_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    save_outputs(out_dir, metrics, te_probs=te_probs, te_true=y_test)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5',        required=True)
    p.add_argument('--split',       required=True)
    p.add_argument('--mode',        choices=['cnn', 'xgb'], required=True)
    p.add_argument('--arch',        choices=['single', 'multi', 'hybrid'], default='single')
    p.add_argument('--results_dir', required=True)
    p.add_argument('--epochs',      type=int,   default=150)
    p.add_argument('--patience',    type=int,   default=40)
    p.add_argument('--lr',          type=float, default=0.01)
    p.add_argument('--dropout',     type=float, default=0.5)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--backbone',    choices=['resnet', 'mobilenetv2'], default='resnet',
                   help='Architecture for single-scale CNN')
    p.add_argument('--pretrained',  action='store_true',
                   help='Use pretrained weights (only for mobilenetv2)')
    args = p.parse_args()

    if args.mode == 'cnn':
        train_cnn(args.hdf5, args.split, args.arch, args.results_dir,
                  args.epochs, args.patience, args.lr, args.dropout,
                  args.batch_size, args.seed, args.label_smoothing,
                  backbone=args.backbone, pretrained=args.pretrained)
    else:
        train_xgb(args.hdf5, args.split, args.arch, args.results_dir, args.seed)
# #

# with h5py.File('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/attention/utrp_new_denoise2.h5', 'r') as f:
#     ss = f['single_scale/utrps'][0]  # (3, 256, 256)
#
# dist_proxy = 1.0 - ss[2]   # inverted blue channel
# epsilon    = 0.1 * dist_proxy.max()
# # binary     = (dist_proxy <= epsilon).astype(np.float32)
# blue = ss[2]
# threshold = np.percentile(blue[blue > 0], 90)  # top 10% = recurrent
# binary = (blue >= threshold).astype(np.float32)
# print(f"Threshold: {threshold:.4f}")
# print(f"Binary fraction: {binary.mean():.4f}")  # should be ~0.10
# print(f"Blue channel: min={ss[2].min():.4f} mean={ss[2].mean()} max={ss[2].max():.4f}")
# print(f"dist_proxy:   min={dist_proxy.min():.4f}  max={dist_proxy.max():.4f}")
# print(f"epsilon:      {epsilon:.4f}")
# print(f"Binary fraction of 1s: {binary.mean():.4f}")
# print(f"Expected ~0.10 for ε=10%")