
import torch
import torch.nn as nn


# ── Shared backbone block ──────────────────────────────────────────────────────

def make_backbone(in_ch: int) -> nn.Sequential:
    """
    Tier4 backbone: 4 conv blocks → GAP → 64-dim feature vector.
    ~85k parameters for in_ch=3.
    """
    return nn.Sequential(
        # Block 1: 256→128
        nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 2: 128→64
        nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 3: 64→32
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 4: 32→32 (no pool)
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),   # → (B, 64)
    )


# ── 1. Early Fusion ────────────────────────────────────────────────────────────

class EarlyFusionCNN(nn.Module):
    """
    All subband channels concatenated before any convolution.
    The network must learn to separate subband information itself.

    Input:  (B, n_subbands*3, 256, 256)
    Output: (B, 2)
    """
    def __init__(self, n_subbands: int = 3, dropout: float = 0.5):
        super().__init__()
        self.backbone = make_backbone(in_ch=n_subbands * 5)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_subbands*3, 256, 256)
        return self.head(self.backbone(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LateFusionCNN(nn.Module):
    """
    Independent backbone per subband. Features concatenated before head.
    Each subband gets its own set of conv filters — theoretically optimal
    because D4, D5, D6 have different spatial statistics.

    Input:  (B, n_subbands*3, 256, 256) — split internally
    Output: (B, 2)

    The input is split into n_subbands chunks of 3 channels each,
    matching the colourmap RGB representation per subband.
    """
    def __init__(self, n_subbands: int = 3, dropout: float = 0.5):
        super().__init__()
        self.n_subbands = n_subbands

        # Independent backbone per subband (each takes 3 RGB channels)

        self.branches = nn.ModuleList([
            make_backbone(in_ch=3) for _ in range(n_subbands)
        ])

        feat_dim = 64 * n_subbands   # 192 for 3 subbands

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_subbands*3, 256, 256)
        # Split into n_subbands chunks of 3 channels
        feats = [
            self.branches[i](x[:, i*3:(i+1)*3, :, :])
            for i in range(self.n_subbands)
        ]
        return self.head(torch.cat(feats, dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


# ── 3. Hybrid Fusion ───────────────────────────────────────────────────────────

class HybridFusionCNN(nn.Module):
    """
    Independent backbone for single-scale + independent backbone per subband.
    The single-scale branch sees the full-band denoised ECG recurrence structure.
    Each subband branch sees frequency-specific dynamics.
    All features concatenated and fused.

    Inputs: x_ss (B, 3, 256, 256)        — single-scale RGB
            x_ms (B, n_subbands*3, 256, 256) — multi-scale RGB
    Output: (B, 2)

    This is the most flexible architecture — the network can learn to weight
    single-scale vs multi-scale features according to their discriminative value.
    """
    def __init__(self, n_subbands: int = 3, dropout: float = 0.5):
        super().__init__()
        self.n_subbands = n_subbands

        # Single-scale branch

        self.ss_branch = make_backbone(in_ch=3)

        # Independent branch per subband
        self.ms_branches = nn.ModuleList([
            make_backbone(in_ch=3) for _ in range(n_subbands)
        ])

        feat_dim = 64 * (1 + n_subbands)   # 256 for 1 ss + 3 subbands

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
        )
        self._init_weights()

    def forward(self, x_ss: torch.Tensor,
                x_ms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ss: (B, 3, 256, 256)              — single-scale RGB
            x_ms: (B, n_subbands*3, 256, 256)   — multi-scale RGB
        """
        # Single-scale features
        ss_feat = self.ss_branch(x_ss)   # (B, 64)

        # Per-subband features
        ms_feats = [
            self.ms_branches[i](x_ms[:, i*3:(i+1)*3, :, :])
            for i in range(self.n_subbands)
        ]   # list of (B, 64)

        # Concatenate all features
        all_feats = torch.cat([ss_feat] + ms_feats, dim=1)  # (B, 64*(1+n_subbands))
        return self.head(all_feats)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

class HybridFusionWithAttention(nn.Module):
    def __init__(self, n_subbands: int = 4, dropout: float = 0.5):
        super().__init__()
        n_branches = 1 + n_subbands   # single-scale + subbands

        self.ss_branch = make_backbone(in_ch=3)
        self.ms_branches = nn.ModuleList([
            make_backbone(in_ch=3) for _ in range(n_subbands)
        ])

        self.attention = nn.Sequential(
            nn.Linear(64 * n_branches, n_branches * 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_branches * 4, n_branches),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.n_subbands  = n_subbands
        self.n_branches  = n_branches
        self._init_weights()

    def forward(self, x_ss, x_ms):
        ss_feat = self.ss_branch(x_ss)   # (B, 64)
        ms_feats = [
            self.ms_branches[i](x_ms[:, i*3:(i+1)*3])
            for i in range(self.n_subbands)
        ]
        all_feats = torch.stack([ss_feat] + ms_feats, dim=1)  # (B, n_branches, 64)

        flat    = all_feats.reshape(all_feats.size(0), -1)
        weights = self.attention(flat).unsqueeze(-1)           # (B, n_branches, 1)
        fused   = (all_feats * weights).sum(dim=1)             # (B, 64)

        return self.head(fused)


    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Call after training to inspect which subbands were weighted highest."""
        with torch.no_grad():
            branch_feats = [
                self.branches[i](x[:, i * 3:(i + 1) * 3, :, :])
                for i in range(self.n_subbands)
            ]
            stacked = torch.stack(branch_feats, dim=1)
            flat = stacked.reshape(stacked.size(0), -1)
            return self.attention(flat)  # (B, n_subbands)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LateFusionWithAttention(nn.Module):
    """
    Late fusion with learned branch weights.
    Each subband gets its own backbone, then an attention gate
    learns to weight each branch's contribution before the classifier.

    The attention weights are interpretable — after training, inspect
    them to see which subbands the model relied on most.
    """

    def __init__(self, n_subbands: int = 4, dropout: float = 0.5):
        super().__init__()
        self.n_subbands = n_subbands

        # Independent backbone per subband
        self.branches = nn.ModuleList([
            make_backbone(in_ch=3) for _ in range(n_subbands)
        ])

        feat_dim = 64  # output of each backbone

        # Branch attention gate
        # Takes all branch features, outputs a scalar weight per branch
        # Softmax ensures weights sum to 1
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * n_subbands, n_subbands * 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_subbands * 4, n_subbands),
            nn.Sigmoid(),  # weights sum to 1 across subbands
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_subbands*3, 256, 256)

        # Extract features from each branch
        branch_feats = []
        for i in range(self.n_subbands):
            feat = self.branches[i](x[:, i * 3:(i + 1) * 3, :, :])  # (B, 64)
            branch_feats.append(feat)

        # Stack: (B, n_subbands, 64)
        stacked = torch.stack(branch_feats, dim=1)

        # Compute attention weights from all features combined
        # (B, n_subbands*64) → (B, n_subbands)
        flat = stacked.reshape(stacked.size(0), -1)
        weights = self.attention(flat)  # (B, n_subbands)

        # Weighted sum of branch features
        # weights: (B, n_subbands, 1), stacked: (B, n_subbands, 64)
        weights = weights.unsqueeze(-1)  # (B, n_subbands, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, 64)

        return self.head(fused)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Call after training to inspect which subbands were weighted highest."""
        with torch.no_grad():
            branch_feats = [
                self.branches[i](x[:, i * 3:(i + 1) * 3, :, :])
                for i in range(self.n_subbands)
            ]
            stacked = torch.stack(branch_feats, dim=1)
            flat = stacked.reshape(stacked.size(0), -1)
            return self.attention(flat)  # (B, n_subbands)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Single-scale baseline (Tier4, same backbone) ──────────────────────────────

class SingleScaleCNN(nn.Module):
    """
    Single-scale baseline using identical Tier4 backbone.
    Input:  (B, 3, 256, 256)
    Output: (B, 2)
    ~88k parameters.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = make_backbone(in_ch=3)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Tier 4: Medium 4-block CNN (~150k params) ─────────────────────────────────
class Tier4MediumCNN(nn.Module):
    """
    4 conv blocks, 64 filters at peak.
    256→128→64→32→32→GAP→64-dim→2
    ~150k parameters.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 256→128
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 128→64
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64→32
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 32→32 (no pool — adds depth without spatial reduction)
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Tier 5: ResNet-style (~500k params) ───────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch))
                      if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))


class Tier5ResNet(nn.Module):
    """
    Lightweight ResNet: stem + 3 residual blocks with skip connections.
    ~500k parameters.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.rb1 = ResBlock(32,  64, stride=2)
        self.rb2 = ResBlock(64,  128, stride=2)
        self.rb3 = ResBlock(128, 128, stride=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.rb1(x); x = self.rb2(x); x = self.rb3(x)
        return self.head(self.gap(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Tier 6: Full Zhu et al. (~2.7M params) ────────────────────────────────────
class Tier6ZhuResNet(nn.Module):
    """
    Full Zhu et al. 2024 architecture.
    ~2.7M parameters — this is the one that overfits on your dataset.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.rb1 = ResBlock(64,  64)
        self.rb2 = ResBlock(64,  128, stride=2)
        self.rb3 = ResBlock(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.rb3(self.rb2(self.rb1(self.stem(x))))))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
