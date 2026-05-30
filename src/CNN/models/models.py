import torch
import torch.nn as nn

def convolutional_backbone(in_ch: int) -> nn.Sequential:
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

        nn.AdaptiveAvgPool2d((1, 1)), #global average pooling
        nn.Flatten(),   # → (B, 64)
    )

class MultiScaleCNN(nn.Module):
    """
    Late feature-fusion with learned branch weights.
    Each subband gets its own backbone, then an attention gate learns to weight each branch's contribution before the classifier.
    """

    def __init__(self, n_subbands: int = 4, dropout: float = 0.5):
        super().__init__()
        self.n_subbands = n_subbands

        # Independent backbone per subband
        self.branches = nn.ModuleList([
            convolutional_backbone(in_ch=3) for _ in range(n_subbands)
        ])

        feat_dim = 64  # output of each backbone

        # Branch attention gate
        # Takes all branch features, outputs a scalar weight per branch
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * n_subbands, n_subbands * 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_subbands * 4, n_subbands),
            nn.Sigmoid(),
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

        # Use learned attention weights
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

class FullBandCNN(nn.Module):
    """
    FullbandCNN baseline using identical backbone and head as each branch in MultiScaleCNN.
    Input:  (B, 3, 256, 256)
    Output: (B, 2)
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = convolutional_backbone(in_ch=3)   # → (B, 64)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

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

