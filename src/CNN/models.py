"""
CNN architectures for UTRP-based pre-AF classification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from base_cnn import BaseCNN

class SingleScaleCNN(BaseCNN):
    """
    CNN for single-scale UTRP images.
    Uses Global Average Pooling (GAP) to drastically reduce parameters.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__(dropout=dropout)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        # FIX: Global Average Pooling (1x1) stops the "parameter explosion"
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout)
        # FIX: Head now only has 256 input features instead of 4096
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    @property
    def input_channels(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ResNet18Transfer(BaseCNN):
    def __init__(self, mode: str = 'multi', dropout: float = 0.5, freeze_backbone: bool = False):
        super().__init__(dropout=dropout)
        self.mode = mode.lower()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 1. FIX: Added logic to handle 1-channel inputs to resolve RuntimeError
        if self.mode == 'multi':
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                self.resnet.conv1.weight = nn.Parameter(avg_weight.repeat(1, 5, 1, 1))
        elif self.mode == 'single':
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                # Average ImageNet's 3 channels into 1 to preserve feature detection
                self.resnet.conv1.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))

        # 2. Strategic Freezing (Unfreeze Block 4 only for bio-adaptation)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        # 3. Lightweight Head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.resnet(x)

    @property
    def input_channels(self) -> int:
        return 5 if self.mode == 'multi' else 1

class MultiScaleCNN(BaseCNN):
    """
    Parallel-branch CNN for multi-scale UTRP (5×128×128).
    ~420k parameters.

    Each subband gets its own branch (shared architecture,
    independent weights). This is motivated by the flickering
    theory: different frequency subbands contribute differently
    to pre-AF dynamics (D4 most discriminative per Experiment 2).
    Independent branch weights allow the network to learn
    subband-specific feature detectors rather than mixing
    subbands in the first layer.

    Branch output: 64-dim vector via GAP.
    Fusion: concatenate → 320-dim → FC.
    """
    def __init__(self, dropout: float = 0.4):
        super().__init__(dropout=dropout)

        # 5 independent branches, one per subband
        # Shared architecture but independent weights
        self.branches = nn.ModuleList([
            self._make_branch() for _ in range(5)
        ])

        # Fusion head: 5 branches × 64 features = 320
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _make_branch() -> nn.Sequential:
        return nn.Sequential(
            # Block 1: 128×128 → 64×64
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64×64 → 32×32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16×16 → 8×8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # GAP: 8×8 → 1×1 → 64-dim vector
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    @property
    def input_channels(self): return 5

    def forward(self, x):
        # x: (batch, 5, 128, 128)
        branch_feats = [
            branch(x[:, i:i+1, :, :]).squeeze(-1).squeeze(-1)
            for i, branch in enumerate(self.branches)
        ]
        # Concatenate along feature dim: (batch, 320)
        fused = torch.cat(branch_feats, dim=1)
        return self.classifier(fused)

# class MultiScaleCNN(BaseCNN):
#     def __init__(self, dropout: float = 0.5, use_parallel_branches: bool = True):
#         super().__init__(dropout=dropout)
#         self.use_parallel_branches = use_parallel_branches
#
#         if use_parallel_branches:
#             # Parallel branch per subband (A5, D1, D2, D3, D4)
#             self.branches = nn.ModuleList([
#                 self._create_branch() for _ in range(5)
#             ])
#             # Each branch outputs 256 features via GAP -> total 1280
#             self.fc1 = nn.Linear(256 * 5, 256)
#         else:
#             # Shared architecture with 5 input channels
#             self.conv_stack = nn.Sequential(
#                 nn.Conv2d(5, 32, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2),
#
#                 nn.Conv2d(32, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2),
#
#                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2),
#
#                 nn.Conv2d(128, 256, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
#             )
#             self.fc1 = nn.Linear(256, 128)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc2 = nn.Linear(256 if use_parallel_branches else 128, 1)
#
#     def _create_branch(self) -> nn.Module:
#         """
#         Creates a branch using smaller kernels (2x2 and 3x3)
#         as suggested by the paper.
#         """
#         return nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=2, padding=1), # Paper uses 2x2 first
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)), # GAP kills overfitting here
#             nn.Flatten()
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.use_parallel_branches:
#             branch_outputs = [branch(x[:, i:i+1, :, :]) for i, branch in enumerate(self.branches)]
#             x = torch.cat(branch_outputs, dim=1)
#         else:
#             x = self.conv_stack(x)
#             x = x.view(x.size(0), -1)
#
#         x = self.dropout(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
