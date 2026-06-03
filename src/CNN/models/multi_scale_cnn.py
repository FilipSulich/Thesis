import torch
import torch.nn as nn

def convolutional_backbone(in_ch: int) -> nn.Sequential:
    """
    creates a convolutional neural network backbone with four blocks of convolution, batch normalization, and pooling layers

    Args:
        in_ch (int): number of input channels for the first convolutional layer

    Returns:
        nn.Sequential: sequential model with convolutional blocks, global average pooling, and flattening
    """
    return nn.Sequential(

        # Block 1: 256 -> 128
        nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 2: 128 -> 64
        nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 3: 64 -> 32
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 4: 32 ->  32 (no pooling)
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d((1, 1)), #global average pooling
        nn.Flatten(),
    )

class MultiScaleCNN(nn.Module):
    """
    multi-scale CNN with late feature fusion using learned attention weights for subband combination.

    This architecture processes multiple frequency subbands through independent convolutional backbones, then uses an attention mechanism to learn optimal weights for combining subband features before classification.
    """

    def __init__(self, n_subbands: int = 7, dropout: float = 0.5):
        """
        initializes the multi-scale CNN with independent branches and attention mechanism

        Args:
            n_subbands (int): number of frequency subbands to process, defaults to 7
            dropout (float): dropout probability for regularization in classifier, defaults to 0.5
        """
        super().__init__()
        self.n_subbands = n_subbands

        # independent backbone per subband
        self.branches = nn.ModuleList([
            convolutional_backbone(in_ch=3) for _ in range(n_subbands)
        ])

        feat_dim = 64  # output of each backbone

        # takes all branch features, outputs a scalar weight per branch
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * n_subbands, n_subbands * 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_subbands * 4, n_subbands),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs forward pass through the network with attention-weighted feature fusion

        Args:
            x (torch.Tensor): input tensor of shape (B, n_subbands*3, 256, 256) where each subband has 3 channels

        Returns:
            torch.Tensor: classification logits of shape (B, 2) for binary classification
        """
        # x: (B, n_subbands*3, 256, 256)

        # extract features from each branch
        branch_feats = []
        for i in range(self.n_subbands):
            feat = self.branches[i](x[:, i * 3:(i + 1) * 3, :, :])  # (B, 64)
            branch_feats.append(feat)

        # stack: (B, n_subbands, 64)
        stacked = torch.stack(branch_feats, dim=1)

        # use learned attention weights
        flat = stacked.reshape(stacked.size(0), -1)
        weights = self.attention(flat)  # (B, n_subbands)

        # weighted sum of branch features
        weights = weights.unsqueeze(-1)
        fused = (stacked * weights).sum(dim=1)  # (B, 64)

        return self.classifier(fused)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes attention weights for each subband for after training analysis

        Args:
            x (torch.Tensor): input tensor of shape (B, n_subbands*3, 256, 256)

        Returns:
            torch.Tensor: attention weights of shape (B, n_subbands) indicating learned importance of each subband
        """
        with torch.no_grad():
            branch_feats = [
                self.branches[i](x[:, i * 3:(i + 1) * 3, :, :])
                for i in range(self.n_subbands)
            ]
            stacked = torch.stack(branch_feats, dim=1)
            flat = stacked.reshape(stacked.size(0), -1)
            return self.attention(flat)  # (B, n_subbands)

    def _init_weights(self):
        """
        initializes network weights using Kaiming initialization for convolutional layers and Xavier for linear layers
        """
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
        """
        counts the total number of trainable parameters in the model

        Returns:
            int: total number of parameters across all layers
        """
        return sum(p.numel() for p in self.parameters())