import torch
import torch.nn as nn
from src.CNN.models.multi_scale_cnn import convolutional_backbone

class FullBandCNN(nn.Module):
    """
    baseline CNN model for single full-band URP classification using the same backbone as MultiScaleCNN
    """

    def __init__(self, dropout: float = 0.5):
        """
        initializes the full-band CNN with encoder and classifier

        Args:
            dropout (float): dropout probability for regularization in classifier, defaults to 0.5
        """
        super().__init__()

        # Use same backbone as MultiScaleCNN branches
        self.encoder = convolutional_backbone(in_ch=3)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs forward pass through encoder and classifier

        Args:
            x (torch.Tensor): input tensor of shape (B, 3, 256, 256) for full-band URP images

        Returns:
            torch.Tensor: classification logits of shape (B, 2) for binary classification
        """
        features = self.encoder(x).view(x.size(0), -1)
        return self.classifier(features)

    def _init_weights(self):
        """
        initializes network weights using Kaiming initialization for convolutional layers and Xavier for linear layers
        """
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

    def count_params(self):
        """
        counts the total number of trainable parameters in the model

        Returns:
            int: total number of parameters across all layers
        """
        return sum(p.numel() for p in self.parameters())

