import torch
import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UTRP_SIZE = 256

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, downsample=False):
        super().__init__()

        if kernel_size == 2:
            self.pad = nn.ZeroPad2d((0, 1, 0, 1))  # Pad right and bottom
            padding = 0
        else:
            self.pad = nn.Identity()
            padding = (kernel_size - 1) // 2  # Standard 1-pixel pad for 3x3

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(self.pad(x))))
        out = self.bn2(self.conv2(self.pad(out)))
        return torch.relu(out + identity)


class ZhuResNetSingleScale(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.rb1 = ResBlock(64,  64,  kernel_size=2, downsample=False)
        self.rb2 = ResBlock(64,  128, kernel_size=3, downsample=True)
        self.rb3 = ResBlock(128, 256, kernel_size=3, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.rb3(self.rb2(self.rb1(self.stem(x))))))


    def count_params(self):
        return sum(p.numel() for p in self.parameters())