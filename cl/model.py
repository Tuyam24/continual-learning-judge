"""The learners.

MLP           Split MNIST backbone: 2 hidden layers x 400 units (~480k params).
ReducedResNet18
              Split CIFAR-10 backbone used across the continual-learning literature
              (GEM, A-GEM, MIR, ER-ACE...): ResNet-18 with base width nf=20 instead of 64
              (~1.1M params), so it trains on a laptop.

Both have a single 10-way output head shared across all tasks. How that head is
used (all 10 logits vs. only the current task's) is decided by the scenario in
`cl.metrics.active_logits`, not by the model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 400, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ReducedResNet18(nn.Module):
    def __init__(self, n_classes: int = 10, nf: int = 20):
        super().__init__()
        self.conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        layers, cin = [], nf
        for i, stride in enumerate((1, 2, 2, 2)):
            cout = nf * 2 ** i
            layers += [_BasicBlock(cin, cout, stride), _BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
