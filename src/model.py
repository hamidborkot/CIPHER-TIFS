"""ThreatNet: MLP classifier for insider threat detection."""
import torch
import torch.nn as nn


class ThreatNet(nn.Module):
    """
    Feedforward MLP with BatchNorm and Dropout.
    Input: feature vector of dimension `input_dim`
    Output: scalar probability (sigmoid-activated)
    """

    def __init__(self, input_dim: int, hidden: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
