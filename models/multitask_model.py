"""Shared-backbone multi-task network with two task heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskNet(nn.Module):
    """
    Shared MLP backbone + classification head + regression head.
    `shared[-1]` is the last shared layer used for GradNorm / gradient monitoring.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_feat = self.shared(x)
        return self.cls_head(shared_feat), self.reg_head(shared_feat).squeeze(-1)

    def shared_parameters(self):
        return self.shared.parameters()

    def task_losses(
        self, x: torch.Tensor, y_cls: torch.Tensor, y_reg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, pred_reg = self.forward(x)
        loss_cls = F.cross_entropy(logits, y_cls)  # 1.0
        loss_reg = F.mse_loss(pred_reg, y_reg)  # 0.01
        return loss_cls, loss_reg
