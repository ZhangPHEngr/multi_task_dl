"""Synthetic multi-task dataset with intentional loss-scale imbalance."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SyntheticMultiTaskDataset(Dataset):
    """
    Two tasks on shared latent features:
      - Task 1 (classification): 3-class, naturally larger CE loss scale
      - Task 2 (regression): small-scale targets, naturally smaller MSE scale

    Deliberate imbalance helps visualize magnitude / direction balancing methods.
    """

    def __init__(self, n_samples: int = 4000, input_dim: int = 32, seed: int = 42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(n_samples, input_dim, generator=generator) # (4000, 32)

        latent = self.features @ torch.randn(input_dim, 8, generator=generator)  # (4000, 8)
        class_logits = latent[:, :4].sum(dim=1, keepdim=True) + 0.5 * latent[:, 4:6].sum(dim=1, keepdim=True)
        self.labels_cls = torch.argmax(
            torch.cat([class_logits, -class_logits, class_logits * 0.5], dim=1), dim=1
        )
        self.labels_reg = 0.01 * (latent[:, 6] + 0.3 * latent[:, 7])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "x": self.features[idx],
            "y_cls": self.labels_cls[idx],
            "y_reg": self.labels_reg[idx],
        }


def create_dataloaders(
    batch_size: int = 128,
    n_samples: int = 4000,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticMultiTaskDataset(n_samples=n_samples, seed=seed)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
