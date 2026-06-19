"""Common training helpers."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from data.synthetic_multitask import create_dataloaders
from models.multitask_model import MultiTaskNet
from utils.grad_utils import compute_task_grads, cosine_similarity
from utils.monitoring import TrainingMonitor, evaluate


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_run(name: str, args):
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = MultiTaskNet().to(device)
    monitor = TrainingMonitor(
        log_dir=f"runs/{name}",
        plot_path=f"outputs/{name}_summary.png",
        title=f"Multi-Task Demo: {name}",
    )
    return model, train_loader, val_loader, monitor, device


def log_batch_metrics(model, shared_params, loss_cls, loss_reg, monitor, step,
                      loss_total, weight_cls=1.0, weight_reg=1.0, extra=None):
    g_cls, g_reg = compute_task_grads(model, shared_params, loss_cls, loss_reg)
    monitor.log_step(
        step=step, loss_cls=loss_cls.item(), loss_reg=loss_reg.item(), loss_total=loss_total,
        grad_norm_cls=g_cls.norm().item(), grad_norm_reg=g_reg.norm().item(),
        grad_cosine=cosine_similarity(g_cls, g_reg),
        weight_cls=weight_cls, weight_reg=weight_reg, extra=extra,
    )


def run_validation(model, val_loader, monitor, step, device):
    acc, mse = evaluate(model, val_loader, device)
    monitor.log_validation(step, acc, mse)
    return acc, mse
