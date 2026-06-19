"""Gradient utilities for multi-task balancing methods."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def flatten_grads(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(grads)


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.pow(2).sum().item()
    return total ** 0.5


def compute_task_grads(
    model: nn.Module,
    shared_params: list[nn.Parameter],
    loss_cls: torch.Tensor,
    loss_reg: torch.Tensor,
    retain_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-task gradients w.r.t. shared parameters (does not mutate .grad)."""
    grads_cls = torch.autograd.grad(
        loss_cls, shared_params, retain_graph=True, create_graph=False, allow_unused=True
    )
    grads_reg = torch.autograd.grad(
        loss_reg, shared_params, retain_graph=retain_graph, create_graph=False, allow_unused=True
    )
    flat_cls = torch.cat([
        g.view(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(grads_cls, shared_params)
    ])
    flat_reg = torch.cat([
        g.view(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(grads_reg, shared_params)
    ])
    return flat_cls, flat_reg


def cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    denom = g1.norm() * g2.norm()
    if denom.item() < 1e-12:
        return 0.0
    return (torch.dot(g1, g2) / denom).item()


def pcgrad_combine(task_grads: list[torch.Tensor]) -> torch.Tensor:
    """PCGrad: project conflicting gradients onto normal planes."""
    merged = [g.clone() for g in task_grads]
    for i in range(len(merged)):
        for j in range(len(merged)):
            if i == j:
                continue
            gi, gj = merged[i], merged[j]
            dot = torch.dot(gi, gj)
            if dot < 0:
                merged[i] = gi - (dot / (gj.pow(2).sum() + 1e-8)) * gj
    return sum(merged)


def assign_flat_grad(params: list[nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for p in params:
        numel = p.numel()
        chunk = flat_grad[offset : offset + numel].view_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += numel


def zero_grads(model: nn.Module) -> None:
    model.zero_grad(set_to_none=True)


def single_task_flat_grad(shared_params, loss: torch.Tensor) -> torch.Tensor:
    device = shared_params[0].device
    if not loss.requires_grad:
        return torch.zeros(1, device=device)
    active = [p for p in shared_params if p.requires_grad]
    if not active:
        return torch.zeros(1, device=device)
    grads = torch.autograd.grad(
        loss, active, retain_graph=True, create_graph=False, allow_unused=True
    )
    return torch.cat([g.view(-1) for g in grads if g is not None])


def monitoring_stats(shared_params, loss_cls, loss_reg):
    g_cls = single_task_flat_grad(shared_params, loss_cls)
    g_reg = single_task_flat_grad(shared_params, loss_reg)
    cos = cosine_similarity(g_cls, g_reg) if g_cls.norm() > 0 and g_reg.norm() > 0 else 0.0
    return g_cls.norm().item(), g_reg.norm().item(), cos
