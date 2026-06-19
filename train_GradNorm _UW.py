"""
Magnitude balancing demo: GradNorm & Uncertainty Weighting (UW).

Usage:
  python train_magnitude_balance.py --method gradnorm
  python train_magnitude_balance.py --method uw
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.train_common import base_parser, setup_run, run_validation
from utils.grad_utils import compute_task_grads, cosine_similarity


class GradNormController(nn.Module):
    def __init__(self, num_tasks: int = 2, alpha: float = 1.5, lr: float = 0.025):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.alpha = alpha
        self.optimizer = torch.optim.Adam([self.weights], lr=lr)
        self.initial_losses = None

    def get_weights(self):
        w = torch.relu(self.weights)
        w = w * (len(w) / (w.sum() + 1e-8))  # normalize weights to sum to len(w)
        return w[0].item(), w[1].item()

    def update(self, shared_params, loss_cls, loss_reg):
        if self.initial_losses is None:
            self.initial_losses = [loss_cls.item(), loss_reg.item()]
        w = torch.relu(self.weights)
        w = w * (len(w) / (w.sum() + 1e-8))  # normalize weights to sum to len(w)
        weighted = [w[0] * loss_cls, w[1] * loss_reg]
        grad_norms = []
        for wl in weighted:
            grads = torch.autograd.grad(
                wl, shared_params, retain_graph=True, create_graph=True, allow_unused=True
            )
            flat = torch.cat([
                g.view(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
                for g, p in zip(grads, shared_params)
            ])
            grad_norms.append(flat.norm())
        mean_norm = sum(grad_norms) / len(grad_norms)  # 多任务梯度平均值

        loss_ratio = torch.tensor([
            (loss_cls / self.initial_losses[0]).detach(),
            (loss_reg / self.initial_losses[1]).detach(),
        ], device=w.device)  # loss衰减比例，相比初始损失的比例，越小说明任务学的越快，损失下降越快
        target = (loss_ratio / (loss_ratio.mean() + 1e-8)) ** self.alpha

        gradnorm_loss = sum((gn - mean_norm * target[i]) ** 2 for i, gn in enumerate(grad_norms))  # 让目标梯度对于简单任务小一点
        self.optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.optimizer.step()
        return self.get_weights()


class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]  # 遇到更大损失的任务，log_vars会增大，从而减小权重
        return total

    def get_weights(self):
        return torch.exp(-self.log_vars[0]).item(), torch.exp(-self.log_vars[1]).item()


def train(args):
    name = f"magnitude_{args.method}"
    model, train_loader, val_loader, monitor, device = setup_run(name, args)
    shared_params = list(model.shared.parameters())
    if args.method == "gradnorm":
        controller = GradNormController().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        uw = UncertaintyWeighting().to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(uw.parameters()), lr=args.lr)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            x, y_cls, y_reg = batch["x"].to(device), batch["y_cls"].to(device), batch["y_reg"].to(device)
            loss_cls, loss_reg = model.task_losses(x, y_cls, y_reg)
            if args.method == "gradnorm":
                if global_step % args.log_every == 0:
                    g_cls, g_reg = compute_task_grads(model, shared_params, loss_cls, loss_reg)
                w_cls, w_reg = controller.update(shared_params, loss_cls, loss_reg)
                loss = w_cls * loss_cls + w_reg * loss_reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                extra = {}
            else:
                loss = uw([loss_cls, loss_reg])
                w_cls, w_reg = uw.get_weights()
                if global_step % args.log_every == 0:
                    g_cls, g_reg = compute_task_grads(model, shared_params, loss_cls, loss_reg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                extra = {"uw/log_var_cls": uw.log_vars[0].item(), "uw/log_var_reg": uw.log_vars[1].item()}
            if global_step % args.log_every == 0:
                monitor.log_step(global_step, loss_cls.item(), loss_reg.item(), loss.item(),
                    g_cls.norm().item(), g_reg.norm().item(), cosine_similarity(g_cls, g_reg), w_cls, w_reg, extra)
        run_validation(model, val_loader, monitor, epoch + 1, device)
    monitor.close()
    print(f"Done. TensorBoard: runs/{name}  Plot: outputs/{name}_summary.png")

if __name__ == "__main__":
    parser = base_parser("Magnitude balancing: GradNorm / UW")
    parser.add_argument("--method", choices=["gradnorm", "uw"], default="gradnorm")
    train(parser.parse_args())
