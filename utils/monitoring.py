"""Training monitor: TensorBoard logging + matplotlib summary plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    def __init__(self, log_dir: str, plot_path: str, title: str):
        self.log_dir = Path(log_dir)
        self.plot_path = Path(plot_path)
        self.title = title
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.history = {
            "loss_cls": [], "loss_reg": [], "loss_total": [],
            "grad_norm_cls": [], "grad_norm_reg": [], "grad_cosine": [],
            "weight_cls": [], "weight_reg": [],
            "val_acc_cls": [], "val_mse_reg": [],
        }

    def log_step(self, step, loss_cls, loss_reg, loss_total,
                 grad_norm_cls, grad_norm_reg, grad_cosine,
                 weight_cls=1.0, weight_reg=1.0, extra=None):
        metrics = {
            "loss/cls": loss_cls, "loss/reg": loss_reg, "loss/total": loss_total,
            "grad/norm_cls": grad_norm_cls, "grad/norm_reg": grad_norm_reg,
            "grad/cosine_sim": grad_cosine,
            "weight/cls": weight_cls, "weight/reg": weight_reg,
        }
        if extra:
            metrics.update(extra)
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.history["loss_cls"].append(loss_cls)
        self.history["loss_reg"].append(loss_reg)
        self.history["loss_total"].append(loss_total)
        self.history["grad_norm_cls"].append(grad_norm_cls)
        self.history["grad_norm_reg"].append(grad_norm_reg)
        self.history["grad_cosine"].append(grad_cosine)
        self.history["weight_cls"].append(weight_cls)
        self.history["weight_reg"].append(weight_reg)

    def log_validation(self, step, acc_cls, mse_reg):
        self.writer.add_scalar("val/acc_cls", acc_cls, step)
        self.writer.add_scalar("val/mse_reg", mse_reg, step)
        self.history["val_acc_cls"].append(acc_cls)
        self.history["val_mse_reg"].append(mse_reg)

    def close(self):
        self.writer.close()
        self._save_plots()

    def _save_plots(self):
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(self.title, fontsize=14)
        steps = range(1, len(self.history["loss_cls"]) + 1)

        axes[0, 0].plot(steps, self.history["loss_cls"], label="cls", color="#e74c3c")
        axes[0, 0].plot(steps, self.history["loss_reg"], label="reg", color="#3498db")
        axes[0, 0].plot(steps, self.history["loss_total"], label="total", color="#2ecc71", linestyle="--")
        axes[0, 0].set_title("Task Losses"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, self.history["grad_norm_cls"], label="||g_cls||", color="#e74c3c")
        axes[0, 1].plot(steps, self.history["grad_norm_reg"], label="||g_reg||", color="#3498db")
        axes[0, 1].set_title("Shared-Layer Gradient Norms"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(steps, self.history["grad_cosine"], color="#9b59b6")
        axes[0, 2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0, 2].set_title("Gradient Cosine Similarity"); axes[0, 2].set_ylim(-1.05, 1.05); axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(steps, self.history["weight_cls"], label="w_cls", color="#e74c3c")
        axes[1, 0].plot(steps, self.history["weight_reg"], label="w_reg", color="#3498db")
        axes[1, 0].set_title("Dynamic Task Weights"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        if self.history["val_acc_cls"]:
            vs = range(1, len(self.history["val_acc_cls"]) + 1)
            axes[1, 1].plot(vs, self.history["val_acc_cls"], color="#e74c3c", marker="o", markersize=3)
        axes[1, 1].set_title("Val Classification Accuracy"); axes[1, 1].grid(True, alpha=0.3)

        if self.history["val_mse_reg"]:
            vs = range(1, len(self.history["val_mse_reg"]) + 1)
            axes[1, 2].plot(vs, self.history["val_mse_reg"], color="#3498db", marker="o", markersize=3)
        axes[1, 2].set_title("Val Regression MSE"); axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    correct, total, mse_sum = 0, 0, 0.0
    for batch in val_loader:
        x = batch["x"].to(device)
        y_cls = batch["y_cls"].to(device)
        y_reg = batch["y_reg"].to(device)
        logits, pred_reg = model(x)
        correct += (logits.argmax(dim=1) == y_cls).sum().item()
        total += y_cls.size(0)
        mse_sum += ((pred_reg - y_reg) ** 2).sum().item()
    return correct / max(total, 1), mse_sum / max(total, 1)
