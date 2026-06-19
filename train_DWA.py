"""
Speed balancing demo: DWA (Dynamic Weight Average).
"""
from __future__ import annotations
import torch
from utils.train_common import base_parser, setup_run, run_validation
from utils.grad_utils import compute_task_grads, cosine_similarity

# def compute_dwa_weights(prev_losses, curr_losses, temperature=2.0):
#     ratios = [prev_losses[i] / (curr_losses[i] + 1e-8) for i in range(len(curr_losses))]
#     exp_ratios = [r ** (1.0 / temperature) for r in ratios]
#     total = sum(exp_ratios)
#     return len(exp_ratios) * exp_ratios[0] / total, len(exp_ratios) * exp_ratios[1] / total

def compute_dwa_weights(prev_losses, curr_losses, temperature=2.0):
    ratios = [curr_losses[i] / (prev_losses[i] + 1e-8) for i in range(len(curr_losses))]
    exp_ratios = [r / temperature for r in ratios]
    total = sum(exp_ratios)
    return len(exp_ratios) * exp_ratios[0] / total, len(exp_ratios) * exp_ratios[1] / total

def train(args):
    name = "speed_dwa"
    model, train_loader, val_loader, monitor, device = setup_run(name, args)
    shared_params = list(model.shared.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    w_cls, w_reg = 1.0, 1.0
    prev_epoch_losses = None
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = [0.0, 0.0]
        n_batches = 0
        for batch in train_loader:
            global_step += 1
            n_batches += 1
            x, y_cls, y_reg = batch["x"].to(device), batch["y_cls"].to(device), batch["y_reg"].to(device)
            loss_cls, loss_reg = model.task_losses(x, y_cls, y_reg)
            epoch_losses[0] += loss_cls.item()
            epoch_losses[1] += loss_reg.item()
            loss = w_cls * loss_cls + w_reg * loss_reg
            if global_step % args.log_every == 0:
                g_cls, g_reg = compute_task_grads(model, shared_params, loss_cls, loss_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if global_step % args.log_every == 0:
                monitor.log_step(global_step, loss_cls.item(), loss_reg.item(), loss.item(),
                    g_cls.norm().item(), g_reg.norm().item(), cosine_similarity(g_cls, g_reg), w_cls, w_reg,
                    {"dwa/epoch": epoch + 1})
        avg = [l / max(n_batches, 1) for l in epoch_losses]
        if prev_epoch_losses is not None and epoch >= args.warmup_epochs:
            w_cls, w_reg = compute_dwa_weights(prev_epoch_losses, avg, args.temperature)
            print(f"Epoch {epoch+1}: DWA weights cls={w_cls:.3f} reg={w_reg:.3f}")
        prev_epoch_losses = avg
        run_validation(model, val_loader, monitor, epoch + 1, device)
    monitor.close()
    print(f"Done. TensorBoard: runs/{name}  Plot: outputs/{name}_summary.png")

if __name__ == "__main__":
    parser = base_parser("Speed balancing: DWA")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    train(parser.parse_args())
