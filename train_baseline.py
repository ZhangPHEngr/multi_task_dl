"""
Baseline: naive fixed-weight sum (Loss1 + Loss2) for comparison.
"""
from __future__ import annotations
import torch
from utils.train_common import base_parser, setup_run, run_validation, log_batch_metrics

def train(args):
    name = "baseline_fixed_sum"
    model, train_loader, val_loader, monitor, device = setup_run(name, args)
    shared_params = list(model.shared.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    global_step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        for batch in train_loader:
            global_step += 1
            x, y_cls, y_reg = batch["x"].to(device), batch["y_cls"].to(device), batch["y_reg"].to(device)  # (128, 32), (128), (128)
            loss_cls, loss_reg = model.task_losses(x, y_cls, y_reg)
            loss = loss_cls + loss_reg
            if global_step % args.log_every == 0:
                log_batch_metrics(model, shared_params, loss_cls, loss_reg, monitor, global_step, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        run_validation(model, val_loader, monitor, epoch + 1, device)
    monitor.close()
    print(f"Done. TensorBoard: runs/{name}  Plot: outputs/{name}_summary.png")

if __name__ == "__main__":
    train(base_parser("Baseline fixed-weight sum").parse_args())
