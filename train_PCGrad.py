"""
Direction balancing demo: PCGrad (Project Conflicting Gradients).
"""
from __future__ import annotations
import torch
from utils.train_common import base_parser, setup_run, run_validation
from utils.grad_utils import compute_task_grads, cosine_similarity, pcgrad_combine, assign_flat_grad

def pcgrad_backward(model, loss_cls, loss_reg):
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    loss_cls.backward(retain_graph=True)
    grads_cls = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    model.zero_grad(set_to_none=True)
    loss_reg.backward()
    grads_reg = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    flat_cls = torch.cat([g.view(-1) for g in grads_cls])
    flat_reg = torch.cat([g.view(-1) for g in grads_reg])
    raw_cosine = cosine_similarity(flat_cls, flat_reg)
    merged = pcgrad_combine([flat_cls, flat_reg])
    model.zero_grad(set_to_none=True)
    assign_flat_grad(params, merged)
    return raw_cosine

def train(args):
    name = "direction_pcgrad"
    model, train_loader, val_loader, monitor, device = setup_run(name, args)
    shared_params = list(model.shared.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            x, y_cls, y_reg = batch["x"].to(device), batch["y_cls"].to(device), batch["y_reg"].to(device)
            loss_cls, loss_reg = model.task_losses(x, y_cls, y_reg)
            if global_step % args.log_every == 0:
                g_cls, g_reg = compute_task_grads(model, shared_params, loss_cls, loss_reg)
            raw_cosine = pcgrad_backward(model, loss_cls, loss_reg)
            optimizer.step()
            if global_step % args.log_every == 0:
                monitor.log_step(global_step, loss_cls.item(), loss_reg.item(), (loss_cls+loss_reg).item(),
                    g_cls.norm().item(), g_reg.norm().item(), raw_cosine, 1.0, 1.0,
                    {"pcgrad/conflict": 1.0 if raw_cosine < 0 else 0.0})
        run_validation(model, val_loader, monitor, epoch + 1, device)
    monitor.close()
    print(f"Done. TensorBoard: runs/{name}  Plot: outputs/{name}_summary.png")

if __name__ == "__main__":
    train(base_parser("Direction balancing: PCGrad").parse_args())
