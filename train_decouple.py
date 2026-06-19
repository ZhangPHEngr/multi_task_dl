"""
Curriculum / multi-stage decoupled training demo.

Stage 1: classification only  -> build shared representation
Stage 2: regression only        -> adapt regression head (shared frozen)
Stage 3: joint fine-tune        -> low-lr multi-task alignment
"""
from __future__ import annotations

import torch

from utils.train_common import base_parser, setup_run, run_validation
from utils.grad_utils import monitoring_stats


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def train(args):
    name = "curriculum_decouple"
    model, train_loader, val_loader, monitor, device = setup_run(name, args)
    shared_params = list(model.shared.parameters())
    global_step = 0

    stages = [
        ("stage1_cls_only", args.stage1_epochs, "cls", args.lr),
        ("stage2_reg_only", args.stage2_epochs, "reg", args.lr),
        ("stage3_joint", args.stage3_epochs, "both", args.joint_lr),
    ]

    for stage_name, stage_epochs, mode, lr in stages:
        print(f"\n=== {stage_name}: mode={mode}, lr={lr}, epochs={stage_epochs} ===")
        if mode == "cls":
            set_requires_grad(model.shared, True)
            set_requires_grad(model.cls_head, True)
            set_requires_grad(model.reg_head, False)
        elif mode == "reg":
            set_requires_grad(model.shared, False)
            set_requires_grad(model.cls_head, False)
            set_requires_grad(model.reg_head, True)
        else:
            set_requires_grad(model.shared, True)
            set_requires_grad(model.cls_head, True)
            set_requires_grad(model.reg_head, True)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        for epoch in range(stage_epochs):
            model.train()
            for batch in train_loader:
                global_step += 1
                x = batch["x"].to(device)
                y_cls = batch["y_cls"].to(device)
                y_reg = batch["y_reg"].to(device)
                loss_cls, loss_reg = model.task_losses(x, y_cls, y_reg)

                if mode == "cls":
                    loss = loss_cls
                    w_cls, w_reg = 1.0, 0.0
                elif mode == "reg":
                    loss = loss_reg
                    w_cls, w_reg = 0.0, 1.0
                else:
                    loss = loss_cls + loss_reg
                    w_cls, w_reg = 1.0, 1.0

                if global_step % args.log_every == 0:
                    gn_cls, gn_reg, grad_cos = monitoring_stats(shared_params, loss_cls, loss_reg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % args.log_every == 0:
                    monitor.log_step(
                        global_step,
                        loss_cls.item(),
                        loss_reg.item(),
                        loss.item(),
                        gn_cls,
                        gn_reg,
                        grad_cos,
                        w_cls,
                        w_reg,
                        {"curriculum/stage": {"stage1_cls_only": 1, "stage2_reg_only": 2, "stage3_joint": 3}[stage_name]},
                    )

            run_validation(model, val_loader, monitor, global_step, device)

    monitor.close()
    print(f"Done. TensorBoard: runs/{name}  Plot: outputs/{name}_summary.png")


if __name__ == "__main__":
    parser = base_parser("Curriculum / multi-stage decoupled training")
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--stage3-epochs", type=int, default=9)
    parser.add_argument("--joint-lr", type=float, default=3e-4)
    train(parser.parse_args())
