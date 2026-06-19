#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS=${EPOCHS:-15}
LOG_EVERY=${LOG_EVERY:-10}
python3 train_baseline.py --epochs "$EPOCHS" --log-every "$LOG_EVERY"
python3 train_magnitude_balance.py --method gradnorm --epochs "$EPOCHS" --log-every "$LOG_EVERY"
python3 train_magnitude_balance.py --method uw --epochs "$EPOCHS" --log-every "$LOG_EVERY"
python3 train_speed_balance.py --epochs "$EPOCHS" --log-every "$LOG_EVERY"
python3 train_direction_balance.py --epochs "$EPOCHS" --log-every "$LOG_EVERY"
python3 train_curriculum_decouple.py --stage1-epochs 5 --stage2-epochs 5 --stage3-epochs 5 --log-every "$LOG_EVERY"
echo "Plots: outputs/*_summary.png | TensorBoard: tensorboard --logdir runs"
