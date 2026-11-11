#!/usr/bin/env bash
# MELD: three seeds of the memory-stream model with self-distillation, then the ensemble.
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  python -m merc.train --dataset meld --data data/meld --model merc_residual \
    --epochs 40 --lr 1e-5 --seed "$seed" --kd 0.5 --kd-cache data/kd/meld_soft.npz \
    --out "runs/meld_residual_s${seed}"
done

python - <<'PY'
from merc.ensemble import ensemble, report
probs, labels = ensemble([f'runs/meld_residual_s{s}/test_predictions.npz' for s in range(3)])
print(report(probs, labels, 7))
PY
