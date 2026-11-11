#!/usr/bin/env bash
# IEMOCAP: three seeds of the graph-memory model with self-distillation, then the ensemble.
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  python -m merc.train --dataset iemocap --data data/iemocap --model merc_graph_memory \
    --epochs 80 --lr 5e-5 --seed "$seed" --kd 0.5 --kd-cache data/kd/iemocap_soft.npz \
    --out "runs/iemocap_graph_memory_s${seed}"
done

python - <<'PY'
from merc.ensemble import ensemble, report
probs, labels = ensemble([f'runs/iemocap_graph_memory_s{s}/test_predictions.npz' for s in range(3)])
print(report(probs, labels, 6))
PY
