# MERC — Memory and Relation Models for Emotion Recognition in Conversation

Two backbones for utterance-level emotion classification in multi-party dialogues.
Both remove masked cross-attention entirely and rebuild the dialogue encoder from
recent sequence-model and graph components:

* **memory stream** (`MERC`, `MERCResidual`) — reliability-routed modality mixture,
  bidirectional gated-memory scans, one recurrent state per speaker, an event stream.
* **relational graph** (`MERCGraph`, `MERCGraphMemory`) — the same fusion, with the
  speaker stream replaced by a multi-relational graph convolution over
  same-speaker / cross-speaker / adjacency / self-loop edges.

Training uses focal loss and optional self-distillation from cached soft targets.

## Results

Test split, mean over three seeds (accuracy / weighted-F1).

| dataset | model | Acc | WF1 | notes |
|---|---|---|---|---|
| MELD | MERCResidual | 67.14 | 66.31 | task-adapted text stream, self-distillation, lr 1e-5, 40 epochs |
| MELD | six-model ensemble | 68.28 | 67.30 | two text streams x 3 seeds |
| MELD | MERCResidual (frozen features) | 66.92 | 65.95 | original stream, self-distillation |
| IEMOCAP | MERCGraphMemory | 69.54 | 69.33 | + self-distillation, lr 5e-5, 80 epochs |
| IEMOCAP | MERCGraphMemory (no distillation) | 68.00 | 67.95 | same recipe, ablation |

IEMOCAP uses 108 training / 12 dev / 31 test dialogues (7433 utterances, 6 classes);
MELD uses 1038 / 114 / 280 dialogues (9989 / 1109 / 2610 utterances, 7 classes).
See `results/RESULTS.md` for the full table, ablations and the bootstrap intervals.

## Install

```bash
pip install -r requirements.txt
```

Only PyTorch and scikit-learn are required; the models have no other dependency.
Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` — the per-turn recurrences launch many
small kernels, and oversubscribing threads makes a run roughly ten times slower.

## Data

No corpora are redistributed here. Place the feature archives as follows:

```
data/
  meld/
    meld_features.pkl     # 17-element list of per-dialogue dicts (~486 MB)
    split.json            # {"train": [...], "dev": [...], "test": [...]} dialogue keys
  iemocap/
    iemocap_features.pkl  # 9-tuple archive
    iemocap_text.pkl      # RoBERTa text features
    iemocap_context.pkl   # context features
```

MELD features are the public utterance-level dumps described in
`scripts/features/README.md`; `merc/data.py` documents every index it reads.
IEMOCAP is distributed by USC under a signed agreement — obtain the corpus from
<https://sail.usc.edu/iemocap/> and extract the same features.

**Licences.** MELD is released for research use; cite Poria et al. (2019) if you
use it. IEMOCAP is distributed by USC under a signed agreement and is not
redistributable. Soft-target caches (`data/kd/*.npz`) can be regenerated from
baseline checkpoints with `merc.ensemble`.

## Train

MELD (memory stream, with self-distillation):

```bash
python -m merc.train --dataset meld --data data/meld --model merc_residual \
    --epochs 40 --lr 1e-5 --seed 0 --kd 0.5 --kd-cache data/kd/meld_soft.npz \
    --out runs/meld_residual_s0
```

IEMOCAP (graph memory, higher learning rate):

```bash
python -m merc.train --dataset iemocap --data data/iemocap --model merc_graph_memory \
    --epochs 80 --lr 5e-5 --seed 0 --kd 0.5 --kd-cache data/kd/iemocap_soft.npz \
    --out runs/iemocap_graph_memory_s0
```

`scripts/run_meld.sh` and `scripts/run_iemocap.sh` run the three seeds and the
ensembles. Each run writes `summary.json`, `history.json`, `test_predictions.npz`
and `best.pt` (state dict plus the constructor config).

### Task-adapted text stream (optional, MELD)

`tools/finetune_text.py` fine-tunes DeBERTa-v3 on the utterance labels of the
training split and dumps 1024-d mean-pooled states as an alternative text stream;
no other part of the model changes. On MELD this stream plus distillation gives
67.14 / 66.31, and averaging three seeds of each stream gives 68.28 / 67.30. On
IEMOCAP the same encoder overfits the sessions and is not used — see
`results/RESULTS.md`.

## Ensemble

```python
from merc.ensemble import ensemble, report
probs, labels = ensemble([f'runs/meld_residual_s{s}/test_predictions.npz' for s in range(3)])
print(report(probs, labels, 7))
```

## Layout

```
merc/layers.py    gated memory cell, scan, speaker stream, reliability MoE, graph conv
merc/models.py    MERC, MERCResidual, MERCGraph, MERCGraphMemory
merc/data.py      MELD and IEMOCAP readers
merc/losses.py    focal loss, distillation term
merc/train.py     training / evaluation entry point
merc/ensemble.py  seed averaging and dialogue-level bootstrap
results/          tables and analysis
scripts/          run scripts and feature extraction
tools/            optional text-encoder fine-tuning
```

## Citation

```bibtex
@inproceedings{merc2026,
  title     = {Memory and Relation Models for Emotion Recognition in Conversation},
  author    = {TODO},
  booktitle = {CSCWD},
  year      = {2026}
}
```
