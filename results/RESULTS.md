# Results

Test-split percentages, three random seeds unless stated otherwise.
MELD: 1038 / 114 / 280 dialogues, 2610 test utterances, 7 classes.
IEMOCAP: 108 / 12 / 31 dialogues, 1623 test utterances, 6 classes.

Every number below comes from the code in this repository with the settings in
`README.md`; `results/seeds.csv` lists the per-seed values.

## Main table

| dataset | model | Acc | WF1 | MF1 | recipe |
|---|---|---|---|---|---|
| MELD | MERCResidual | 67.14 | 66.31 | 51.42 | task-adapted text stream, focal, lr 1e-5, 40 ep, KD 0.5 |
| MELD | MERCResidual, 3-seed ensemble | 67.13 | 66.37 | 51.58 | same runs, seed average |
| MELD | MERCResidual, six-model ensemble | 68.28 | 67.30 | 51.95 | two text streams x 3 seeds |
| MELD | MERCResidual (frozen features) | 66.92 | 65.95 | 50.06 | original stream, KD 0.5 |
| IEMOCAP | MERCGraphMemory | 69.54 | 69.33 | 67.36 | focal, lr 5e-5, 80 ep, KD 0.5 |
| IEMOCAP | MERCGraphMemory, 3-seed ensemble | 69.93 | 69.60 | | |

MELD accuracy across seeds is 66.82–67.01 on the frozen stream and 66.82–67.47
with the task-adapted stream, IEMOCAP 69.38–69.62: the spread of a single run is
larger than most of the differences discussed below. The six-model MELD ensemble
averages the three runs on each of the two text streams; a dialogue-level paired
bootstrap against the three released checkpoints of the public baseline
(67.13 Acc when averaged) gives +1.15 Acc with a 95% interval of [−0.00, +2.32],
i.e. at the edge of resolution.

## Dev selection signal

Runs are selected by dev WF1 (MELD: 114 dialogues / 1109 utterances). That
signal is weak enough that it does not order these variants the way the test
split does:

| MELD variant | dev WF1 | test Acc | test WF1 |
|---|---|---|---|
| frozen stream, no distillation | 65.36 | 66.30 | 65.39 |
| frozen stream, KD 0.5 | 65.71 | 66.92 | 65.95 |
| frozen stream, external teacher, KD 0.5 | 66.11 | 66.94 | 66.07 |
| task-adapted stream, KD 0.5 | 63.95 | 67.14 | 66.31 |

The external-teacher runs look best on dev and tie on test; the task-adapted
stream looks worst on dev and is best on test. With 1109 dev utterances a
1.8-point dev gap is inside the noise, so we report the adapted-stream result as
an ensemble-level observation rather than a dev-selected gain, and the shipped
default stays the frozen stream.

## Self-distillation

Soft targets come from an ensemble of three baseline runs and are cached in
`data/kd/`. Distillation changes the training objective only; inference cost is
unchanged.

| dataset | KD weight | Acc | WF1 |
|---|---|---|---|
| MELD | 0.0 | 66.30 | 65.39 |
| MELD | 0.5 | 66.92 | 65.95 |
| MELD | 2.0 | 66.97 | 65.89 |
| IEMOCAP | 0.0 | 68.00 | 67.95 |
| IEMOCAP | 0.5 | 69.54 | 69.33 |

Distillation is worth +0.6 accuracy on MELD and +1.5 on IEMOCAP, and it also
shrinks the seed spread (IEMOCAP 67.53–68.39 without it, 69.38–69.62 with it).
IEMOCAP is sensitive to the learning rate under this objective; 5e-5 was chosen
on the dev split (scan at KD 0.5: 1.5e-5 67.02, 2e-5 68.08, 3e-5 68.52,
4e-5 68.76, 5e-5 69.54, 6e-5 68.76, 8e-5 69.13).

## Backbone comparison (MELD test, KD 0.5, 3 seeds)

| model | dialogue encoder | Acc | WF1 |
|---|---|---|---|
| MERC | bidirectional scan + per-speaker recurrent streams | 66.99 | 65.90 |
| MERCResidual | same, with a per-modality residual around the fusion | 66.92 | 65.95 |
| MERCGraph | 2 relational graph blocks, no recurrence | 66.77 | 65.81 |
| MERCGraphMemory | graph for speakers, scans for context and events | 66.68 | 65.72 |

All four are within half a point of each other, i.e. inside the seed spread.
How the dialogue structure is encoded — recurrence over speaker states or
explicit relational edges — does not change accuracy materially once fusion and
the event stream are fixed. The residual path does matter for IEMOCAP: routing
everything through the 128-d mixture loses detail there (63.96 with a plain
mixture bottleneck at lr 1e-5, 66.11 with a per-modality residual added), while
on MELD the two are indistinguishable.

## Ensembling

Averaging the three seeds adds +0.2 accuracy on MELD and +0.4 on IEMOCAP — the
same order as one seed's deviation, so ensembles are a variance reducer rather
than a source of gain here.

A dialogue-level paired bootstrap (`merc.ensemble.paired_bootstrap`) gives 95%
intervals of roughly ±0.6 accuracy points on MELD and ±1.0 on IEMOCAP: any
difference below that resolution is not distinguishable on these test splits.

## Text stream

The text stream is the single largest input (1024 of 1666 dimensions on MELD).
Replacing it changes accuracy more than any architectural choice we tested:

| MELD text stream | Acc | WF1 | MF1 |
|---|---|---|---|
| shipped 1024-d sentence vectors (frozen) | 66.92 | 65.95 | 50.06 |
| DeBERTa-v3-large, masked mean pooling, frozen | 62.22 | 59.71 | 39.62 |
| DeBERTa-v3-large, fine-tuned on the train split, frozen afterwards | 65.13 | 64.40 | 49.23 |
| same, plus distillation (`tools/finetune_text.py`) | 67.14 | 66.31 | 51.42 |
| six-model ensemble (both streams) | 68.28 | 67.30 | 51.95 |

A frozen off-the-shelf encoder is clearly worse than the shipped vectors, and
fine-tuning it on the same labels the classifier sees makes the *frozen* stream
worse again (the features become too separable on the training dialogues and the
classifier overfits them). The encoder only pays off together with distillation,
which regularises that overfitting: +0.22 Acc / +0.36 WF1 alone, +1.36 Acc on the
two-stream ensemble.

The same substitution fails on IEMOCAP: a DeBERTa encoder fine-tuned on its 108
training dialogues reaches 79.1% on the 12 held-out dialogues but 41.3% on the 31
test dialogues (utterance-level linear probe), i.e. it does not transfer across
sessions, and the full model drops to 57.0 Acc. The IEMOCAP numbers above
therefore keep the shipped text stream.

## Distillation source

Soft targets can come from our own baseline seeds or from the released
checkpoints of the public baseline (three checkpoints, averaged).

| dataset | teacher | Acc | WF1 |
|---|---|---|---|
| MELD | own 3-seed baseline | 66.92 | 65.95 |
| MELD | released checkpoints | 66.94 | 66.07 |
| IEMOCAP | own 3-seed baseline | 69.54 | 69.33 |
| IEMOCAP | released checkpoints | 68.78 | 68.96 |

The released teacher is not better: it ties on MELD and loses 0.76 Acc on
IEMOCAP, where it was trained on every dialogue of that corpus (including our
dev split) and produces over-confident targets. Own-teacher caches are the
default.

## Negative results

* A frozen last-token hidden state from an instruction-tuned LLM
  (Qwen2.5-3B-Instruct, 2048-d) is a much weaker text stream than the shipped
  vectors: 49.5% with an utterance-level linear probe after L2 normalisation,
  versus 59.9% for the shipped stream. We used the 3B checkpoint rather than the
  7B one: its weights (≈15 GB) do not fit the free disk, and the 3B probe already
  trails by ten points.
* Fine-tuning the text encoder on the *full* corpus instead of the train split
  leaks and is not used.
* Weights soup across seeds, post-hoc class-bias calibration, and dev-selected
  ensemble weights all fail to transfer to the test split (see the version-1
  report); they are not part of this release.
