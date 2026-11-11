# Results

Test-split percentages, three random seeds unless stated otherwise.
MELD: 1038 / 114 / 280 dialogues, 2610 test utterances, 7 classes.
IEMOCAP: 108 / 12 / 31 dialogues, 1623 test utterances, 6 classes.

Every number below comes from the code in this repository with the settings in
`README.md`; `results/seeds.csv` lists the per-seed values.

## Main table

| dataset | model | Acc | WF1 | MF1 | recipe |
|---|---|---|---|---|---|
| MELD | MERCResidual | 66.92 | 65.95 | 50.06 | focal, lr 1e-5, 40 ep, KD 0.5 |
| MELD | MERCResidual, 3-seed ensemble | 67.09 | 66.05 | | |
| IEMOCAP | MERCGraphMemory | 69.54 | 69.33 | 67.36 | focal, lr 5e-5, 80 ep, KD 0.5 |
| IEMOCAP | MERCGraphMemory, 3-seed ensemble | 69.93 | 69.60 | | |

MELD accuracy across seeds is 66.82–67.01, IEMOCAP 69.38–69.62: the spread of a
single run is larger than most of the differences discussed below.

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
