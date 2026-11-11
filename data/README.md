# Data

No corpora are redistributed with this repository. Training expects this layout:

```
data/
  meld/
    meld_features.pkl   list of 17-element per-dialogue records
    split.json          {"train": [...], "dev": [...], "test": [...]} dialogue keys
  iemocap/
    iemocap_features.pkl  9-tuple archive
    iemocap_text.pkl      RoBERTa text features (1024-d)
    iemocap_context.pkl   context features (1024-d)
  kd/                     optional cached teacher distributions (*.npz)
```

`merc/data.py` documents which tuple/list indices the loaders read.

## MELD

`meld_features.pkl` is a list of 17-element dialogue records. Only these indices
are used: `0` speaker one-hot, `1` label, `3` text features (1024-d), `7` context
features, `11` audio features (300-d), `12` visual features (342-d), `13`
speaker names, and `14` / `15` / `16` for the train / test / dev key lists.
`split.json` repeats those three lists and can be rebuilt from `14/15/16`.

## IEMOCAP

Distributed by USC under a signed agreement, obtain it from
<https://sail.usc.edu/iemocap/> and extract features with the scripts in
`scripts/features/`. Nothing from the corpus is stored in this repository.

## Soft targets

`merc.train --kd` expects a `.npz` with an `ids` array and either a `probs`
matrix (MELD) or an object array of per-dialogue probability matrices
(IEMOCAP). Regenerate them by running three baseline seeds and averaging their
predictions with `merc.ensemble`.
