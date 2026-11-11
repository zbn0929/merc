"""Dataset readers for MELD and IEMOCAP.

Both readers emit one dict per batch::

    e       (L, B, d_t + d_a + d_v)  utterance features, text / audio / visual
    s       (L, B, d_t)              context (trigger) features
    q       (L, B, n_speakers)       one-hot speaker of each turn
    mask    (B, L)                   utterance validity
    lengths list[int]                valid turns per dialogue
    y       (sum(lengths),)          labels
    ids     list[str]                dialogue id per sample

Feature order is always text -> audio -> visual, which is what the model
slicing in :mod:`merc.models` assumes.
"""
import json
import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

DATASETS = {
    'meld': dict(d_t=1024, d_a=300, d_v=342, n_speakers=9, n_classes=7),
    'iemocap': dict(d_t=1024, d_a=1582, d_v=342, n_speakers=2, n_classes=6),
}


def _pack(rows):
    lengths = [len(r['y']) for r in rows]
    mask = torch.zeros(len(rows), max(lengths))
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    return dict(
        e=pad_sequence([r['e'] for r in rows]),
        s=pad_sequence([r['s'] for r in rows]),
        q=pad_sequence([r['q'] for r in rows]),
        mask=mask,
        lengths=lengths,
        y=torch.cat([r['y'] for r in rows]),
        ids=[r['id'] for r in rows],
    )


class MELD(Dataset):
    """MELD feature archive.

    The archive is a 17-element list of per-dialogue dictionaries:
    index 0 speaker one-hot, 1 labels, 3 text features, 7 context features,
    11 audio frames, 12 visual frames.  ``split.json`` holds three lists of
    dialogue keys (train / dev / test).
    """

    def __init__(self, root, split='train'):
        self.root = root
        self.d = pickle.load(open(os.path.join(root, 'meld_features.pkl'), 'rb'), encoding='latin1')
        parts = json.loads(open(os.path.join(root, 'split.json')).read())
        self.keys = parts[split]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        d = self.d
        e = np.concatenate([d[3][k], d[11][k], d[12][k]], -1)
        return dict(
            e=torch.tensor(e, dtype=torch.float32),
            s=torch.tensor(np.asarray(d[7][k]), dtype=torch.float32),
            q=torch.tensor(np.asarray(d[0][k]), dtype=torch.float32),
            y=torch.tensor(d[1][k], dtype=torch.long),
            id=str(k),
        )


class IEMOCAP(Dataset):
    """IEMOCAP archive: dialogue keys, speakers, labels, audio and visual frames.

    The last tenth of the training dialogues is the held-out dev partition,
    matching the numbers reported in ``results/``.
    """

    def __init__(self, root, split='train'):
        (self.vid, self.spk, self.lab, self.txt, self.aud, self.vis,
         self.sent, tr, te) = pickle.load(open(os.path.join(root, 'iemocap_features.pkl'), 'rb'),
                                          encoding='latin1')
        emo = pickle.load(open(os.path.join(root, 'iemocap_text.pkl'), 'rb'), encoding='latin1')[2]
        sem = pickle.load(open(os.path.join(root, 'iemocap_context.pkl'), 'rb'), encoding='latin1')[6]
        self.emo, self.sem = emo, sem
        n_valid = max(1, int(0.1 * len(tr)))
        self.keys = {'dev': list(tr[:n_valid]), 'train': list(tr[n_valid:]), 'test': list(te)}[split]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        v = self.keys[i]
        e = np.concatenate([np.asarray(self.emo[v], dtype=np.float32),
                            np.asarray(self.aud[v], dtype=np.float32),
                            np.asarray(self.vis[v], dtype=np.float32)], 1)
        q = np.zeros((len(self.lab[v]), 2), dtype=np.float32)
        for j, name in enumerate(self.spk[v]):
            q[j, 0 if name == 'M' else 1] = 1
        return dict(
            e=torch.tensor(e, dtype=torch.float32),
            s=torch.tensor(np.asarray(self.sem[v], dtype=np.float32)),
            q=torch.tensor(q),
            y=torch.tensor(np.asarray(self.lab[v]), dtype=torch.long),
            id=str(v),
        )


def loaders(name, root, batch_size=32):
    cls = {'meld': MELD, 'iemocap': IEMOCAP}[name]
    out = {}
    for split in ('train', 'dev', 'test'):
        ds = cls(root, split)
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                                collate_fn=_pack, num_workers=0)
    return out


def model_kwargs(name, hidden=256):
    cfg = DATASETS[name]
    return dict(d_t=cfg['d_t'], d_a=cfg['d_a'], d_v=cfg['d_v'], h=hidden,
                n_speakers=cfg['n_speakers'], n_classes=cfg['n_classes'])
