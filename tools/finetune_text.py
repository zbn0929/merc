"""Optional: replace the text stream with a task-adapted encoder.

The shipped features are 1024-d sentence vectors from a frozen encoder. This
tool fine-tunes DeBERTa-v3 on the utterance emotion labels of a *training split*
and dumps its mean-pooled hidden states (also 1024-d) as an alternative text
stream. Nothing else about the model changes.

    python tools/finetune_text.py --dataset meld --root data/meld --out data/meld/text_ft.pkl
    python tools/finetune_text.py --dataset iemocap --root data/iemocap --out data/iemocap/text_ft.pkl

The output has the layout the loaders expect: for MELD a ``{dialogue: (L, 1024)}``
dict, for IEMOCAP a 3-element list whose third slot (index 2) is that dict. Copy
it next to the original archive, point the loader at it, and retrain.

Training entirely in bf16 diverges on this encoder; keep the weights in fp32 and
forward under ``torch.autocast``.
"""
import argparse
import os
import pickle

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DEV = 'cuda'


def meld_pairs(root):
    d = pickle.load(open(os.path.join(root, 'meld_features.pkl'), 'rb'), encoding='latin1')
    import json
    parts = json.loads(open(os.path.join(root, 'split.json')).read())
    keys = parts['train'] + parts['dev'] + parts['test']
    data = {k: ([str(s) for s in d[13][k]], np.asarray(d[1][k])) for k in keys}
    return data, list(parts['train']), list(parts['dev']), 7


def iemocap_pairs(root):
    (vid, spk, lab, txt, aud, vis, sent, tr, te) = pickle.load(
        open(os.path.join(root, 'iemocap_features.pkl'), 'rb'), encoding='latin1')
    keys = list(tr) + list(te)
    data = {k: ([str(s) for s in txt[k]], np.asarray(lab[k])) for k in keys}
    n_valid = max(1, int(0.1 * len(tr)))
    return data, list(tr[n_valid:]), list(tr[:n_valid]), 6


class Enc(nn.Module):
    def __init__(self, name, n_classes):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        self.head = nn.Linear(self.enc.config.hidden_size, n_classes)

    def pooled(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1)

    def forward(self, ids, mask):
        return self.head(self.pooled(ids, mask))


def batches(keys, data, tok, bs, shuffle, max_len=96):
    order = np.random.permutation(len(keys)) if shuffle else np.arange(len(keys))
    for i in range(0, len(order), bs):
        ks = [keys[j] for j in order[i:i + bs]]
        texts = [t for k in ks for t in data[k][0]]
        b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEV)
        y = torch.tensor(np.concatenate([data[k][1] for k in ks]), device=DEV)
        yield ks, b, y


def evaluate(m, keys, data, tok, bs=64):
    m.eval()
    P, Y = [], []
    with torch.no_grad():
        for ks, b, y in batches(keys, data, tok, bs, False):
            P.append(m(b['input_ids'], b['attention_mask']).argmax(1).cpu().numpy())
            Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return f1_score(y, p, average='weighted') * 100, (p == y).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['meld', 'iemocap'], required=True)
    ap.add_argument('--root', required=True, help='directory holding the feature archive')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='microsoft/deberta-v3-large')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch', type=int, default=8)
    a = ap.parse_args()

    data, tr, dv, n_cls = meld_pairs(a.root) if a.dataset == 'meld' else iemocap_pairs(a.root)
    tok = AutoTokenizer.from_pretrained(a.model)
    m = Enc(a.model, n_cls).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.batch + 1)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = nn.CrossEntropyLoss()
    best = (-1, None)
    for ep in range(1, a.epochs + 1):
        m.train()
        for ks, b, y in batches(tr, data, tok, a.batch, True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = m(b['input_ids'], b['attention_mask'])
            loss = lossf(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
        wf1, acc = evaluate(m, dv, data, tok)
        print('epoch %d dev_wf1 %.2f dev_acc %.2f' % (ep, wf1, acc), flush=True)
        if wf1 > best[0]:
            best = (wf1, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])

    m.eval()
    out = {}
    with torch.no_grad():
        for k in data:
            texts = data[k][0]
            vecs = []
            for i in range(0, len(texts), 64):
                b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=96,
                        return_tensors='pt').to(DEV)
                vecs.append(m.pooled(b['input_ids'], b['attention_mask']).cpu().numpy())
            out[k] = np.concatenate(vecs).astype(np.float32)
    if a.dataset == 'iemocap':
        arch = list(pickle.load(open(os.path.join(a.root, 'iemocap_text.pkl'), 'rb'), encoding='latin1'))
        arch[2] = out
        out = arch
    pickle.dump(out, open(a.out, 'wb'), protocol=4)
    print('saved', a.out)


if __name__ == '__main__':
    main()
