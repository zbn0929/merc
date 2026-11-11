"""Seed ensembling and paired bootstrap over dialogues."""
import numpy as np


def ensemble(paths):
    """Average the probability files written by ``merc.train`` (test_predictions.npz)."""
    probs, labels = [], None
    for p in paths:
        z = np.load(p)
        probs.append(z['probs'])
        labels = z['labels']
    return np.mean(probs, 0), labels


def report(probs, labels, n_classes):
    pred = probs.argmax(1)
    from sklearn.metrics import accuracy_score, f1_score
    return dict(accuracy=100 * accuracy_score(labels, pred),
                weighted_f1=100 * f1_score(labels, pred, average='weighted', zero_division=0),
                macro_f1=100 * f1_score(labels, pred, average='macro', zero_division=0))


def paired_bootstrap(y, pred_a, pred_b, groups, n_boot=2000, seed=0):
    """Bootstrap the accuracy gap (a - b), resampling whole dialogues."""
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    index = {k: np.where(groups == k)[0] for k in keys}
    hit_a, hit_b = (pred_a == y), (pred_b == y)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        idx = np.concatenate([index[k] for k in pick])
        deltas.append(100 * (hit_a[idx].mean() - hit_b[idx].mean()))
    deltas = np.array(deltas)
    return dict(mean=float(deltas.mean()),
                ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))])
