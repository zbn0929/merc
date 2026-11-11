"""Train and evaluate one model on MELD or IEMOCAP.

Example
-------
python -m merc.train --dataset meld --data data/meld --model merc_residual \
    --epochs 40 --lr 1e-5 --seed 0 --kd 0.5 --kd-cache data/kd/meld_soft.npz \
    --out runs/meld_residual_kd_s0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from . import data as D
from .losses import FocalLoss, distillation_loss
from .models import build


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, loader, n_classes):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        logp = model(batch['e'].cuda(), batch['s'].cuda(), batch['q'].cuda(),
                     batch['mask'].cuda(), batch['lengths'])
        probs.append(logp.exp().cpu().numpy())
        labels.append(batch['y'].numpy())
    p, y = np.concatenate(probs), np.concatenate(labels)
    pred = p.argmax(1)
    return dict(accuracy=100 * accuracy_score(y, pred),
                weighted_f1=100 * f1_score(y, pred, average='weighted', zero_division=0),
                macro_f1=100 * f1_score(y, pred, average='macro', zero_division=0),
                samples=int(len(y))), p, y


def load_soft_targets(path):
    if not path or not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    ids = [str(i) for i in z['ids']]
    return dict(zip(ids, z['probs']))


def targets(soft, batch_ids, lengths):
    rows = []
    for key, n in zip(batch_ids, lengths):
        for j in range(int(n)):
            rows.append(soft[f'{key}_u{j}'])
    return torch.tensor(np.stack(rows), dtype=torch.float32, device='cuda')


def run(args):
    set_seed(args.seed)
    cfg = D.DATASETS[args.dataset]
    loaders = D.loaders(args.dataset, args.data, args.batch_size)
    model = build(args.model, **D.model_kwargs(args.dataset, args.hidden)).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.cosine:
        steps = len(loaders['train']) * args.epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=args.lr * 0.1)
    loss_fn = FocalLoss(gamma=args.gamma)
    soft = load_soft_targets(args.kd_cache) if args.kd > 0 else None
    if args.kd > 0 and soft is None:
        raise SystemExit(f'--kd {args.kd} needs a teacher cache at {args.kd_cache}')

    os.makedirs(args.out, exist_ok=True)
    history, best = [], (-1.0, None, 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for batch in loaders['train']:
            logp = model(batch['e'].cuda(), batch['s'].cuda(), batch['q'].cuda(),
                         batch['mask'].cuda(), batch['lengths'])
            loss = loss_fn(logp, batch['y'].cuda())
            if soft is not None:
                tgt = targets(soft, batch['ids'], batch['lengths'])
                loss = loss + distillation_loss(logp, tgt, args.kd, args.kd_temperature)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()
            if sched is not None:
                sched.step()
            total += float(loss)
        dev, _, _ = evaluate(model, loaders['dev'], cfg['n_classes'])
        history.append(dict(epoch=epoch, loss=total / len(loaders['train']),
                            dev_accuracy=dev['accuracy'], dev_weighted_f1=dev['weighted_f1']))
        print(json.dumps(history[-1]), flush=True)
        if dev['weighted_f1'] > best[0]:
            best = (dev['weighted_f1'], {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)

    model.load_state_dict(best[1])
    test, probs, labels = evaluate(model, loaders['test'], cfg['n_classes'])
    summary = dict(dataset=args.dataset, model=args.model, seed=args.seed, hidden=args.hidden,
                   lr=args.lr, epochs=args.epochs, kd=args.kd, best_epoch=best[2],
                   dev_weighted_f1=best[0], **{f'test_{k}': v for k, v in test.items()},
                   seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
    json.dump(history, open(os.path.join(args.out, 'history.json'), 'w'))
    np.savez(os.path.join(args.out, 'test_predictions.npz'), probs=probs, labels=labels)
    torch.save(dict(model=best[1], config=dict(model=args.model, dataset=args.dataset,
                                               hidden=args.hidden, **D.model_kwargs(args.dataset, args.hidden)),
                    seed=args.seed, epoch=best[2]), os.path.join(args.out, 'best.pt'))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='meld', choices=list(D.DATASETS))
    ap.add_argument('--data', default=None, help='dataset directory')
    ap.add_argument('--model', default='merc_residual', choices=['merc', 'merc_residual',
                                                                'merc_graph', 'merc_graph_memory'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--weight-decay', type=float, default=2e-4)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--clip', type=float, default=5.0)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--kd', type=float, default=0.0, help='weight of the soft-target term')
    ap.add_argument('--kd-temperature', type=float, default=2.0)
    ap.add_argument('--kd-cache', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.data is None:
        args.data = os.path.join('data', args.dataset)
    run(args)


if __name__ == '__main__':
    main()
