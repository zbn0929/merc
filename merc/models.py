"""The models: a memory-stream variant and a relational-graph variant.

Both consume the same tensors as the reference pipeline so the data loaders and
the evaluator are shared:

    forward(U_e, U_s, qmask, umask, lengths) -> (num_utterances, num_classes) log-prob

U_e is (L, B, d_t + d_a + d_v) utterance features, U_s is (L, B, d_t) sequence
(trigger/context) features, qmask is (L, B, n_speakers), umask is (B, L).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ReliabilityMoE, RGCBlock, Scan, SpeakerStream, flatten


def relation_adjacency(qmask, umask, length):
    """Four dialogue relations: same-speaker causal, cross-speaker causal,
    adjacent turn, self loop.  Returns (B, 4, L, L) float."""
    spk = qmask.argmax(-1)
    causal = torch.tril(torch.ones(length, length, device=spk.device, dtype=torch.bool))
    same = (spk.unsqueeze(1) == spk.unsqueeze(2)) & causal
    cross = (spk.unsqueeze(1) != spk.unsqueeze(2)) & causal
    near = torch.zeros(length, length, device=spk.device, dtype=torch.bool)
    idx = torch.arange(length - 1, device=spk.device)
    near[idx, idx + 1] = True
    near = near | near.t()
    eye = torch.eye(length, device=spk.device, dtype=torch.bool)
    rels = [same, cross, near.unsqueeze(0).expand_as(same), eye.unsqueeze(0).expand_as(same)]
    pad = umask[:, :, None] & umask[:, None, :]
    return torch.stack([r & pad for r in rels], 1).float()


class MERC(nn.Module):
    """Memory-stream model: MoE fusion -> bidirectional scan + speaker streams."""

    def __init__(self, d_t=1024, d_a=300, d_v=342, h=256, n_speakers=9, n_classes=7, speaker=None):
        super().__init__()
        self.h = h
        self.dims = (d_t, d_a, d_v)
        self.fuse = ReliabilityMoE(d_t, d_a, d_v, h)
        self.fwd = Scan(h, h)
        self.bwd = Scan(h, h)
        self.spk = speaker(h, n_speakers) if speaker else SpeakerStream(h, h, n_speakers)
        self.eve = Scan(h, h)
        self.proj_eve = nn.Linear(d_t, h)
        self.head = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, 2 * h), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(2 * h, n_classes))
        self.drop = nn.Dropout(0.1)

    def _inputs(self, U_e, U_s, qmask):
        U_e, U_s, qmask = U_e.transpose(0, 1), U_s.transpose(0, 1), qmask.transpose(0, 1)
        t = U_e[..., :self.dims[0]]
        a = U_e[..., self.dims[0]:self.dims[0] + self.dims[1]]
        v = U_e[..., self.dims[0] + self.dims[1]:]
        return t, a, v, U_s, qmask

    def forward(self, U_e, U_s, qmask, umask, lengths):
        t, a, v, U_s, qmask = self._inputs(U_e, U_s, qmask)
        um = umask.bool()
        x = self.drop(self.fuse(t, a, v))
        f = self.fwd(x, um)
        b = self.bwd(x.flip(1), um.flip(1)).flip(1)
        s = self.spk(x, qmask, um)
        e = self.eve(self.proj_eve(U_s), um)
        hidden = self.head(torch.cat([f, b, s, e, x], -1))
        return flatten(F.log_softmax(hidden, -1).transpose(0, 1), lengths)


class MERCResidual(MERC):
    """Adds a per-modality residual path around the fusion (keeps utterance detail)."""

    def __init__(self, d_t=1024, d_a=300, d_v=342, h=256, n_speakers=9, n_classes=7):
        super().__init__(d_t, d_a, d_v, h, n_speakers, n_classes)
        self.keep = nn.ModuleList([nn.Linear(d_t, h), nn.Linear(d_a, h), nn.Linear(d_v, h)])
        self.up = nn.Linear(4 * h, h)

    def forward(self, U_e, U_s, qmask, umask, lengths):
        t, a, v, U_s, qmask = self._inputs(U_e, U_s, qmask)
        um = umask.bool()
        mix = self.fuse(t, a, v)
        keep = torch.cat([self.keep[0](t), self.keep[1](a), self.keep[2](v)], -1)
        x = self.drop(self.up(torch.cat([mix, keep], -1)))
        f = self.fwd(x, um)
        b = self.bwd(x.flip(1), um.flip(1)).flip(1)
        s = self.spk(x, qmask, um)
        e = self.eve(self.proj_eve(U_s), um)
        hidden = self.head(torch.cat([f, b, s, e, x], -1))
        return flatten(F.log_softmax(hidden, -1).transpose(0, 1), lengths)


class SpeakerGraph(nn.Module):
    """Explicit relational graph over speakers (replaces the recurrent streams)."""

    def __init__(self, h, n_layer=2, drop=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([RGCBlock(h, drop=drop) for _ in range(n_layer)])

    def forward(self, x, qmask, umask):
        adj = relation_adjacency(qmask, umask.bool(), x.size(1))
        for blk in self.blocks:
            x = blk(x, adj)
        return x * umask.unsqueeze(-1)


class MERCGraphMemory(MERC):
    """Memory streams for context/events, explicit speaker graph for structure."""

    def __init__(self, d_t=1024, d_a=300, d_v=342, h=256, n_speakers=9, n_classes=7, gcn_layer=2):
        super().__init__(d_t, d_a, d_v, h, n_speakers, n_classes,
                         speaker=lambda hh, nn_spk: SpeakerGraph(hh, n_layer=gcn_layer))

    def forward(self, U_e, U_s, qmask, umask, lengths):
        t, a, v, U_s, qmask = self._inputs(U_e, U_s, qmask)
        um = umask.bool()
        x = self.drop(self.fuse(t, a, v))
        f = self.fwd(x, um)
        b = self.bwd(x.flip(1), um.flip(1)).flip(1)
        s = self.spk(x, qmask, um)
        e = self.eve(self.proj_eve(U_s), um)
        hidden = self.head(torch.cat([f, b, s, e, x], -1))
        return flatten(F.log_softmax(hidden, -1).transpose(0, 1), lengths)


class MERCGraph(nn.Module):
    """Pure relational model: fusion + speaker/event embedding + 2 graph blocks."""

    def __init__(self, d_t=1024, d_a=300, d_v=342, h=256, n_speakers=9, n_classes=7, n_layer=2):
        super().__init__()
        self.dims = (d_t, d_a, d_v)
        self.fuse = ReliabilityMoE(d_t, d_a, d_v, h)
        self.spk_emb = nn.Embedding(n_speakers, h)
        self.proj_eve = nn.Linear(d_t, h)
        self.blocks = nn.ModuleList([RGCBlock(h) for _ in range(n_layer)])
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h // 2), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(h // 2, n_classes))
        self.drop = nn.Dropout(0.1)

    def forward(self, U_e, U_s, qmask, umask, lengths):
        U_e, U_s, qmask = U_e.transpose(0, 1), U_s.transpose(0, 1), qmask.transpose(0, 1)
        d_t, d_a, _ = self.dims
        x = self.fuse(U_e[..., :d_t], U_e[..., d_t:d_t + d_a], U_e[..., d_t + d_a:])
        x = x + self.spk_emb(qmask.argmax(-1).long()) + self.proj_eve(U_s)
        x = self.drop(x)
        adj = relation_adjacency(qmask, umask.bool(), x.size(1))
        for blk in self.blocks:
            x = blk(x, adj)
        hidden = self.head(x)
        return flatten(F.log_softmax(hidden, -1).transpose(0, 1), lengths)


MODELS = {
    'merc': MERC,
    'merc_residual': MERCResidual,
    'merc_graph_memory': MERCGraphMemory,
    'merc_graph': MERCGraph,
}


def build(name, **kwargs):
    return MODELS[name](**kwargs)
