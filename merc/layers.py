"""Building blocks: gated memory cells, reliability routing, relational blocks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(log_prob, lengths):
    """(L, B, C) log-probs -> (sum(lengths), C), batch-major inside each dialogue."""
    return torch.cat([log_prob[:, j, :][:lengths[j]] for j in range(len(lengths))])


class SLSTMCell(nn.Module):
    """Stabilised LSTM cell: exponential input/forget gates in log space.

    Follows the sLSTM recurrence (Beck et al., 2024): the normaliser state is
    kept as a running sum and the stabiliser ``m`` prevents overflow.
    """

    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        self.wx = nn.Linear(d_in, 4 * d_h)
        self.ry = nn.Linear(d_h, 4 * d_h, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * d_h))

    def init_state(self, batch, device):
        z = torch.zeros(batch, self.d_h, device=device)
        return [z.clone(), z.clone(), z.clone(), z.clone()]  # y, c, n, m

    def forward(self, x, state):
        y, c, n, m = state
        raw = self.wx(x) + self.ry(y) + self.b
        i_raw, f_raw, z_raw, o_raw = raw.chunk(4, -1)
        log_f = m + F.logsigmoid(f_raw)
        m_new = torch.where(n == 0, i_raw, torch.maximum(i_raw, log_f))
        o = torch.sigmoid(o_raw)
        i = torch.minimum(torch.exp(i_raw - m_new), torch.ones_like(i_raw))
        f = torch.minimum(torch.exp(log_f - m_new), torch.ones_like(f_raw))
        c_new = f * c + i * torch.tanh(z_raw)
        n_new = f * n + i
        y_new = o * c_new / n_new.clamp_min(1e-6)
        return y_new, [y_new, c_new, n_new, m_new]


class Scan(nn.Module):
    """One directional scan of an SLSTM cell over a padded dialogue."""

    def __init__(self, d, h):
        super().__init__()
        self.cell = SLSTMCell(d, h)

    def forward(self, x, umask):
        b, length, _ = x.shape
        state = self.cell.init_state(b, x.device)
        outs = []
        for t in range(length):
            y, state = self.cell(x[:, t], state)
            outs.append(y)
        return torch.stack(outs, 1) * umask.unsqueeze(-1)


class SpeakerStream(nn.Module):
    """One recurrence per speaker; a turn updates only its own speaker's state."""

    def __init__(self, d, h, n_speakers):
        super().__init__()
        self.h, self.n_spk = h, n_speakers
        self.inp = nn.Linear(d, h)
        self.cell = SLSTMCell(h, h)
        self.out = nn.Linear(h, h)

    def forward(self, x, qmask, umask):
        b, length, _ = x.shape
        spk = qmask.argmax(-1).long()
        state = [z.view(b, self.n_spk, self.h) for z in self.cell.init_state(b * self.n_spk, x.device)]
        outs = []
        for t in range(length):
            flat = self.inp(x[:, t]).unsqueeze(1).expand(b, self.n_spk, self.h)
            y_new, new_state = self.cell(flat.reshape(b * self.n_spk, self.h),
                                         [z.reshape(-1, self.h) for z in state])
            y_new = y_new.view(b, self.n_spk, self.h)
            new_state = [z.view(b, self.n_spk, self.h) for z in new_state]
            gate = (umask[:, t] > 0).float().view(b, 1, 1)
            sel = F.one_hot(spk[:, t], self.n_spk).float().unsqueeze(-1) * gate
            state = [s * (1 - sel) + ns * sel for s, ns in zip(state, new_state)]
            cur = y_new.gather(1, spk[:, t].view(b, 1, 1).expand(b, 1, self.h)).squeeze(1)
            outs.append(cur)
        return self.out(torch.stack(outs, 1)) * umask.unsqueeze(-1)


class ReliabilityMoE(nn.Module):
    """Unimodal + cross-modal experts, reliability scored and top-2 routed."""

    def __init__(self, d_t, d_a, d_v, h):
        super().__init__()
        self.pt, self.pa, self.pv = nn.Linear(d_t, h), nn.Linear(d_a, h), nn.Linear(d_v, h)

        def mlp(d_in):
            return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))

        self.e_t, self.e_a, self.e_v = mlp(h), mlp(h), mlp(h)
        self.e_ta, self.e_tv = mlp(2 * h), mlp(2 * h)
        self.rel = nn.ModuleList([nn.Linear(h, 1) for _ in range(3)])
        self.router = nn.Sequential(nn.Linear(3 * h + 3, h), nn.GELU(), nn.Linear(h, 5))
        self.norm = nn.LayerNorm(h)

    def forward(self, t, a, v):
        t, a, v = self.pt(t), self.pa(a), self.pv(v)
        rel = torch.cat([torch.sigmoid(f(x)) for f, x in zip(self.rel, (t, a, v))], -1)
        experts = torch.stack([self.e_t(t), self.e_a(a), self.e_v(v),
                               self.e_ta(torch.cat([t, a], -1)),
                               self.e_tv(torch.cat([t, v], -1))], 2)
        w = torch.softmax(self.router(torch.cat([t, a, v, rel], -1)), -1)
        top2 = w.topk(2, dim=-1)
        keep = torch.zeros_like(w).scatter(-1, top2.indices, 1.0)
        w = w * keep
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        return self.norm(t + (experts * w.unsqueeze(-1)).sum(2))


class RGCBlock(nn.Module):
    """Relational graph convolution: relation-specific weights, degree normalised."""

    def __init__(self, h, n_rel=4, drop=0.2):
        super().__init__()
        self.w = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(n_rel)])
        self.self_w = nn.Linear(h, h, bias=False)
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x, adj):
        deg = adj.sum(-1).clamp_min(1.0)
        msgs = torch.stack([torch.bmm(a, self.w[r](x)) for r, a in enumerate(adj.permute(1, 0, 2, 3))], 0)
        msgs = msgs / deg.permute(1, 0, 2).unsqueeze(-1)
        return self.norm(x + self.drop(self.act(msgs.sum(0) + self.self_w(x))))
