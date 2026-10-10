"""Graph neural network: GraphSAGE over the account graph, classifying each transaction (edge).

Accounts are nodes; transactions are directed edges. Each account starts from unsupervised
statistics of its own activity (how many payments in and out, to how many counterparties, which
rails). Two GraphSAGE layers then mix in what the account's senders and receivers look like, in
both directions, so the embedding of an account reflects the scheme it sits in (a mule collecting
from many fresh accounts, a hub fanning out, a ring). A transaction is scored from the embeddings
of its two accounts plus its own amount, rail and time.

Evaluation follows the AMLworld benchmark protocol (Egressy et al., 2024): split by time; the
graph used to score a period contains all transactions up to the end of that period, labels hidden.
"""
import numpy as np
import polars as pl
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .ingest import FORMATS

def node_table(t):
    """Unsupervised per-account statistics over the transactions in t (no labels). Counts and sums are
    per day of the period, so a graph of 10 days and one of 14 days give comparable values."""
    days = max(1.0, (t['ts'].max() - t['ts'].min()).total_seconds() / 86400)
    out = t.group_by(acct=pl.col('src')).agg(
        n_out=pl.len(), cp_out=pl.col('dst').n_unique(), usd_out=pl.col('amount_usd').sum(),
        self_out=(pl.col('src') == pl.col('dst')).mean(), hour_out=pl.col('ts').dt.hour().mean(),
        *[(pl.col('fmt') == f).mean().alias(f'fo_{i}') for i, f in enumerate(FORMATS)])
    inn = t.group_by(acct=pl.col('dst')).agg(
        n_in=pl.len(), cp_in=pl.col('src').n_unique(), usd_in=pl.col('amount_usd').sum(),
        *[(pl.col('fmt') == f).mean().alias(f'fi_{i}') for i, f in enumerate(FORMATS)])
    nodes = out.join(inn, on='acct', how='full', coalesce=True).fill_null(0)
    for c in ['n_out', 'cp_out', 'usd_out', 'n_in', 'cp_in', 'usd_in']:
        nodes = nodes.with_columns((pl.col(c).cast(pl.Float64) / days).log1p())
    return nodes.with_columns(net=pl.col('usd_in') - pl.col('usd_out'), fan_ratio=pl.col('cp_in') - pl.col('cp_out'))


def edge_attrs(t):
    """Raw attributes of each transaction: log USD amount, rail one-hot, hour and weekday on a circle."""
    h = t['ts'].dt.hour().to_numpy() + t['ts'].dt.minute().to_numpy() / 60
    wd = t['ts'].dt.weekday().to_numpy()
    fmt = np.stack([(t['fmt'] == f).to_numpy() for f in FORMATS], 1)
    a = np.column_stack([np.log1p(t['amount_usd'].to_numpy()), fmt,
                         np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7),
                         (t['src'] == t['dst']).to_numpy()])
    return a.astype(np.float32)


class Graph:
    """Tensors for one period: node features, unique directed account pairs, and per-transaction endpoints."""

    def __init__(self, t, scaler=None):
        nodes = node_table(t)
        self.ids = nodes['acct']
        index = {a: i for i, a in enumerate(self.ids.to_list())}
        feats = nodes.drop('acct').to_numpy().astype(np.float32)
        self.scaler = scaler or (feats.mean(0), feats.std(0) + 1e-6)
        self.x = torch.from_numpy((feats - self.scaler[0]) / self.scaler[1])
        src = t['src'].replace_strict(index, return_dtype=pl.Int64).to_numpy()
        dst = t['dst'].replace_strict(index, return_dtype=pl.Int64).to_numpy()
        pairs = np.unique(np.stack([src, dst], 1), axis=0)   # repeated payments between two accounts: one edge
        self.edge_index = torch.from_numpy(pairs.T.copy())
        self.src, self.dst = torch.from_numpy(src.copy()), torch.from_numpy(dst.copy())


class SAGEEdge(nn.Module):
    def __init__(self, n_node, n_edge, hidden=48):
        super().__init__()
        self.inp = nn.Linear(n_node, hidden)
        self.fwd = nn.ModuleList([SAGEConv(hidden, hidden, aggr='mean') for _ in range(2)])   # money arriving from senders
        self.rev = nn.ModuleList([SAGEConv(hidden, hidden, aggr='mean') for _ in range(2)])   # money leaving to receivers
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(2 * hidden + n_edge, 2 * hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, g):
        h = torch.relu(self.inp(g.x)); ei = g.edge_index
        for f, r, n in zip(self.fwd, self.rev, self.norm):
            h = n(h + torch.relu(f(h, ei) + r(h, ei.flip(0))))   # residual keeps the account's own profile
        return h

    def score(self, h, src, dst, e):
        return self.head(torch.cat([h[src], h[dst], e], 1)).squeeze(1)


def train(g_tr, e_tr, y_tr, g_va, e_va, y_va, va_idx, steps=400, batch_neg=40_000, lr=3e-3, eval_every=20, log=print, seed=0):
    """Full-graph embeddings, mini-batches of transactions (every laundering edge + sampled normal ones)."""
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SAGEEdge(g_tr.x.shape[1], e_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos, neg = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    e_tr_t, e_va_t = torch.from_numpy(e_tr), torch.from_numpy(e_va)
    best, best_state, bad = -1.0, None, 0
    for step in range(1, steps + 1):
        model.train()
        idx = np.concatenate([pos, rng.choice(neg, min(batch_neg, len(neg)), replace=False)])
        yb = torch.from_numpy(y_tr[idx].astype(np.float32))
        h = model.embed(g_tr)
        logit = model.score(h, g_tr.src[idx], g_tr.dst[idx], e_tr_t[idx])
        pw = torch.tensor((len(idx) - len(pos)) / len(pos))   # positives and negatives weigh the same in total
        loss = nn.functional.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps:
            p = predict(model, g_va, e_va_t, va_idx)
            ap = average_precision_score(y_va, p)
            log(f'step {step}: loss {loss.item():.4f}, validation PR-AUC {ap:.4f}')
            if ap > best: best, best_state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 5: break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, g, e, idx=None, chunk=500_000):
    """Raw scores (logits) for the transactions idx of graph g (all of them by default)."""
    model.eval()
    h = model.embed(g)
    e = e if isinstance(e, torch.Tensor) else torch.from_numpy(e)
    idx = torch.arange(len(g.src)) if idx is None else torch.as_tensor(idx)
    out = [model.score(h, g.src[j], g.dst[j], e[j]) for j in idx.split(chunk)]
    return torch.cat(out).numpy()
