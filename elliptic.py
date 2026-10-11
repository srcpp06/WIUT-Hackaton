"""External check of the modelling pipeline on Elliptic: real Bitcoin transactions labelled
illicit / licit (Weber et al., 2019). Different domain (crypto, transactions as nodes), so this
validates the approach (LightGBM + GraphSAGE + ensemble), not the bank model itself.

Temporal split as in the paper: time steps 1-34 for learning (1-29 train, 30-34 validation for
early stopping and the threshold), 35-49 test.

    python elliptic.py          # reads data/raw/elliptic/raw/*.csv, writes models/elliptic.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import SAGEConv

from src.evaluate import best_f1_threshold, f1_at

RAW = Path('data/raw/elliptic/raw')


def load():
    f = pd.read_csv(RAW / 'elliptic_txs_features.csv', header=None)
    f.columns = ['txId', 'step'] + [f'f{i}' for i in range(f.shape[1] - 2)]
    cls = pd.read_csv(RAW / 'elliptic_txs_classes.csv')
    e = pd.read_csv(RAW / 'elliptic_txs_edgelist.csv')
    f = f.merge(cls, on='txId')
    f['y'] = f['class'].astype(str).map({'1': 1.0, '2': 0.0})   # 'unknown' -> NaN
    idx = pd.Series(np.arange(len(f)), index=f.txId)
    ei = np.stack([idx[e.txId1].to_numpy(), idx[e.txId2].to_numpy()])
    return f, ei


class SAGENode(nn.Module):
    """The same GraphSAGE blocks as the bank model: money-in and money-out directions, residual, LayerNorm."""

    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.fwd = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(2)])
        self.rev = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x, ei):
        h = torch.relu(self.inp(x))
        for f, r, n in zip(self.fwd, self.rev, self.norm):
            h = n(h + torch.relu(f(h, ei) + r(h, ei.flip(0))))
        return self.head(h).squeeze(1)


def main():
    f, ei = load()
    feats = [c for c in f.columns if c.startswith('f')]
    lab = f.y.notna().to_numpy()
    tr = lab & (f.step <= 29).to_numpy(); va = lab & f.step.between(30, 34).to_numpy(); te = lab & (f.step >= 35).to_numpy()
    y = f.y.fillna(0).to_numpy().astype(int)
    print(f'{len(f):,} transactions, {ei.shape[1]:,} edges; labelled train/val/test {tr.sum():,}/{va.sum():,}/{te.sum():,}; '
          f'illicit share in test {y[te].mean():.3f}', flush=True)
    X = f[feats].to_numpy(np.float32)

    # LightGBM on the transaction's own (and its neighbours' aggregated) features
    m = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=20, feature_fraction=0.7,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, metric='average_precision'),
                  lgb.Dataset(X[tr], y[tr]), 2000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    raw_l = m.predict(X, num_iteration=m.best_iteration)

    # GraphSAGE node classifier on the whole graph (labels of validation/test nodes never used)
    torch.manual_seed(0)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    xt, eit = torch.from_numpy((X - mu) / sd), torch.from_numpy(ei)
    yt = torch.from_numpy(y.astype(np.float32)); trt = torch.from_numpy(tr)
    net = SAGENode(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    pw = torch.tensor((1 - y[tr].mean()) / y[tr].mean())
    best, state, bad = -1, None, 0
    for epoch in range(1, 401):
        net.train(); opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(xt, eit)[trt], yt[trt], pos_weight=pw)
        loss.backward(); opt.step()
        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad(): p = net(xt, eit).numpy()
            ap = average_precision_score(y[va], p[va])
            if ap > best: best, state, bad = ap, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): raw_g = net(xt, eit).numpy()
    print(f'GraphSAGE stopped at epoch {epoch}, validation PR-AUC {best:.4f}', flush=True)

    out = []
    probs = {}
    for name, raw in [('LightGBM', raw_l), ('GraphSAGE', raw_g)]:
        probs[name] = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw[va], y[va]).predict(raw)
    Z = np.column_stack([np.log(raw_l / (1 - raw_l + 1e-9) + 1e-9), raw_g])   # stacking on raw log-odds, fitted on validation
    stack = LogisticRegression(C=1.0).fit(Z[va], y[va])
    probs['Ensemble (stacking)'] = stack.predict_proba(Z)[:, 1]
    print(f'stacking weights: LightGBM {stack.coef_[0, 0]:.3f}, GraphSAGE {stack.coef_[0, 1]:.3f}', flush=True)
    for name, p in probs.items():
        thr = best_f1_threshold(y[va], p[va]); r = f1_at(y[te], p[te], thr)
        out.append({'name': name, 'pr_auc': float(average_precision_score(y[te], p[te])), 'f1': r['f1'],
                    'precision': r['precision'], 'recall': r['recall']})
        print(f'{name:10s} test PR-AUC {out[-1]["pr_auc"]:.4f}  illicit F1 {r["f1"]:.4f}  P {r["precision"]:.3f}  R {r["recall"]:.3f}', flush=True)
    note = elliptic_note(len(f), ei.shape[1], int(te.sum()))
    Path('models').mkdir(exist_ok=True)
    Path('models/elliptic.json').write_text(json.dumps({'note': note, 'models': out}, indent=2, ensure_ascii=False))


def elliptic_note(n_tx, n_edges, n_test):
    return (f"Elliptic (Weber et al., 2019): {n_tx:,} Bitcoin transactions, {n_edges:,} edges. Time steps 1–29 train, "
            f"30–34 validation, 35–49 test ({n_test:,} labelled transactions). A different domain (crypto, a transaction is "
            "a node), so this checks the approach itself, not the bank model.")


if __name__ == '__main__':
    main()
