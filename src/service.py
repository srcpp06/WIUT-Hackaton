"""What the web app does, without the web: load the models, score the demo bank's transactions,
explain alerts, take new transactions live, score uploaded CSVs and log the officer's decisions."""
import json
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch

from . import gnn
from .audit import AuditLog
from .explain import Explainer, subgraph
from .features import build, risk_history
from .ingest import UZS_PER_USD, from_uz, read_uz

ROOT = Path(__file__).resolve().parent.parent
HIGH, MEDIUM = 0.005, 0.02          # top 0.5% of transactions: high priority; up to the top 2%: medium
VM_USD_PER_HOUR, VM_CORES = 0.038, 2  # Oracle A1 list price for the demo VM (2 OCPU + 12 GB)


class Models:
    """The trained LightGBM, GNN and the ensemble calibrator (all from train.py)."""

    def __init__(self, d):
        self.lgbm = joblib.load(d / 'baseline.joblib')
        meta = joblib.load(d / 'gnn_meta.joblib')
        self.gnn = gnn.SAGEEdge(meta['n_node'], meta['n_edge'])
        self.gnn.load_state_dict(torch.load(d / 'gnn.pt', weights_only=True)); self.gnn.eval()
        self.scaler, self.cal_g = meta['scaler'], meta['calibrator']
        self.stack = joblib.load(d / 'ensemble.joblib')
        self.metrics = json.loads((d / 'metrics.json').read_text())

    def gnn_raw(self, t, e=None):
        return gnn.predict(self.gnn, gnn.Graph(t, self.scaler), gnn.edge_attrs(t) if e is None else e)

    def scores(self, raw_l, raw_g):
        """Calibrated probabilities of each model and of the stacked ensemble."""
        p = self.stack.predict_proba(np.column_stack([raw_l, raw_g]))[:, 1]
        return self.lgbm.calibrator.predict(raw_l), self.cal_g.predict(raw_g), p


class Service:
    def __init__(self, models_dir=ROOT / 'models', demo_dir=ROOT / 'data' / 'demo', audit=ROOT / 'var' / 'audit.jsonl'):
        t0 = time.time()
        self.m = Models(Path(models_dir))
        self.base = read_uz(Path(demo_dir) / 'transactions.csv').with_columns(pl.col('row').cast(pl.Int64))
        self.accounts = pl.read_csv(Path(demo_dir) / 'accounts.csv', schema_overrides={'account': pl.Utf8}).with_columns(
            node=pl.col('bank') + ':' + pl.col('account'))
        self.audit = AuditLog(audit)
        self.lock = threading.Lock()
        self.latency = deque(maxlen=2000)   # (wall ms per transaction, CPU ms per transaction)
        self.reset()
        self.startup_s = time.time() - t0

    # ------------------------------------------------------------------ state
    def reset(self):
        """Back to the generated demo data (drops injected transactions)."""
        with self.lock:
            self.raw = self.base.clone()
            self.pos = {}
            self.t = from_uz(self.raw)
            X = build(self.t)
            self.s1 = self.m.lgbm.stage1.predict(X)   # first-stage scores: the accounts' earlier risk for stage 2
            self.X = self.m.lgbm.features(self.t, X)
            self.raw_l = self.m.lgbm.raw(self.X)
            self.E = gnn.edge_attrs(self.t)   # per-transaction GNN inputs, extended as transactions arrive
            self.explainer = Explainer(self.m.lgbm.columns, self.X)
            self._rescore_graph()
            self.injected = []

    def _rescore_graph(self):
        self.p_l, self.p_g, self.p = self.m.scores(self.raw_l, self.m.gnn_raw(self.t, self.E))
        order = np.argsort(-self.p, kind='stable')
        self.rank = np.empty(len(order), dtype=np.int64); self.rank[order] = np.arange(len(order))
        n = len(order)
        self.level = np.where(self.rank < HIGH * n, 'high', np.where(self.rank < MEDIUM * n, 'medium', 'low'))
        self.order = order
        if len(getattr(self, 'pos', {})) != self.t.height:   # rows only ever get appended
            for i, r in enumerate(self.t['row'][len(getattr(self, 'pos', {})):].to_list(), start=len(getattr(self, 'pos', {}))):
                self.pos[r] = i

    def now(self):
        return self.raw['ts'].max()

    # ------------------------------------------------------------------ live input
    def inject(self, rows):
        """Append new transactions (UZ columns, timestamps after every existing one) and score them.
        Features need only the two accounts' own histories, so only those are recomputed; the GNN
        re-reads the whole graph, which also updates the network view of every other transaction."""
        with self.lock:
            w0, c0 = time.perf_counter(), time.process_time()
            start = int(self.raw['row'].max()) + 1
            rows = rows.with_columns(row=pl.int_range(start, start + rows.height, dtype=pl.Int64)).select(self.raw.columns)
            self.raw = pl.concat([self.raw, rows])
            t = from_uz(self.raw)
            new = rows['row'].to_list()
            fresh = t.filter(pl.col('row').is_in(new))
            accts = set(fresh['src'].to_list()) | set(fresh['dst'].to_list())
            sub = t.filter(pl.col('src').is_in(list(accts)) | pl.col('dst').is_in(list(accts)))
            Xs = build(sub)
            sub_rows = sub['row'].to_numpy(); is_new = np.isin(sub_rows, new)
            Xn = Xs[is_new].reset_index(drop=True)
            # stage 2 needs the first-stage scores of every earlier payment of these accounts: kept from the
            # full graph for existing rows, computed here for the new ones
            s1_new = self.m.lgbm.stage1.predict(Xn)
            s1_sub = np.empty(len(sub_rows))
            s1_sub[is_new] = s1_new
            s1_sub[~is_new] = self.s1[[self.pos[r] for r in sub_rows[~is_new]]]
            Xn = pd.concat([Xn, risk_history(sub, s1_sub)[is_new].reset_index(drop=True)], axis=1)
            assert t['row'].to_list()[-len(new):] == new, 'injected transactions must be the latest ones'
            self.t = t
            self.X = pd.concat([self.X, Xn], ignore_index=True)
            self.s1 = np.concatenate([self.s1, s1_new])
            self.E = np.concatenate([self.E, gnn.edge_attrs(fresh)])
            self.raw_l = np.concatenate([self.raw_l, self.m.lgbm.raw(Xn)])
            self._rescore_graph()
            self.injected += new
            wall, cpu = (time.perf_counter() - w0) * 1000, (time.process_time() - c0) * 1000
            self.latency.append((wall / len(new), cpu / len(new)))
            return {'rows': new, 'ms_total': round(wall, 1), 'ms_per_tx': round(wall / len(new), 1)}

    def score_frame(self, df, lang='en'):
        """Score a separate dataset (a jury CSV) on its own graph; nothing is merged into the demo bank."""
        w0, c0 = time.perf_counter(), time.process_time()
        t = from_uz(df.with_row_index('row').with_columns(pl.col('row').cast(pl.Int64)))
        X = self.m.lgbm.features(t, build(t))
        p_l, p_g, p = self.m.scores(self.m.lgbm.raw(X), self.m.gnn_raw(t))
        wall, cpu = (time.perf_counter() - w0) * 1000, (time.process_time() - c0) * 1000
        self.latency.append((wall / t.height, cpu / t.height))
        order = np.argsort(-p, kind='stable')[:50]
        contrib = self.m.lgbm.contributions(X.iloc[order[:10]])
        ex = Explainer(self.m.lgbm.columns, X)
        out = []
        for j, i in enumerate(order):
            r = t.row(int(i), named=True)
            item = self._row_dict(r, p[i], p_l[i], p_g[i], None)
            if j < 10:
                item['reasons'] = ex.reasons(contrib[j], X.iloc[int(i)].to_dict(), r, top=2, lang=lang)
            out.append(item)
        return {'n': t.height, 'ms_total': round(wall, 1), 'ms_per_tx': round(wall / t.height, 2), 'top': out,
                'flagged': int((p >= np.quantile(p, 1 - MEDIUM)).sum()) if t.height > 50 else None}

    # ------------------------------------------------------------------ views
    def _row_dict(self, r, p, pl_, pg, level):
        return {'row': int(r['row']), 'ts': r['ts'], 'src': r['src'], 'dst': r['dst'],
                'src_short': short(r['src']), 'dst_short': short(r['dst']), 'amount_uzs': float(r['amount_uzs']),
                'rail': r['rail'], 'prob': float(p), 'p_lgbm': float(pl_), 'p_gnn': float(pg), 'level': level}

    def queue(self, level=None, limit=50, offset=0):
        idx = self.order if level is None else self.order[self.level[self.order] == level]
        decided = self.audit.latest_by_alert()
        items = []
        for i in idx[offset:offset + limit]:
            r = self.t.row(int(i), named=True)
            d = self._row_dict(r, self.p[i], self.p_l[i], self.p_g[i], self.level[i])
            d['decision'] = decided.get(d['row'], {}).get('action')
            d['injected'] = d['row'] in self.injected
            items.append(d)
        counts = {k: int((self.level == k).sum()) for k in ('high', 'medium', 'low')}
        return {'items': items, 'counts': counts, 'total': len(idx)}

    def alert(self, row, lang='en'):
        i = self.pos[row]
        r = self.t.row(i, named=True)
        contrib = self.m.lgbm.contributions(self.X.iloc[[i]])[0]
        d = self._row_dict(r, self.p[i], self.p_l[i], self.p_g[i], self.level[i])
        d.update(reasons=self.explainer.reasons(contrib, self.X.iloc[i].to_dict(), r, lang=lang),
                 graph=subgraph(self.t, self.p, r['src'], r['dst'], r['ts'], row),
                 rank=int(self.rank[i]) + 1, n=len(self.rank), amount_usd=float(r['amount_usd']),
                 src_info=self.owner(r['src']), dst_info=self.owner(r['dst']),
                 decision=self.audit.latest_by_alert().get(row), injected=row in self.injected)
        return d

    def owner(self, node):
        a = self.accounts.filter(pl.col('node') == node)
        if a.height:
            x = a.row(0, named=True)
            return {'name': x['owner'], 'city': x['city'], 'persona': x['persona'], 'bank': x['bank']}
        return {'name': None, 'bank': node.split(':')[0]}

    def account(self, node):
        mask = ((self.t['src'] == node) | (self.t['dst'] == node)).to_numpy()
        idx = np.where(mask)[0]
        rows = [self._row_dict(self.t.row(int(i), named=True), self.p[i], self.p_l[i], self.p_g[i], self.level[i]) for i in idx]
        for x in rows: x['direction'] = 'out' if x['src'] == node else 'in'
        return {'node': node, 'info': self.owner(node), 'rows': rows[::-1][:300], 'n': len(rows),
                'max_prob': max((x['prob'] for x in rows), default=0.0)}

    def personas(self):
        """A few accounts per everyday persona, plus the accounts the model finds riskiest."""
        out = {}
        for kind, grp in self.accounts.group_by('persona', maintain_order=True):
            out[kind[0]] = [{'node': x['node'], 'name': x['owner'], 'city': x['city']} for x in grp.head(4).iter_rows(named=True)]
        risky = {}
        for i in self.order[:400]:
            for node in (self.t['src'][int(i)], self.t['dst'][int(i)]):
                if node in self.accounts['node'] and node not in risky:
                    risky[node] = float(self.p[i])
        return out, [{'node': n, 'prob': p, **self.owner(n)} for n, p in list(risky.items())[:8]]

    def live_latency(self):
        if not self.latency:
            return None
        a = np.array(self.latency)
        cpu = float(a[:, 1].mean())
        return {'n': len(a), 'p50_ms': float(np.percentile(a[:, 0], 50)), 'p95_ms': float(np.percentile(a[:, 0], 95)),
                'cpu_ms': cpu, 'usd_per_million': cpu / 1000 / 3600 * VM_USD_PER_HOUR / VM_CORES * 1e6}


def short(node):
    bank, acct = node.split(':', 1)
    return f"{bank} ·{acct[-4:]}"


def scheme_rows(kind, target, now, accounts, rng, n=10):
    """Rows for the 'inject a scheme' button: the jury picks a target account and a typology;
    counterparties are other demo accounts. The model is not told what the rows are."""
    others = accounts.filter(pl.col('node') != target).sample(n + 1, seed=int(rng.integers(1 << 30)))
    tb, ta = target.split(':', 1)
    ob, oa = others['bank'].to_list(), others['account'].to_list()
    rows, t = [], now
    step = lambda: timedelta(minutes=float(rng.uniform(2, 15)))
    if kind == 'fan_in':
        total = 0.0
        for b, a in zip(ob[:n], oa[:n]):
            t += step(); amt = round(float(rng.lognormal(np.log(1.8e6), 0.3)), -3); total += amt
            rows.append((t, b, a, tb, ta, amt, str(rng.choice(['Click', 'Payme', 'Humo', 'Uzcard']))))
        t += step(); rows.append((t, tb, ta, tb, f'ATM-{rng.integers(10**8, 10**9)}', round(total * 0.95, -3), 'Cash'))
    elif kind == 'fan_out':
        amt = float(rng.lognormal(np.log(3e6), 0.2))
        for b, a in zip(ob[:n], oa[:n]):
            t += step(); rows.append((t, tb, ta, b, a, round(amt * rng.uniform(0.9, 1.1), -3), str(rng.choice(['Click', 'Payme']))))
    else:   # cycle through four accounts and back
        ring = [(tb, ta)] + list(zip(ob[:3], oa[:3]))
        amt = float(rng.lognormal(np.log(4e7), 0.2))
        for (b1, a1), (b2, a2) in zip(ring, ring[1:] + ring[:1]):
            t += timedelta(minutes=float(rng.uniform(20, 90))); amt *= 0.98
            rows.append((t, b1, a1, b2, a2, round(amt, -3), 'Bank transfer'))
    return pl.DataFrame(rows, schema=['ts', 'src_bank', 'src_account', 'dst_bank', 'dst_account', 'amount_uzs', 'rail'],
                        orient='row').with_columns(pl.col('ts').cast(pl.Datetime('us')))
