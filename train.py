"""Train the LightGBM baseline and the GNN on IBM AMLworld, evaluate them on the last 20% of time,
and write the artifacts the app serves.

    python train.py                      # data/raw/amlworld/HI-Small_Trans.csv
    python train.py --rows 2000000       # quicker run on the earliest transactions
    python train.py --out models_b       # write the artifacts somewhere else

Writes to models/: baseline.joblib, gnn.pt, gnn_meta.joblib, ensemble.joblib, metrics.json.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from src import baseline, evaluate, gnn
from src.features import build
from src.ingest import read_amlworld, temporal_split


def pr_points(y, prob, n=150):
    p, r, _ = precision_recall_curve(y, prob)
    keep = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return {'precision': p[keep].round(4).tolist(), 'recall': r[keep].round(4).tolist()}


def main(a):
    t0 = time.time(); timing = {}
    log = lambda s: print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)
    t = read_amlworld(a.data, n_rows=a.rows)
    y = t['label'].to_numpy().astype(np.int8)
    tr, va, te = temporal_split(t)
    log(f'{t.height:,} transactions, {int(y.sum()):,} laundering ({y.mean():.4%}); '
        f'train/val/test {len(tr):,}/{len(va):,}/{len(te):,} (split by time)')
    timing['load_s'] = time.time() - t0

    t1 = time.time(); X = build(t); timing['features_s'] = time.time() - t1
    log(f'features {X.shape}')

    # 1) LightGBM on backward-looking transaction features
    t1 = time.time()
    lgbm = baseline.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], log=log)
    p_va = {'lgbm': lgbm.predict(X.iloc[va])}; p_te = {'lgbm': lgbm.predict(X.iloc[te])}
    timing['lgbm_train_s'] = time.time() - t1
    importance = dict(zip(lgbm.columns, lgbm.booster.feature_importance('gain').round(1).tolist()))
    del X

    # 2) GNN over the account graph (graph of each period = all transactions up to its end, labels hidden)
    t1 = time.time()
    e = gnn.edge_attrs(t); upto_va = va[-1] + 1
    g_tr = gnn.Graph(t[tr]); g_va = gnn.Graph(t[:upto_va], g_tr.scaler); g_te = gnn.Graph(t, g_tr.scaler)
    log(f'graphs: train {g_tr.x.shape[0]:,} accounts / {g_tr.edge_index.shape[1]:,} account pairs; '
        f'test {g_te.x.shape[0]:,} / {g_te.edge_index.shape[1]:,}')
    model, _ = gnn.train(g_tr, e[tr], y[tr], g_va, e[:upto_va], y[va], va, steps=a.gnn_steps, log=log)
    r_va, r_te = gnn.predict(model, g_va, e[:upto_va], va), gnn.predict(model, g_te, e, te)
    cal_g = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(r_va, y[va])
    p_va['gnn'], p_te['gnn'] = cal_g.predict(r_va), cal_g.predict(r_te)
    timing['gnn_train_s'] = time.time() - t1

    # 3) Ensemble: mean of the two calibrated probabilities, re-calibrated on the validation period
    cal_e = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit((p_va['lgbm'] + p_va['gnn']) / 2, y[va])
    p_va['ensemble'] = cal_e.predict((p_va['lgbm'] + p_va['gnn']) / 2)
    p_te['ensemble'] = cal_e.predict((p_te['lgbm'] + p_te['gnn']) / 2)

    results = {}
    for name in ['lgbm', 'gnn', 'ensemble']:
        thr = evaluate.best_f1_threshold(y[va], p_va[name])
        results[name] = {**evaluate.report(y[te], p_te[name], thr), 'pr_curve': pr_points(y[te], p_te[name])}
        log(f'{name:9s} test PR-AUC {results[name]["pr_auc"]:.4f}  ROC-AUC {results[name]["roc_auc"]:.4f}  '
            f'F1 {results[name]["f1_at_val_threshold"]["f1"]:.4f}')
    timing['total_s'] = time.time() - t0
    timing['max_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    out = Path(a.out); out.mkdir(exist_ok=True)
    joblib.dump(lgbm, out / 'baseline.joblib')
    torch.save(model.state_dict(), out / 'gnn.pt')
    joblib.dump({'scaler': g_tr.scaler, 'n_node': g_tr.x.shape[1], 'n_edge': e.shape[1], 'calibrator': cal_g}, out / 'gnn_meta.joblib')
    joblib.dump(cal_e, out / 'ensemble.joblib')
    metrics = {'data': {'source': 'IBM AMLworld HI-Small', 'n': t.height, 'laundering': int(y.sum()),
                        'split': {'train': len(tr), 'val': len(va), 'test': len(te)},
                        'test_period': [str(t['ts'][int(te[0])]), str(t['ts'][int(te[-1])])]},
               'models': results, 'calibration': evaluate.calibration(y[te], p_te['ensemble']),
               'importance': dict(sorted(importance.items(), key=lambda kv: -kv[1])[:20]),
               'timing': {k: round(v, 1) for k, v in timing.items()}}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    log(f'artifacts written to {out}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/raw/amlworld/HI-Small_Trans.csv')
    ap.add_argument('--rows', type=int, default=None, help='only the first N transactions of the file')
    ap.add_argument('--gnn-steps', type=int, default=300)
    ap.add_argument('--out', default='models')
    main(ap.parse_args())
