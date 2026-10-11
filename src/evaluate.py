"""Metrics for rare-event detection: PR-AUC, alert-budget precision/recall, calibration."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def at_budget(y, score, shares=(0.001, 0.005, 0.01, 0.02)):
    """Review the top X% of transactions by score -> precision and recall of laundering among them."""
    order = np.argsort(-score, kind='stable'); ys = y[order]; rows = []
    for s in shares:
        k = max(1, int(round(s * len(y)))); hit = int(ys[:k].sum())
        rows.append({'top_share': s, 'alerts': k, 'caught': hit, 'precision': hit / k, 'recall': hit / max(1, int(y.sum()))})
    return rows


def best_f1_threshold(y, prob):
    """Threshold with the best F1 on the data given (use the validation period, then apply it to test)."""
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def f1_at(y, prob, thr):
    pred = prob >= thr; tp = int((pred & (y == 1)).sum())
    prec = tp / max(1, int(pred.sum())); rec = tp / max(1, int(y.sum()))
    return {'threshold': thr, 'precision': prec, 'recall': rec, 'f1': 2 * prec * rec / max(prec + rec, 1e-12)}


def calibration(y, prob, edges=(0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0001)):
    """Mean predicted vs observed rate in fixed probability bins (rare events: most mass sits near 0)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum() >= 20:
            out.append({'pred': float(prob[m].mean()), 'obs': float(y[m].mean()), 'n': int(m.sum())})
    return out


def pr_points(y, prob, n=120):
    """Precision-recall curve on an even recall grid (interpolated precision: best precision at recall >= r)."""
    p, r, _ = precision_recall_curve(y, prob)
    grid = np.linspace(0, 1, n)
    best = np.array([p[r >= g].max() if (r >= g).any() else 0.0 for g in grid])
    return {'precision': best.round(4).tolist(), 'recall': grid.round(4).tolist()}


def report(y, prob, thr):
    return {'pr_auc': float(average_precision_score(y, prob)), 'roc_auc': float(roc_auc_score(y, prob)),
            'base_rate': float(y.mean()), 'n': int(len(y)), 'positives': int(y.sum()),
            'budget': at_budget(y, prob), 'f1_at_val_threshold': f1_at(y, prob, thr)}
