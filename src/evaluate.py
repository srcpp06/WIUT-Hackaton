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


def calibration(y, prob, bins=10):
    """Mean predicted vs observed rate in equal-count bins of the predicted probability."""
    order = np.argsort(prob, kind='stable'); out = []
    for idx in np.array_split(order, bins):
        out.append({'pred': float(prob[idx].mean()), 'obs': float(y[idx].mean()), 'n': int(len(idx))})
    return out


def report(y, prob, thr):
    return {'pr_auc': float(average_precision_score(y, prob)), 'roc_auc': float(roc_auc_score(y, prob)),
            'base_rate': float(y.mean()), 'n': int(len(y)), 'positives': int(y.sum()),
            'budget': at_budget(y, prob), 'f1_at_val_threshold': f1_at(y, prob, thr)}
