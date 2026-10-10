"""LightGBM baseline: transaction features -> calibrated probability of money laundering.

Training keeps every laundering transaction and a random 20% of the normal ones (5M rows would
otherwise be slow on a CPU). Undersampling leaves the ranking intact, and the raw scores are
re-calibrated on the untouched validation period, so the probabilities match real frequencies.
"""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .features import CATEGORICAL

PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, metric='average_precision',
              verbose=-1, n_jobs=-1, seed=42, deterministic=True, force_row_wise=True)


class Baseline:
    def __init__(self, booster, calibrator, columns):
        self.booster, self.calibrator, self.columns = booster, calibrator, columns

    def raw(self, X):
        return self.booster.predict(X[self.columns].to_numpy(np.float32), raw_score=True)

    def predict(self, X):
        return self.calibrator.predict(self.raw(X))

    def contributions(self, X):
        """Exact TreeSHAP values per feature (last column = base value), computed by LightGBM itself."""
        return self.booster.predict(X[self.columns].to_numpy(np.float32), pred_contrib=True)


def fit(X_tr, y_tr, X_va, y_va, neg_share=0.2, rounds=3000, log=print):
    keep = (y_tr == 1) | (np.random.default_rng(42).random(len(y_tr)) < neg_share)
    cat = [c for c in CATEGORICAL if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr[keep], y_tr[keep], categorical_feature=cat, free_raw_data=True)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat, reference=dtr)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)])
    log(f'LightGBM: {m.best_iteration} trees, validation PR-AUC {m.best_score["valid_0"]["average_precision"]:.4f}')
    booster = lgb.Booster(model_str=m.model_to_string(num_iteration=m.best_iteration))
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(
        booster.predict(X_va.to_numpy(np.float32), raw_score=True), y_va)
    return Baseline(booster, cal, list(X_tr.columns))
