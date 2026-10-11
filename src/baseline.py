"""LightGBM baseline: transaction features -> calibrated probability of money laundering.

Two stages: stage 1 scores each payment from its own features; stage 2 also sees each account's
earlier risk (the stage-1 scores of the payments it recently sent and received, features.risk_history).

Training keeps every laundering transaction and a random 20% of the normal ones (5M rows would
otherwise be slow on a CPU). Undersampling leaves the ranking intact, and the raw scores are
re-calibrated on the untouched validation period, so the probabilities match real frequencies.
"""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .features import CATEGORICAL, with_risk

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


class TwoStage:
    """Stage 1 on the transaction features, stage 2 with the accounts' earlier risk added; the app uses
    stage 2's scores, calibration and TreeSHAP values."""

    def __init__(self, stage1, stage2):
        self.stage1, self.stage2 = stage1, stage2
        self.booster, self.calibrator, self.columns = stage2.booster, stage2.calibrator, stage2.columns

    def features(self, t, X):
        """Stage-2 features of every transaction in t, from X = features.build(t)."""
        return with_risk(t, X, self.stage1.predict(X))

    def raw(self, X):
        return self.stage2.raw(X)

    def predict(self, X):
        return self.stage2.predict(X)

    def contributions(self, X):
        return self.stage2.contributions(X)


def fit_two_stage(t, X, y, tr, va, folds=3, log=print):
    """Training rows get out-of-fold stage-1 scores (stage 1 refitted without each of `folds` time blocks),
    so stage 2 learns from scores like those it sees on new data; all later rows are scored by the full
    stage 1. Returns the model and the stage-2 feature matrix of t."""
    stage1 = fit(X.iloc[tr], y[tr], X.iloc[va], y[va], log=log)
    s = np.zeros(len(X))
    rest = np.setdiff1d(np.arange(len(X)), tr)
    s[rest] = stage1.predict(X.iloc[rest])
    for k, block in enumerate(np.array_split(tr, folds)):
        other = np.setdiff1d(tr, block)
        s[block] = fit(X.iloc[other], y[other], X.iloc[va], y[va], log=lambda _: None).predict(X.iloc[block])
        log(f'stage 1, fold {k + 1}/{folds}: out-of-fold scores for {len(block):,} training rows')
    X2 = with_risk(t, X, s)
    stage2 = fit(X2.iloc[tr], y[tr], X2.iloc[va], y[va], log=log)
    return TwoStage(stage1, stage2), X2
