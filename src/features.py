"""Transaction features that only look backwards in time.

Every transaction is described by
  * its own attributes: USD amount, payment rail, hour, weekday, same bank, self-transfer;
  * the sender's and the receiver's recent history *before* this transaction: how many payments
    went in and out in the last 24 hours and 7 days, how much money, and how many *new*
    counterparties (accounts never paid / paid by before: fan-in and fan-out of mule schemes);
  * pass-through timing: how soon money leaves an account after money arrived;
  * the relationship: has this sender paid this receiver before?
  * (second stage) each account's earlier risk: first-stage scores of the payments it recently sent
    and received, so money moving on from a risky payment is visible (layering).

Windows end just before the transaction's own second, so it never sees itself or anything later. The same
code featurises the 5M-row training data and a single transaction injected live in the demo.
"""
import numpy as np
import pandas as pd
import polars as pl

from .ingest import FORMATS

SIDE_STATS = ['n_in', 'n_out', 'usd_in', 'usd_out', 'cp_in', 'cp_out']


def compact(t):
    """Transactions with integer account ids (s, d) and 32-bit numbers: the 5M-row training set fits in memory."""
    ids = pl.concat([t['src'], t['dst']]).unique().to_frame('acct').with_row_index('aid')
    c = t.select(pl.col('tx_id').cast(pl.UInt32), 'ts', pl.col('amount_usd').cast(pl.Float32), 'src', 'dst')
    c = c.join(ids.rename({'acct': 'src', 'aid': 's'}), on='src').join(ids.rename({'acct': 'dst', 'aid': 'd'}), on='dst')
    return c.drop('src', 'dst')


def account_events(c):
    """Each transaction seen from both ends: an 'out' event for the sender and an 'in' event for the receiver."""
    cols = ['tx_id', 'ts', 'amount_usd']
    out = c.select(*cols, acct=pl.col('s'), cp=pl.col('d'), is_in=pl.lit(0, pl.Int8))
    inn = c.select(*cols, acct=pl.col('d'), cp=pl.col('s'), is_in=pl.lit(1, pl.Int8))
    return pl.concat([out, inn]).sort('acct', 'ts', 'tx_id', 'is_in')   # self-transfer: out before in, deterministic


WINDOW_SECONDS = {'1d': 86_400, '7d': 7 * 86_400}


def history(ev):
    """Per-account statistics over the events strictly before each event (same second excluded).
    ev must be sorted by (acct, ts). Window sums come from cumulative sums and a binary search
    for each window's start, so memory stays linear even for hub accounts with 100k payments."""
    acct = ev['acct'].to_numpy().astype(np.int64)
    sec = ev['ts'].dt.epoch('s').to_numpy().astype(np.int64)
    sec = sec - sec.min()
    key = acct * 10**10 + sec                          # sorted, because ev is sorted by (acct, ts)
    first_of_acct = np.searchsorted(key, acct * 10**10, 'left')
    stop = np.searchsorted(key, key, 'left')            # events of this account strictly before this second
    is_in = ev['is_in'].to_numpy().astype(bool)
    usd = ev['amount_usd'].to_numpy().astype(np.float64)
    # first time this account meets this counterparty in this direction: "new counterparty"
    new_cp = ev.select((pl.int_range(pl.len()).over('acct', 'cp', 'is_in') == 0).alias('f'))['f'].to_numpy()
    cum = lambda v: np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    sums = {'n_in': cum(is_in), 'n_out': cum(~is_in), 'usd_in': cum(usd * is_in), 'usd_out': cum(usd * ~is_in),
            'cp_in': cum(new_cp & is_in), 'cp_out': cum(new_cp & ~is_in)}
    cols = {}
    for name, period in WINDOW_SECONDS.items():
        begin = np.maximum(np.searchsorted(key, key - period, 'left'), first_of_acct)
        for stat, c in sums.items():
            cols[f'{stat}_{name}'] = (c[stop] - c[begin]).astype(np.float32)
    h = ev.select('tx_id', 'is_in', 'acct', 'ts').with_columns(**{k: pl.Series(v) for k, v in cols.items()})
    # timing within the account: previous event of any kind, previous incoming money, first ever event
    s = pl.col('ts').dt.epoch('s')
    return h.with_columns(
        prev_any=s - s.shift(1).over('acct'),
        prev_in=s - pl.when(pl.col('is_in') == 1).then(s).shift(1).forward_fill().over('acct'),
        age=s - s.min().over('acct'),
        n_before=pl.Series(np.arange(len(acct)) - first_of_acct))


RISK_THRESHOLD = 0.1


def risk_history(t, s):
    """Each account's earlier risk: first-stage scores s (aligned with t's rows) of the payments the sender
    and the receiver sent and received before this transaction, summed and counted above RISK_THRESHOLD,
    over 24 hours and 7 days. Money that just arrived from a risky payment makes the onward payment
    suspicious too (layering), which the transaction's own features cannot see."""
    ev = account_events(compact(t))
    acct = ev['acct'].to_numpy().astype(np.int64)
    sec = ev['ts'].dt.epoch('s').to_numpy().astype(np.int64)
    key = acct * 10**10 + (sec - sec.min())
    first_of_acct = np.searchsorted(key, acct * 10**10, 'left')
    stop = np.searchsorted(key, key, 'left')
    pos = np.full(int(t['tx_id'].max()) + 1, -1, dtype=np.int64)
    pos[t['tx_id'].to_numpy().astype(np.int64)] = np.arange(t.height)
    score = np.asarray(s, dtype=np.float64)[pos[ev['tx_id'].to_numpy().astype(np.int64)]]
    is_in = ev['is_in'].to_numpy().astype(bool); high = score > RISK_THRESHOLD
    cum = lambda v: np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    sums = {'risk_in': cum(score * is_in), 'risk_out': cum(score * ~is_in), 'hi_in': cum(high & is_in), 'hi_out': cum(high & ~is_in)}
    cols = {}
    for name, period in WINDOW_SECONDS.items():
        begin = np.maximum(np.searchsorted(key, key - period, 'left'), first_of_acct)
        for stat, c in sums.items():
            cols[f'{stat}_{name}'] = (c[stop] - c[begin]).astype(np.float32)
    h = pl.DataFrame({'tx_id': ev['tx_id'], 'is_in': ev['is_in'], **cols})
    snd = h.filter(pl.col('is_in') == 0).select('tx_id', *[pl.col(k).alias(f's_{k}') for k in cols])
    rcv = h.filter(pl.col('is_in') == 1).select('tx_id', *[pl.col(k).alias(f'r_{k}') for k in cols])
    return snd.join(rcv, on='tx_id').sort('tx_id').drop('tx_id').to_pandas()


def with_risk(t, X, s):
    """Stage-two feature matrix: build(t) plus risk_history(t, s)."""
    return pd.concat([X, risk_history(t, s)], axis=1)


def build(t):
    """Feature matrix (pandas, one row per transaction in t, same order)."""
    c = compact(t)
    h = history(account_events(c))
    side = [x for x in h.columns if x not in ('tx_id', 'is_in', 'acct', 'ts')]
    h = h.with_columns(pl.col(x).cast(pl.Float32) for x in side)
    snd = h.filter(pl.col('is_in') == 0).select('tx_id', *[pl.col(x).alias(f's_{x}') for x in side])
    rcv = h.filter(pl.col('is_in') == 1).select('tx_id', *[pl.col(x).alias(f'r_{x}') for x in side])
    del h
    sec = pl.col('ts').dt.epoch('s')
    pair = c.sort('ts', 'tx_id').with_columns(
        pair_n_before=pl.int_range(pl.len()).over('s', 'd').cast(pl.Float32),
        pair_gap=(sec - sec.shift(1).over('s', 'd')).cast(pl.Float32)).select('tx_id', 'pair_n_before', 'pair_gap')
    del c
    usd = pl.col('amount_usd')
    X = t.select(
        pl.col('tx_id').cast(pl.UInt32), 'amount_usd',
        log_usd=usd.log1p(),
        hour=pl.col('ts').dt.hour(), weekday=pl.col('ts').dt.weekday(),
        same_bank=(pl.col('src_bank') == pl.col('dst_bank')).cast(pl.Int8),
        self_loop=(pl.col('src') == pl.col('dst')).cast(pl.Int8),
        fmt=pl.col('fmt').replace_strict({f: i for i, f in enumerate(FORMATS)}, default=-1, return_dtype=pl.Int8),
    ).join(snd, on='tx_id', how='left').join(rcv, on='tx_id', how='left').join(pair, on='tx_id', how='left')
    X = X.with_columns(   # money arriving vs leaving the sender: pass-through ratio, and the payment's share of it
        s_pass_1d=pl.col('s_usd_out_1d') / (pl.col('s_usd_in_1d') + 1.0),
        amt_vs_s_in_1d=usd / (pl.col('s_usd_in_1d') + 1.0))
    X = X.sort('tx_id').drop('tx_id', 'amount_usd').to_pandas()
    for c in X.columns:
        if c.endswith(('prev_any', 'prev_in', 'age', 'pair_gap')):
            X[c] = np.log1p(X[c].astype('float64'))   # seconds -> log scale; NaN when there is no earlier event
    return X.astype('float32')


CATEGORICAL = ['fmt']
