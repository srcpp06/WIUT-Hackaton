"""Transaction features that only look backwards in time.

Every transaction is described by
  * its own attributes: USD amount, payment rail, hour, weekday, same bank, self-transfer;
  * the sender's and the receiver's recent history *before* this transaction: how many payments
    went in and out in the last 24 hours and 7 days, how much money, and how many different
    counterparties (fan-in / fan-out, the shape of mule and laundering schemes);
  * pass-through timing: how soon money leaves an account after money arrived;
  * the relationship: has this sender paid this receiver before?

Windows are closed on the left, so a transaction never sees itself or anything later. The same
code featurises the 5M-row training data and a single transaction injected live in the demo.
"""
import numpy as np
import polars as pl

from .ingest import FORMATS

WINDOWS = {'1d': '1d', '7d': '7d'}
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


def history(ev, chunks=1):
    """Rolling per-account statistics over the events strictly before each event. Accounts are
    processed in `chunks` groups to bound memory (an account's events always stay in one group)."""
    if chunks > 1:
        return pl.concat([history(ev.filter(pl.col('acct') % chunks == k)) for k in range(chunks)])
    parts = []
    for name, period in WINDOWS.items():
        i, o = pl.col('is_in') == 1, pl.col('is_in') == 0
        r = ev.rolling(index_column='ts', period=period, group_by='acct', closed='left').agg(
            pl.col('tx_id').last().alias('_last'),   # placeholder so empty windows still yield a row
            i.sum().alias(f'n_in_{name}'), o.sum().alias(f'n_out_{name}'),
            pl.col('amount_usd').filter(i).sum().alias(f'usd_in_{name}'),
            pl.col('amount_usd').filter(o).sum().alias(f'usd_out_{name}'),
            pl.col('cp').filter(i).n_unique().alias(f'cp_in_{name}'),
            pl.col('cp').filter(o).n_unique().alias(f'cp_out_{name}'))
        r = r.sort('acct', 'ts', maintain_order=True)   # rolling() may emit the accounts in any order
        assert r['acct'].equals(ev['acct']) and r['ts'].equals(ev['ts']), 'rolling output is not aligned with the events'
        parts.append(r.drop('acct', 'ts', '_last'))
    # aligned on (account, time); events tied on both have identical windows, so their order does not matter
    base = ev.select('tx_id', 'is_in', 'acct', 'ts')
    h = pl.concat([base] + parts, how='horizontal')
    # timing within the account: previous event of any kind, previous incoming money, first ever event
    sec = pl.col('ts').dt.epoch('s')
    h = h.with_columns(
        prev_any=sec - sec.shift(1).over('acct'),
        prev_in=sec - pl.when(pl.col('is_in') == 1).then(sec).shift(1).forward_fill().over('acct'),
        age=sec - sec.min().over('acct'),
        n_before=pl.int_range(pl.len()).over('acct'))
    return h


def build(t):
    """Feature matrix (pandas, one row per transaction in t, same order)."""
    c = compact(t)
    h = history(account_events(c), chunks=max(1, t.height // 1_000_000))
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
