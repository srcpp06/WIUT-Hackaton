"""Why an alert was raised: LightGBM TreeSHAP grouped into plain-English reasons, each with the facts
behind it, plus the evidence subgraph (the transactions around the two accounts)."""
from datetime import timedelta

import numpy as np
import polars as pl

from .features import RISK_THRESHOLD
from .ingest import UZS_PER_USD

# (key, title, rule on the feature name); the first matching rule wins
GROUPS = [
    ('fan_in', 'Receiver is collecting money from many new accounts', lambda c: c.startswith(('r_cp_in', 'r_n_in'))),
    ('fan_out', 'Sender is spreading money across many new accounts', lambda c: c.startswith(('s_cp_out', 's_n_out'))),
    ('pass', 'Money passes through the account quickly',
     lambda c: c in ('s_prev_in', 's_pass_1d', 'amt_vs_s_in_1d', 'r_prev_in') or c.startswith(('s_usd_in', 'r_usd_out'))),
    ('relation', 'Payment history between sender and receiver', lambda c: c.startswith('pair_')),
    ('history', 'Age and activity of the accounts', lambda c: c.endswith(('_age', '_n_before', '_prev_any'))),
    ('amount', 'Payment amount', lambda c: c == 'log_usd'),
    ('rail', 'Payment channel', lambda c: c == 'fmt'),
    ('time', 'Time of payment', lambda c: c in ('hour', 'weekday')),
    ('structure', 'Same-bank or own-account transfer', lambda c: c in ('same_bank', 'self_loop')),
    ('risk', "Risk of the accounts' earlier payments", lambda c: c.startswith(('s_risk', 'r_risk', 's_hi_', 'r_hi_'))),
    ('volume', 'Recent turnover of the accounts', lambda c: True),
]
TITLES = {k: t for k, t, _ in GROUPS}
WEEKDAYS = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}


def group_of(c):
    return next(k for k, _, rule in GROUPS if rule(c))


def dur(seconds):
    """Seconds -> short duration."""
    if seconds is None or not np.isfinite(seconds): return 'no data'
    m = seconds / 60
    if m < 1: return 'under a minute'
    if m < 120: return f'{m:.0f} min'
    if m < 2880: return f'{m / 60:.1f} hours'
    return f'{m / 1440:.1f} days'


def count(n, word):
    return f"{n:.0f} {word}{'' if n == 1 else 's'}"


def uzs(usd):
    return f'{usd * UZS_PER_USD:,.0f} UZS'


class Explainer:
    def __init__(self, columns, X_ref):
        self.columns = list(columns)
        self.groups = np.array([group_of(c) for c in self.columns])
        self.median = X_ref.median(numeric_only=True).to_dict()

    def reasons(self, contrib, x, row, top=4):
        """contrib: TreeSHAP row (features + base value); x: feature dict; row: the transaction (dict)."""
        sums = {k: float(contrib[:-1][self.groups == k].sum()) for k, _, _ in GROUPS if (self.groups == k).any()}
        ranked = sorted(sums, key=lambda k: -abs(sums[k]))[:top]
        return [{'group': k, 'title': TITLES[k], 'impact': round(sums[k], 3), 'up': sums[k] > 0,
                 'evidence': self.evidence(k, x, row)} for k in ranked]

    def evidence(self, k, x, row):
        g = lambda f: float(x.get(f, np.nan))
        med = lambda f: self.median.get(f, np.nan)
        if k == 'fan_in':
            return (f"The receiver got {count(g('r_n_in_1d'), 'payment')} in the last 24 hours; {g('r_cp_in_1d'):.0f} of them came from "
                    f"accounts that had never paid it before ({g('r_cp_in_7d'):.0f} in 7 days; a typical account has {med('r_cp_in_7d'):.0f}).")
        if k == 'fan_out':
            return (f"The sender made {count(g('s_n_out_1d'), 'payment')} in the last 24 hours; {g('s_cp_out_1d'):.0f} of them went to "
                    f"accounts it had never paid before ({g('s_cp_out_7d'):.0f} in 7 days; a typical account has {med('s_cp_out_7d'):.0f}).")
        if k == 'pass':
            if not np.isfinite(g('s_prev_in')):
                return "No money had come into the sender's account before: this is one of its first outgoing payments."
            return (f"Money last came into the sender's account {dur(np.expm1(g('s_prev_in')))} ago; in 24 hours "
                    f"{uzs(g('s_usd_in_1d'))} came in and {uzs(g('s_usd_out_1d'))} went out.")
        if k == 'relation':
            n = g('pair_n_before')
            return ('The first payment between these two accounts (a new counterparty).' if n == 0
                    else f"These two accounts have paid each other {count(n, 'time')} before, most recently {dur(np.expm1(g('pair_gap')))} ago.")
        if k == 'history':
            return (f"Sender's account: {count(g('s_n_before'), 'operation')} over {dur(np.expm1(g('s_age')))}; "
                    f"receiver's: {g('r_n_before'):.0f} over {dur(np.expm1(g('r_age')))}.")
        if k == 'amount':
            return f"Amount {row['amount_uzs']:,.0f} UZS (≈ {row['amount_usd']:,.0f} USD)."
        if k == 'rail':
            return f"Channel: {row['rail']}."
        if k == 'time':
            return f"Time: {row['ts']:%H:%M}, {WEEKDAYS.get(row['ts'].isoweekday(), '')}."
        if k == 'risk':
            si, so, ri, ro = (g(f) for f in ('s_hi_in_7d', 's_hi_out_7d', 'r_hi_in_7d', 'r_hi_out_7d'))
            if si + so + ri + ro == 0:
                return 'Neither account sent or received a high-risk payment in the last 7 days.'
            return (f"High-risk payments in the last 7 days: the sender received {si:.0f} and sent {so:.0f}; the receiver "
                    f"received {ri:.0f} and sent {ro:.0f} (high-risk: first-stage score above {RISK_THRESHOLD:.0%}).")
        if k == 'structure':
            return 'Transfer to own account.' if g('self_loop') else ('Within one bank.' if g('same_bank') else 'Between different banks.')
        return (f"In 7 days the sender received {uzs(g('s_usd_in_7d'))} and sent {uzs(g('s_usd_out_7d'))}; "
                f"the receiver received {uzs(g('r_usd_in_7d'))}.")


def subgraph(t, scores, src, dst, ts, row_id, max_edges=36, window_days=7):
    """Transactions touching either account within +-window_days, riskiest first, for the evidence graph.
    Returns nodes with a simple layout (sender left, receiver right, counterparties around them) and edges."""
    lo, hi = ts - timedelta(days=window_days), ts + timedelta(days=window_days)
    near = t.with_columns(risk=pl.Series(scores)).filter(
        (pl.col('src').is_in([src, dst]) | pl.col('dst').is_in([src, dst])) & pl.col('ts').is_between(lo, hi))
    near = near.sort(pl.col('row') == row_id, 'risk', descending=True).head(max_edges)
    names = {src: 'Sender', dst: 'Receiver'}
    others = [a for a in dict.fromkeys(near['src'].to_list() + near['dst'].to_list()) if a not in (src, dst)]
    left = [a for a in others if near.filter((pl.col('src') == a) & (pl.col('dst') == src) | (pl.col('dst') == a) & (pl.col('src') == src)).height]
    right = [a for a in others if a not in left]
    pos = {src: (-0.35, 0.0), dst: (0.35, 0.0)}
    for side, accts, x0 in [(-1, left, -0.85), (1, right, 0.85)]:
        for i, a in enumerate(accts):
            y = (i - (len(accts) - 1) / 2) / max(1, len(accts) - 1) * 1.6 if len(accts) > 1 else 0.0
            pos[a] = (x0, y)
    nodes = [{'id': a, 'x': p[0], 'y': p[1], 'label': names.get(a, a.split(':')[-1][-6:]), 'main': a in (src, dst)} for a, p in pos.items()]
    edges = [{'src': r['src'], 'dst': r['dst'], 'risk': float(r['risk']), 'alert': r['row'] == row_id,
              'amount_uzs': float(r['amount_uzs']), 'ts': r['ts'].isoformat(timespec='minutes'), 'row': int(r['row'])}
             for r in near.iter_rows(named=True)]
    return {'nodes': nodes, 'edges': edges}
