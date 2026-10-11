"""Why an alert was raised: LightGBM TreeSHAP grouped into plain-English reasons, each with the facts
behind it, plus the evidence subgraph (the transactions around the two accounts)."""
from datetime import timedelta

import numpy as np
import polars as pl

from .features import RISK_THRESHOLD
from .i18n import Lang
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
WEEKDAYS = {'en': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'ru': ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']}
DURATION = {'en': ('no data', 'under a minute', '{} min', '{} hours', '{} days'),
            'ru': ('нет данных', 'меньше минуты', '{} мин', '{} ч', '{} дн')}


def group_of(c):
    return next(k for k, _, rule in GROUPS if rule(c))


def dur(seconds, L=Lang('en')):
    """Seconds -> short duration in the page's language."""
    d = DURATION[L.code]
    if seconds is None or not np.isfinite(seconds): return d[0]
    m = seconds / 60
    if m < 1: return d[1]
    if m < 120: return d[2].format(L.num(m))
    if m < 2880: return d[3].format(L.num(m / 60, 1))
    return d[4].format(L.num(m / 1440, 1))


def count(n, word):
    return f"{n:.0f} {word}{'' if n == 1 else 's'}"


class Explainer:
    def __init__(self, columns, X_ref):
        self.columns = list(columns)
        self.groups = np.array([group_of(c) for c in self.columns])

    def reasons(self, contrib, x, row, top=4, lang='en'):
        """contrib: TreeSHAP row (features + base value); x: feature dict; row: the transaction (dict).
        Titles stay in English (the page translates them); the evidence is written in `lang`."""
        sums = {k: float(contrib[:-1][self.groups == k].sum()) for k, _, _ in GROUPS if (self.groups == k).any()}
        ranked = sorted(sums, key=lambda k: -abs(sums[k]))[:top]
        return [{'group': k, 'title': TITLES[k], 'impact': round(sums[k], 3), 'up': sums[k] > 0,
                 'evidence': self.evidence(k, x, row, Lang(lang))} for k in ranked]

    def evidence(self, k, x, row, L=Lang('en')):
        g = lambda f: float(x.get(f, np.nan))
        ru = L.code == 'ru'
        uzs = lambda usd: L.uzs(usd * UZS_PER_USD)
        if k == 'fan_in':
            n, cp, cp7 = g('r_n_in_1d'), g('r_cp_in_1d'), g('r_cp_in_7d')
            if ru:
                return (f'Получателю за последние 24 часа пришло платежей: {n:.0f}, из них от счетов, которые раньше ему не платили: '
                        f'{cp:.0f} (таких новых плательщиков за 7 дней: {cp7:.0f}).')
            return (f"The receiver got {count(n, 'payment')} in the last 24 hours; {cp:.0f} of them came from accounts that had "
                    f"never paid it before ({cp7:.0f} such new payers in 7 days).")
        if k == 'fan_out':
            n, cp, cp7 = g('s_n_out_1d'), g('s_cp_out_1d'), g('s_cp_out_7d')
            if ru:
                return (f'Отправитель за последние 24 часа сделал платежей: {n:.0f}, из них на счета, которым раньше не платил: '
                        f'{cp:.0f} (таких новых получателей за 7 дней: {cp7:.0f}).')
            return (f"The sender made {count(n, 'payment')} in the last 24 hours; {cp:.0f} of them went to accounts it had "
                    f"never paid before ({cp7:.0f} such new payees in 7 days).")
        if k == 'pass':
            if not np.isfinite(g('s_prev_in')):
                return ('На счёт отправителя раньше не поступали деньги: это один из его первых исходящих платежей.' if ru else
                        "No money had come into the sender's account before: this is one of its first outgoing payments.")
            ago, cin, cout = dur(np.expm1(g('s_prev_in')), L), uzs(g('s_usd_in_1d')), uzs(g('s_usd_out_1d'))
            if ru:
                return f'Последнее поступление на счёт отправителя было {ago} назад; за 24 часа поступило {cin}, ушло {cout}.'
            return f"Money last came into the sender's account {ago} ago; in 24 hours {cin} came in and {cout} went out."
        if k == 'relation':
            n = g('pair_n_before')
            if n == 0:
                return ('Первый платёж от этого отправителя этому получателю (новый контрагент).' if ru else
                        'The first payment from this sender to this receiver (a new counterparty).')
            ago = dur(np.expm1(g('pair_gap')), L)
            if ru:
                return f'Отправитель уже платил этому получателю раньше (раз: {n:.0f}), последний раз {ago} назад.'
            return f"The sender has paid this receiver {count(n, 'time')} before, most recently {ago} ago."
        if k == 'history':
            sa, ra = dur(np.expm1(g('s_age')), L), dur(np.expm1(g('r_age')), L)
            if ru:
                return f"Счёт отправителя: операций {g('s_n_before'):.0f} за {sa}; счёт получателя: {g('r_n_before'):.0f} за {ra}."
            return f"Sender's account: {count(g('s_n_before'), 'operation')} over {sa}; receiver's: {g('r_n_before'):.0f} over {ra}."
        if k == 'amount':
            return f"{'Сумма' if ru else 'Amount'} {L.uzs(row['amount_uzs'])} (≈ {L.num(row['amount_usd'])} USD)."
        if k == 'rail':
            return f"{'Канал' if ru else 'Channel'}: {L.t(row['rail'])}."
        if k == 'time':
            return f"{'Время' if ru else 'Time'}: {row['ts']:%H:%M}, {WEEKDAYS[L.code][row['ts'].isoweekday() - 1]}."
        if k == 'risk':
            si, so, ri, ro = (g(f) for f in ('s_hi_in_7d', 's_hi_out_7d', 'r_hi_in_7d', 'r_hi_out_7d'))
            if si + so + ri + ro == 0:
                return ('Ни один из счетов не получал и не отправлял высокорисковых платежей за последние 7 дней.' if ru else
                        'Neither account sent or received a high-risk payment in the last 7 days.')
            if ru:
                return (f'Высокорисковые платежи за 7 дней: отправитель получил {si:.0f} и отправил {so:.0f}; получатель получил '
                        f'{ri:.0f} и отправил {ro:.0f} (высокий риск: оценка первой стадии выше {RISK_THRESHOLD:.0%}).')
            return (f"High-risk payments in the last 7 days: the sender received {si:.0f} and sent {so:.0f}; the receiver "
                    f"received {ri:.0f} and sent {ro:.0f} (high-risk: first-stage score above {RISK_THRESHOLD:.0%}).")
        if k == 'structure':
            if g('self_loop'):
                return 'Перевод на собственный счёт.' if ru else 'Transfer to own account.'
            if g('same_bank'):
                return 'Внутри одного банка.' if ru else 'Within one bank.'
            return 'Между разными банками.' if ru else 'Between different banks.'
        sin, sout, rin = uzs(g('s_usd_in_7d')), uzs(g('s_usd_out_7d')), uzs(g('r_usd_in_7d'))
        if ru:
            return f'За 7 дней отправитель получил {sin} и отправил {sout}; получатель получил {rin}.'
        return f"In 7 days the sender received {sin} and sent {sout}; the receiver received {rin}."


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
