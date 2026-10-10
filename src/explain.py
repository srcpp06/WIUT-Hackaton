"""Why an alert was raised: LightGBM TreeSHAP grouped into plain-Uzbek reasons, each with the facts
behind it, plus the evidence subgraph (the transactions around the two accounts)."""
from datetime import timedelta

import numpy as np
import polars as pl

from .ingest import UZS_PER_USD

# (key, Uzbek title, rule on the feature name); the first matching rule wins
GROUPS = [
    ('fan_in', "Qabul qiluvchiga ko'p turli hisobdan pul kelmoqda (yig'ish)", lambda c: c.startswith(('r_cp_in', 'r_n_in'))),
    ('fan_out', "Jo'natuvchi pulni ko'p turli hisobga tarqatmoqda", lambda c: c.startswith(('s_cp_out', 's_n_out'))),
    ('pass', "Pul hisobdan tez o'tib ketmoqda (tranzit)",
     lambda c: c in ('s_prev_in', 's_pass_1d', 'amt_vs_s_in_1d', 'r_prev_in') or c.startswith(('s_usd_in', 'r_usd_out'))),
    ('relation', "Jo'natuvchi va qabul qiluvchi o'rtasidagi aloqa tarixi", lambda c: c.startswith('pair_')),
    ('history', "Hisoblarning yoshi va faolligi", lambda c: c.endswith(('_age', '_n_before', '_prev_any'))),
    ('amount', "To'lov summasi", lambda c: c == 'log_usd'),
    ('rail', "To'lov kanali", lambda c: c == 'fmt'),
    ('time', "To'lov vaqti", lambda c: c in ('hour', 'weekday')),
    ('structure', "Bank ichidagi yoki o'z hisobiga o'tkazma", lambda c: c in ('same_bank', 'self_loop')),
    ('volume', "Hisoblarning oxirgi kunlardagi aylanmasi", lambda c: True),
]
TITLES = {k: t for k, t, _ in GROUPS}
WEEKDAYS = {1: 'dushanba', 2: 'seshanba', 3: 'chorshanba', 4: 'payshanba', 5: 'juma', 6: 'shanba', 7: 'yakshanba'}


def group_of(c):
    return next(k for k, _, rule in GROUPS if rule(c))


def dur(seconds):
    """Seconds -> short Uzbek duration."""
    if seconds is None or not np.isfinite(seconds): return "ma'lumot yo'q"
    m = seconds / 60
    if m < 1: return 'bir daqiqadan kam'
    if m < 120: return f'{m:.0f} daqiqa'
    if m < 2880: return f'{m / 60:.1f} soat'
    return f'{m / 1440:.1f} kun'


def som(usd):
    return f"{usd * UZS_PER_USD:,.0f} so'm".replace(',', ' ')


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
            return (f"Qabul qiluvchi oxirgi 24 soatda {g('r_cp_in_1d'):.0f} ta, 7 kunda {g('r_cp_in_7d'):.0f} ta turli hisobdan "
                    f"pul olgan (odatiy hisobda 7 kunda {med('r_cp_in_7d'):.0f} ta).")
        if k == 'fan_out':
            return (f"Jo'natuvchi oxirgi 24 soatda {g('s_cp_out_1d'):.0f} ta, 7 kunda {g('s_cp_out_7d'):.0f} ta turli hisobga "
                    f"pul yuborgan (odatiy hisobda 7 kunda {med('s_cp_out_7d'):.0f} ta).")
        if k == 'pass':
            if not np.isfinite(g('s_prev_in')):
                return "Jo'natuvchiga bundan oldin pul kelmagan: bu to'lov hisobning birinchi chiqimlaridan."
            return (f"Jo'natuvchiga oxirgi pul {dur(np.expm1(g('s_prev_in')))} oldin kelgan; 24 soatda "
                    f"{som(g('s_usd_in_1d'))} kirib, {som(g('s_usd_out_1d'))} chiqib ketgan.")
        if k == 'relation':
            n = g('pair_n_before')
            return ("Bu ikki hisob o'rtasidagi birinchi to'lov (yangi qarshi tomon)." if n == 0
                    else f"Bu ikki hisob o'rtasida avval {n:.0f} marta to'lov bo'lgan, oxirgisi {dur(np.expm1(g('pair_gap')))} oldin.")
        if k == 'history':
            return (f"Jo'natuvchi hisobi {dur(np.expm1(g('s_age')))} davomida {g('s_n_before'):.0f} ta operatsiya qilgan; "
                    f"qabul qiluvchi {dur(np.expm1(g('r_age')))} davomida {g('r_n_before'):.0f} ta.")
        if k == 'amount':
            return f"Summa {row['amount_uzs']:,.0f} so'm (≈ {row['amount_usd']:,.0f} USD).".replace(',', ' ')
        if k == 'rail':
            return f"Kanal: {row['rail']}."
        if k == 'time':
            return f"Vaqt: {row['ts']:%H:%M}, {WEEKDAYS.get(row['ts'].isoweekday(), '')}."
        if k == 'structure':
            return "O'z hisobiga o'tkazma." if g('self_loop') else ("Bitta bank ichida." if g('same_bank') else "Turli banklar o'rtasida.")
        return (f"Jo'natuvchi 7 kunda {som(g('s_usd_in_7d'))} qabul qilib, {som(g('s_usd_out_7d'))} yuborgan; "
                f"qabul qiluvchi {som(g('r_usd_in_7d'))} qabul qilgan.")


def subgraph(t, scores, src, dst, ts, row_id, max_edges=36, window_days=7):
    """Transactions touching either account within +-window_days, riskiest first, for the evidence graph.
    Returns nodes with a simple layout (sender left, receiver right, counterparties around them) and edges."""
    lo, hi = ts - timedelta(days=window_days), ts + timedelta(days=window_days)
    near = t.with_columns(risk=pl.Series(scores)).filter(
        (pl.col('src').is_in([src, dst]) | pl.col('dst').is_in([src, dst])) & pl.col('ts').is_between(lo, hi))
    near = near.sort(pl.col('row') == row_id, 'risk', descending=True).head(max_edges)
    names = {src: 'Jo\'natuvchi', dst: 'Qabul qiluvchi'}
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
