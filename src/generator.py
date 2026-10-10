"""Uzbek-flavoured synthetic transactions for the live demo (released CC0 with this script).

Everyday activity on local rails (Humo / Uzcard cards, Click / Payme / Uzum wallets, bank transfers,
cash) for a population of people and small businesses, plus planted money-laundering and fraud
schemes taken from FATF typologies and the AMLworld pattern list: mule fan-in with cash-out,
fan-out, cycles, scatter-gather and gather-scatter. Ground-truth labels and the scheme each planted
transaction belongs to are written to a separate file, so the app never reads them when scoring.

    python -m src.generator --out data/demo --days 30 --accounts 3000 --seed 7
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

BANKS = ['MKBANK', 'Bank-A', 'Bank-B', 'Bank-C', 'Bank-D', 'Bank-E']
BANK_P = [0.35, 0.2, 0.15, 0.12, 0.1, 0.08]
FIRST = ['Aziz', 'Dilnoza', 'Jasur', 'Malika', 'Sardor', 'Nodira', 'Bekzod', 'Gulnora', 'Otabek', 'Shahnoza',
         'Rustam', 'Madina', 'Javlon', 'Zarina', 'Ulugbek', 'Kamola', 'Sherzod', 'Feruza', 'Doniyor', 'Laylo']
LAST = ['Karimov', 'Tursunova', 'Rahimov', 'Yusupova', 'Aliyev', 'Saidova', 'Ergashev', 'Nazarova',
        'Qodirov', 'Ismoilova', 'Xolmatov', 'Abdullayeva', 'Mirzayev', 'Sultonova']
CITIES = ['Toshkent', 'Samarqand', 'Buxoro', 'Andijon', "Farg'ona", 'Namangan', 'Qarshi', 'Nukus']
PERSONAS = {'talaba': "Talaba", 'xodim': 'Oylik oluvchi xodim', 'pensioner': 'Pensioner',
            'biznes': 'Kichik biznes', 'freelancer': 'IT frilanser'}
SCHEMES = {'mule_fan_in': "Mule hisob: ko'p jo'natuvchidan yig'ib, naqdlashtirish",
           'fan_out': "Bir hisobdan ko'p hisoblarga tarqatish",
           'cycle': "Aylanma o'tkazmalar (pul boshlang'ich hisobga qaytadi)",
           'scatter_gather': "Tarqatib, qayta yig'ish (scatter-gather)",
           'gather_scatter': "Yig'ib, qayta tarqatish (gather-scatter)"}


class World:
    def __init__(self, n_accounts, days, seed):
        self.rng = np.random.default_rng(seed)
        self.start = datetime(2026, 9, 1)
        self.days = days
        r = self.rng
        kinds = r.choice(list(PERSONAS), n_accounts, p=[0.15, 0.45, 0.12, 0.18, 0.10])
        self.accounts = pl.DataFrame({
            'account': [f'2020{r.integers(10**15, 10**16)}' for _ in range(n_accounts)],
            'bank': r.choice(BANKS, n_accounts, p=BANK_P),
            'owner': [f'{r.choice(FIRST)} {r.choice(LAST)}' for _ in range(n_accounts)],
            'city': r.choice(CITIES, n_accounts),
            'persona': kinds})
        # a few hundred merchants / utilities receive card and wallet payments
        n_m = max(50, n_accounts // 15)
        self.merchants = [(f'2021{r.integers(10**15, 10**16)}', r.choice(BANKS, p=BANK_P)) for _ in range(n_m)]
        self.rows, self.truth = [], []

    def sink(self, kind, bank):
        """A counterparty outside the bank's customer base: an ATM / cash desk or a foreign payer."""
        return (f'{kind}-{self.rng.integers(10**8, 10**9)}', bank)

    def when(self, day, hour_mu=14, hour_sd=4):
        h = float(np.clip(self.rng.normal(hour_mu, hour_sd), 0, 23.98))
        return self.start + timedelta(days=int(day), hours=h, seconds=int(self.rng.integers(0, 60)))

    def add(self, ts, src, dst, amount, rail, scheme=None, attempt=None):
        self.rows.append((ts, src[1], src[0], dst[1], dst[0], round(float(amount), -2), rail))
        self.truth.append((int(scheme is not None), scheme or '', attempt if attempt is not None else -1))

    # ------------------------------------------------------------ everyday behaviour
    def everyday(self):
        r = self.rng
        acc = list(zip(self.accounts['account'], self.accounts['bank']))
        employers = [acc[i] for i in r.choice(len(acc), 40, replace=False)]
        for (a, b), kind in zip(acc, self.accounts['persona']):
            me = (a, b)
            friends = [acc[i] for i in r.choice(len(acc), 6, replace=False)]
            income = {'talaba': 1.5e6, 'xodim': 7e6, 'pensioner': 2.2e6, 'biznes': 25e6, 'freelancer': 14e6}[kind]
            if kind in ('xodim', 'pensioner'):   # salary or pension once a month
                src = employers[hash(a) % len(employers)]
                self.add(self.when(r.integers(0, 3), 11, 1), src, me, income * r.uniform(0.9, 1.1), "Bank o'tkazmasi")
            for _ in range(r.poisson(self.days * {'biznes': 3.0}.get(kind, 0.15))):   # customers paying a business
                self.add(self.when(r.integers(0, self.days)), friends[r.integers(0, 6)], me, r.lognormal(np.log(4e5), 0.8),
                         r.choice(['Click', 'Payme', 'Uzum', 'Humo', 'Uzcard']))
            if kind == 'freelancer':   # foreign clients pay a few times a month
                for _ in range(r.poisson(3)):
                    self.add(self.when(r.integers(0, self.days)), self.sink('SWIFT', 'FOREIGN'), me, r.lognormal(np.log(5e6), 0.5), 'SWIFT')
            for _ in range(r.poisson(self.days * 1.2)):   # card purchases at merchants
                self.add(self.when(r.integers(0, self.days)), me, self.merchants[r.integers(0, len(self.merchants))],
                         r.lognormal(np.log(income / 60), 0.9), r.choice(['Humo', 'Uzcard'], p=[0.55, 0.45]))
            for _ in range(r.poisson(self.days * 0.15)):   # transfers to family and friends
                self.add(self.when(r.integers(0, self.days)), me, friends[r.integers(0, 6)], r.lognormal(np.log(income / 12), 0.7),
                         r.choice(['Click', 'Payme', 'Uzum']))
            if r.random() < 0.5:   # occasional cash withdrawal
                self.add(self.when(r.integers(0, self.days)), me, self.sink('ATM', b), r.lognormal(np.log(income / 5), 0.5), 'Naqd')

    # ------------------------------------------------------------ planted schemes
    def plant(self, n_attempts):
        r = self.rng
        acc = list(zip(self.accounts['account'], self.accounts['bank']))
        kinds = r.choice(list(SCHEMES), n_attempts)
        for k, scheme in enumerate(kinds):
            day = int(r.integers(2, self.days - 2))
            t0 = self.when(day, hour_mu=r.choice([3, 13, 22]), hour_sd=2)
            pick = lambda n: [acc[i] for i in r.choice(len(acc), n, replace=False)]
            step = lambda m: timedelta(minutes=float(r.uniform(1, m)))
            if scheme == 'mule_fan_in':   # victims or smurfs pay one mule, who withdraws cash or forwards quickly
                mule, *senders = pick(int(r.integers(8, 20)))
                t, total = t0, 0.0
                for s in senders:
                    t += step(25); amt = r.lognormal(np.log(1.8e6), 0.4); total += amt
                    self.add(t, s, mule, amt, r.choice(['Click', 'Payme', 'Humo', 'Uzcard']), scheme, k)
                t += step(40)
                cash = r.random() < 0.6
                self.add(t, mule, self.sink('ATM', mule[1]) if cash else pick(1)[0], total * r.uniform(0.85, 0.97),
                         'Naqd' if cash else "Bank o'tkazmasi", scheme, k)
            elif scheme == 'fan_out':
                src, *dsts = pick(int(r.integers(8, 16)))
                t, base = t0, r.lognormal(np.log(3e6), 0.3)
                for d in dsts:
                    t += step(15); self.add(t, src, d, base * r.uniform(0.9, 1.1), r.choice(['Click', 'Payme', "Bank o'tkazmasi"]), scheme, k)
            elif scheme == 'cycle':
                ring = pick(int(r.integers(3, 7)))
                t, amt = t0, r.lognormal(np.log(4e7), 0.4)
                for a, b in zip(ring, ring[1:] + ring[:1]):
                    t += step(180); amt *= r.uniform(0.95, 0.995)
                    self.add(t, a, b, amt, "Bank o'tkazmasi", scheme, k)
            elif scheme == 'scatter_gather':
                src, dst, *mid = pick(int(r.integers(5, 10)))
                t, amt = t0, r.lognormal(np.log(2.5e7), 0.4)
                for m in mid:
                    self.add(t + step(30), src, m, amt / len(mid) * r.uniform(0.95, 1.05), "Bank o'tkazmasi", scheme, k)
                for m in mid:
                    self.add(t + timedelta(hours=float(r.uniform(1, 6))), m, dst, amt / len(mid) * r.uniform(0.9, 0.98), "Bank o'tkazmasi", scheme, k)
            else:   # gather_scatter
                hub, *rest = pick(int(r.integers(8, 14)))
                ins, outs = rest[:len(rest) // 2], rest[len(rest) // 2:]
                t, total = t0, 0.0
                for s in ins:
                    t += step(30); amt = r.lognormal(np.log(5e6), 0.4); total += amt
                    self.add(t, s, hub, amt, r.choice(['Click', 'Payme', "Bank o'tkazmasi"]), scheme, k)
                for d in outs:
                    t += step(20); self.add(t, hub, d, total / len(outs) * r.uniform(0.9, 0.98), "Bank o'tkazmasi", scheme, k)

    def frames(self):
        tx = pl.DataFrame(self.rows, schema=['ts', 'src_bank', 'src_account', 'dst_bank', 'dst_account', 'amount_uzs', 'rail'],
                          orient='row')
        truth = pl.DataFrame(self.truth, schema=['label', 'scheme', 'attempt'], orient='row')
        both = pl.concat([tx, truth], how='horizontal').sort('ts', maintain_order=True).with_row_index('row')
        return both.drop('label', 'scheme', 'attempt'), both.select('row', 'label', 'scheme', 'attempt')


def generate(n_accounts=3000, days=30, n_attempts=60, seed=7):
    w = World(n_accounts, days, seed)
    w.everyday()
    w.plant(n_attempts)
    tx, truth = w.frames()
    return tx, truth, w.accounts


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/demo')
    ap.add_argument('--accounts', type=int, default=3000)
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--attempts', type=int, default=60)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    tx, truth, accounts = generate(a.accounts, a.days, a.attempts, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tx.write_csv(out / 'transactions.csv')
    truth.write_csv(out / 'labels.csv')   # ground truth, kept apart from what the app scores
    accounts.write_csv(out / 'accounts.csv')
    print(f'{tx.height} transactions, {int(truth["label"].sum())} planted in {a.attempts} schemes, {accounts.height} accounts -> {out}')
