"""Load transaction data into one internal schema.

Both the public training data (IBM AMLworld) and our Uzbek synthetic demo data are converted to
the same columns, so one feature pipeline and one set of models serve both:

    tx_id, ts, src, dst, src_bank, dst_bank, amount, currency, amount_usd, fmt, label

`src` / `dst` are account node ids ("bank:account"). `amount_usd` puts every currency on one
scale; `fmt` is the payment format mapped to a small set of rails shared by both datasets.
"""
import numpy as np
import polars as pl

AMLWORLD_COLUMNS = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received',
                    'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']

# Payment rails shared by both datasets. AMLworld formats keep their names; Uzbek rails map onto
# the closest AMLworld format (declared in the README as a modelling assumption).
FORMATS = ['ACH', 'Bitcoin', 'Cash', 'Cheque', 'Credit Card', 'Reinvestment', 'Wire']
UZ_RAILS = {'Humo': 'Credit Card', 'Uzcard': 'Credit Card', 'Click': 'ACH', 'Payme': 'ACH', 'Uzum': 'ACH',
            "Bank o'tkazmasi": 'ACH', 'Naqd': 'Cash', 'SWIFT': 'Wire', 'Kripto': 'Bitcoin'}
UZS_PER_USD = 12_800.0   # fixed demo rate (assumption): converts UZS amounts to the USD scale


def read_amlworld(path, n_rows=None):
    """AMLworld *_Trans.csv -> internal schema, sorted by time."""
    raw = pl.read_csv(path, n_rows=n_rows, schema_overrides={'From Bank': pl.Utf8, 'To Bank': pl.Utf8})
    raw.columns = AMLWORLD_COLUMNS
    t = raw.select(
        ts=pl.col('Timestamp').str.to_datetime('%Y/%m/%d %H:%M'),
        src_bank=pl.col('From Bank'), dst_bank=pl.col('To Bank'),
        src=pl.col('From Bank') + ':' + pl.col('Account'), dst=pl.col('To Bank') + ':' + pl.col('Account.1'),
        amount=pl.col('Amount Paid'), currency=pl.col('Payment Currency'),
        amount_received=pl.col('Amount Received'), recv_currency=pl.col('Receiving Currency'),
        fmt=pl.col('Payment Format'), label=pl.col('Is Laundering').cast(pl.Int8))
    rates = usd_rates(t)
    t = t.with_columns(amount_usd=pl.col('amount') * pl.col('currency').replace_strict(rates, return_dtype=pl.Float64))
    return t.sort('ts', maintain_order=True).with_row_index('tx_id')


def usd_rates(t):
    """Exchange rates to USD implied by the data itself: in cross-currency rows the generator records
    both the paid and the received amount, so their ratio is the rate it used."""
    pairs = t.filter(pl.col('currency') != pl.col('recv_currency')).with_columns(
        r=pl.col('amount_received') / pl.col('amount')).group_by('currency', 'recv_currency').agg(r=pl.col('r').median())
    to_usd = {'US Dollar': 1.0}
    for row in pairs.filter(pl.col('recv_currency') == 'US Dollar').iter_rows(named=True):
        to_usd[row['currency']] = row['r']
    for row in pairs.filter(pl.col('currency') == 'US Dollar').iter_rows(named=True):   # fill gaps from the other side
        to_usd.setdefault(row['recv_currency'], 1.0 / row['r'])
    missing = set(t['currency'].unique()) - set(to_usd)
    assert not missing, f'no exchange rate found for {missing}'
    return to_usd


UZ_COLUMNS = {'ts': pl.Datetime('us'), 'src_bank': pl.Utf8, 'src_account': pl.Utf8, 'dst_bank': pl.Utf8,
              'dst_account': pl.Utf8, 'amount_uzs': pl.Float64, 'rail': pl.Utf8}


def read_uz(source):
    """Uzbek demo CSV (a path or file-like object) with exactly the UZ_COLUMNS, account numbers kept as text."""
    df = pl.read_csv(source, schema_overrides={c: pl.Utf8 for c in UZ_COLUMNS})
    return df.with_columns(pl.col('ts').str.to_datetime(time_unit='us'), pl.col('amount_uzs').cast(pl.Float64))


def from_uz(df):
    """Uzbek demo transactions (generator output or a jury CSV) -> internal schema."""
    t = df.with_columns(
        src=pl.col('src_bank') + ':' + pl.col('src_account'), dst=pl.col('dst_bank') + ':' + pl.col('dst_account'),
        fmt=pl.col('rail').replace_strict(UZ_RAILS, return_dtype=pl.Utf8),
        amount=pl.col('amount_uzs'), currency=pl.lit('UZS'),
        amount_usd=pl.col('amount_uzs') / UZS_PER_USD)
    if 'label' not in t.columns:
        t = t.with_columns(label=pl.lit(None, dtype=pl.Int8))
    return t.sort('ts', maintain_order=True).with_row_index('tx_id')


def temporal_split(t, train=0.6, val=0.2):
    """Split by time, never at random: the model is evaluated on transactions after those it learnt from."""
    n = t.height
    a, b = int(n * train), int(n * (train + val))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)
