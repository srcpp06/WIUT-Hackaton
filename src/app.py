"""FlowLens web app: FastAPI + server-rendered Jinja pages (HTMX for partial updates, Chart.js for charts)."""
import io
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .generator import BANKS, PERSONAS, SCHEMES
from .i18n import LANGS, Lang
from .ingest import UZ_COLUMNS, UZ_RAILS
from .service import Service, scheme_rows

ROOT = Path(__file__).resolve().parent.parent
BRAND = "FlowLens"
VERSION = "0.6.0"
LEVELS = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
ACTIONS = {'confirm': 'Confirm as suspicious', 'escalate': 'Escalate to the financial monitoring unit',
           'dismiss': 'Dismiss (false alarm)'}
LOGO_SVG = ('<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs><linearGradient id="fl-g" '
            'x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#06b6d4"/></linearGradient>'
            '</defs><circle cx="14" cy="14" r="10" fill="none" stroke="url(#fl-g)" stroke-width="3"/><path d="M21.5 21.5 28 28" '
            'stroke="url(#fl-g)" stroke-width="3.4" stroke-linecap="round"/><path d="M8.5 16.5l3.2-3.8 2.8 2.4 4-4.6" fill="none" '
            'stroke="url(#fl-g)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
state = {}


@asynccontextmanager
async def lifespan(app):
    try:
        state['svc'] = Service()
    except FileNotFoundError as e:   # models or demo data missing: serve the pages with a clear message
        state['error'] = f"Model or demo data not found: {e.filename}"
    yield


app = FastAPI(title=BRAND, version=VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")
templates.env.filters.update(
    uzs=lambda v: f"{v:,.0f} UZS",
    pct=lambda v: f"{100 * v:.1f}%",
    dt=lambda v: v.strftime('%Y-%m-%d %H:%M') if hasattr(v, 'strftime') else str(v)[:16],
    level=lambda v: LEVELS.get(str(v), str(v)))
templates.env.globals.update(BRAND=BRAND, LOGO_SVG=LOGO_SVG, VERSION=VERSION, LEVELS=LEVELS, ACTIONS=ACTIONS, PERSONAS=PERSONAS)


def svc() -> Service:
    if 'svc' not in state:
        raise HTTPException(503, state.get('error', 'The service is starting'))
    return state['svc']


def lang_of(request):
    """?lang= switches the language (and is remembered in a cookie); otherwise the cookie; English by default."""
    for code in (request.query_params.get('lang'), request.cookies.get('lang')):
        if code in LANGS:
            return code
    return 'en'


def lang_url(request, code):
    query = '&'.join(f'{k}={v}' for k, v in request.query_params.items() if k != 'lang')
    return f"{request.url.path}?{query + '&' if query else ''}lang={code}"


def page(request, name, **ctx):
    L = Lang(lang_of(request))
    resp = templates.TemplateResponse(request, name, {**ctx, 'lang': L.code, '_': L.t, 'num': L.num, 'pct': L.pct, 'uzs': L.uzs,
                                                      'lang_url': lambda code: lang_url(request, code)})
    if request.query_params.get('lang') in LANGS:
        resp.set_cookie('lang', L.code, max_age=365 * 86400, samesite='lax')
    return resp


# ---------------------------------------------------------------- pages

@app.get("/health")
def health() -> dict:
    return {"status": "ok" if 'svc' in state else "degraded", "version": VERSION, "error": state.get('error')}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request, level: Literal['high', 'medium'] | None = None, offset: int = 0):
    s = svc()
    q = s.queue(level=level, limit=50, offset=max(0, offset))
    ctx = dict(q=q, level=level, offset=max(0, offset), dev=s.m.metrics.get('dev', False), now=s.now(),
               n_injected=len(s.injected))
    if request.headers.get('HX-Request'):
        return page(request, '_queue_table.html', **ctx)
    return page(request, 'queue.html', **ctx)


@app.get("/alert/{row}", response_class=HTMLResponse, include_in_schema=False)
def alert_page(request: Request, row: int):
    s = svc()
    if row not in s.pos:
        raise HTTPException(404, f"Transaction not found: {row}")
    return page(request, 'alert.html', a=s.alert(row, lang_of(request)), dev=s.m.metrics.get('dev', False))


@app.post("/alert/{row}/decision", response_class=HTMLResponse, include_in_schema=False)
def decide(request: Request, row: int, action: Literal['confirm', 'escalate', 'dismiss'] = Form(...),
           reason: str = Form(..., min_length=3, max_length=500), officer: str = Form(..., min_length=2, max_length=60)):
    s = svc()
    if row not in s.pos:
        raise HTTPException(404, f"Transaction not found: {row}")
    a = s.alert(row)   # log exactly what the officer saw
    e = s.audit.append(alert_id=row, action=action, reason=reason.strip(), officer=officer.strip(),
                       prob=round(a['prob'], 4), p_lgbm=round(a['p_lgbm'], 4), p_gnn=round(a['p_gnn'], 4),
                       level=str(a['level']), top_reasons=[r['title'] for r in a['reasons'][:3]], model_version=VERSION)
    return page(request, '_decision.html', d=e)


@app.get("/account/{node}", response_class=HTMLResponse, include_in_schema=False)
def account_page(request: Request, node: str):
    return page(request, 'account.html', acc=svc().account(node))


@app.get("/inject", response_class=HTMLResponse, include_in_schema=False)
def inject_page(request: Request):
    s = svc()
    personas, risky = s.personas()
    return page(request, 'inject.html', personas=personas, risky=risky, banks=BANKS, rails=list(UZ_RAILS),
                schemes=SCHEMES, now=s.now(), n_injected=len(s.injected))


@app.post("/inject/tx", response_class=HTMLResponse, include_in_schema=False)
def inject_tx(request: Request, src: str = Form(...), dst_bank: str = Form(...), dst_account: str = Form(...),
              amount: float = Form(..., ge=1_000, le=10_000_000_000), rail: str = Form(...),
              minutes: int = Form(1, ge=1, le=1440)):
    s = svc()
    if rail not in UZ_RAILS or dst_bank not in BANKS or ':' not in src:
        raise HTTPException(422, "Unknown channel, bank or sender")
    acct = ''.join(ch for ch in dst_account if ch.isdigit())
    if not 8 <= len(acct) <= 20:
        raise HTTPException(422, "The receiver's account number must have 8 to 20 digits")
    sb, sa = src.split(':', 1)
    rows = pl.DataFrame([(s.now() + timedelta(minutes=minutes), sb, sa, dst_bank, acct, round(amount, 2), rail)],
                        schema=list(UZ_COLUMNS), orient='row').with_columns(pl.col('ts').cast(pl.Datetime('us')))
    res = s.inject(rows)
    return page(request, '_injected.html', res=res, items=[s.alert(r, lang_of(request)) for r in res['rows']])


@app.post("/inject/scheme", response_class=HTMLResponse, include_in_schema=False)
def inject_scheme(request: Request, target: str = Form(...), kind: Literal['fan_in', 'fan_out', 'cycle'] = Form(...),
                  n: int = Form(10, ge=3, le=30)):
    s = svc()
    if target not in set(s.accounts['node'].to_list()):
        raise HTTPException(422, "Account not found")
    res = s.inject(scheme_rows(kind, target, s.now(), s.accounts, np.random.default_rng(), n=n))
    return page(request, '_injected.html', res=res, items=[s.alert(r, lang_of(request)) for r in res['rows']])


@app.post("/inject/reset", include_in_schema=False)
def reset():
    svc().reset()
    return RedirectResponse('/inject', status_code=303)


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page(request: Request):
    return page(request, 'upload.html', result=None, error=None)


@app.post("/upload", response_class=HTMLResponse, include_in_schema=False)
async def upload(request: Request, file: UploadFile = File(...)):
    s = svc()
    L = Lang(lang_of(request))
    data = await file.read(20_000_001)
    try:
        if len(data) > 20_000_000:
            raise ValueError(L.t("The file is larger than 20 MB."))
        result = s.score_frame(parse_csv(data, L), L.code)
        return page(request, 'upload.html', result=result, error=None, filename=file.filename)
    except ValueError as e:
        return page(request, 'upload.html', result=None, error=str(e))


@app.get("/sample.csv", include_in_schema=False)
def sample_csv():
    """The last two days of the demo bank (planted scheme rows included, labels not) to try the upload."""
    s = svc()
    df = s.base.filter(pl.col('ts') >= s.base['ts'].max() - timedelta(days=2)).drop('row')
    buf = io.StringIO(); df.write_csv(buf)
    return PlainTextResponse(buf.getvalue(), media_type='text/csv',
                             headers={'Content-Disposition': 'attachment; filename="flowlens_sample.csv"'})


@app.get("/evidence", response_class=HTMLResponse, include_in_schema=False)
def evidence_page(request: Request):
    s = svc()
    return page(request, 'evidence.html', m=s.m.metrics, live=s.live_latency(), dev=s.m.metrics.get('dev', False),
                elliptic=load_json(ROOT / 'models' / 'elliptic.json'), uzcheck=load_json(ROOT / 'models' / 'uz_check.json'))


@app.get("/audit", response_class=HTMLResponse, include_in_schema=False)
def audit_page(request: Request):
    s = svc()
    ok, n, broken = s.audit.verify()
    return page(request, 'audit.html', ok=ok, n=n, broken=broken, entries=s.audit.entries()[-200:][::-1])


# ---------------------------------------------------------------- JSON API

class Tx(BaseModel):
    ts: datetime
    src_bank: str = Field(min_length=1, max_length=40)
    src_account: str = Field(min_length=1, max_length=40)
    dst_bank: str = Field(min_length=1, max_length=40)
    dst_account: str = Field(min_length=1, max_length=40)
    amount_uzs: float = Field(gt=0, le=1e13, allow_inf_nan=False)
    rail: Literal['Humo', 'Uzcard', 'Click', 'Payme', 'Uzum', 'Bank transfer', 'Cash', 'SWIFT', 'Crypto']


class Batch(BaseModel):
    transactions: list[Tx] = Field(min_length=1, max_length=50_000)


@app.post("/api/score")
def api_score(body: Batch):
    """Score a batch of transactions as its own graph; returns probabilities and the top reasons."""
    df = pl.DataFrame([t.model_dump() for t in body.transactions]).with_columns(pl.col('ts').cast(pl.Datetime('us')))
    return svc().score_frame(df.select(list(UZ_COLUMNS)))


@app.get("/api/metrics")
def api_metrics():
    s = svc()
    return {**s.m.metrics, 'live_latency': s.live_latency()}


# ---------------------------------------------------------------- helpers

def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def parse_csv(data: bytes, L=Lang('en')) -> pl.DataFrame:
    """CSV with the UZ columns -> validated frame, or ValueError with a message for the user (in L's language)."""
    _ = L.t
    try:
        df = pl.read_csv(io.BytesIO(data), infer_schema=False)
    except Exception:
        raise ValueError(_("Could not read the CSV: a comma-separated file with a header row is expected."))
    missing = [c for c in UZ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(_("Missing columns: {cols}. Required columns: {req}.").format(cols=', '.join(missing), req=', '.join(UZ_COLUMNS)))
    if not 10 <= df.height <= 50_000:
        raise ValueError(_("The file must have 10 to 50,000 rows (it has {n}).").format(n=L.num(df.height)))
    c = pl.col
    df = df.select(list(UZ_COLUMNS)).with_columns(
        c('ts').str.strip_chars().str.to_datetime(strict=False, time_unit='us'),
        c('amount_uzs').str.strip_chars().str.replace_all(' ', '').cast(pl.Float64, strict=False),
        *[c(k).str.strip_chars() for k in ('src_bank', 'src_account', 'dst_bank', 'dst_account', 'rail')])
    checks = [(c('ts').is_null(), _("invalid date/time (example: 2026-09-30 14:05:00)")),
              (c('amount_uzs').is_null() | (c('amount_uzs') <= 0), _("amount must be a positive number")),
              (~c('rail').is_in(list(UZ_RAILS)), _("rail must be one of: {r}").format(r=', '.join(UZ_RAILS))),
              (c('src_account').is_null() | c('dst_account').is_null() | c('src_bank').is_null() | c('dst_bank').is_null(),
               _("bank and account columns must not be empty"))]
    errors = []
    for cond, msg in checks:
        bad = df.with_row_index('i').filter(cond.fill_null(True))['i']
        if len(bad):
            rows = ', '.join(str(i + 2) for i in bad[:5].to_list()) + (' …' if len(bad) > 5 else '')
            errors.append(_("{msg} (rows: {rows})").format(msg=msg, rows=rows))
    if errors:
        raise ValueError('; '.join(errors))
    return df
