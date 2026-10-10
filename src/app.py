"""QALQON: real-time fraud and money-laundering scoring service (FastAPI)."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.1.0"

app = FastAPI(title="QALQON", version=VERSION)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"version": VERSION})
