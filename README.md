# QALQON

Real-time detection of fraud and money laundering in bank transactions, built for the WIUT FinTech
hackathon final (partner: MKBANK anti-fraud / AML desk).

QALQON scores each transaction with trained models (a LightGBM baseline and a graph neural
network), explains every alert and leaves the decision to a human officer. No LLM and no external
API sit on the decision path; everything runs on a plain CPU server.

> Work in progress: this README grows with the system. Every metric reported here will be
> measured on held-out public data, never invented.

## Run locally

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/uvicorn src.app:app --reload
```

## Deploy

Live demo: **https://qalqon.132-226-186-157.sslip.io**

`./deploy.sh` checks out the pushed commit on the server, rebuilds the `qalqon` container
(`docker compose up -d --build`), waits for `/health` and records the result as a GitHub
deployment. `./deploy.sh --data` also copies the trained `models/`.
