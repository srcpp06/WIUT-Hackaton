# FlowLens

Real-time detection of fraud and money laundering in bank transactions, built for the WIUT FinTech
hackathon final (partner: MKBANK anti-fraud / AML desk).

**Live demo: https://flowlens.132-226-186-157.sslip.io**

FlowLens scores every transaction with two trained models, a LightGBM baseline and a graph neural
network (GraphSAGE), explains each alert in plain English and leaves the decision to a human officer.
No LLM and no external API sit on the decision path; everything runs on one CPU server.

## What you can do in the demo

| Page | What it shows |
|---|---|
| **Queue** (`/`) | A synthetic Uzbek bank (2,000 customers, about 125,000 transactions on Humo, Uzcard, Click, Payme, Uzum, bank transfers, cash and SWIFT), every transaction scored, the riskiest first. |
| **Alert** (`/alert/{id}`) | The probability from each model, the reasons (TreeSHAP grouped into plain-English reasons with the facts behind them), the evidence subgraph around the two accounts, and the officer's decision form. |
| **Live input** (`/inject`) | Add one transaction or a whole scheme (mule fan-in with cash-out, fan-out, cycle) to the bank. Features are computed on the spot, both models score it and the queue updates. The models are not told what was added. |
| **CSV** (`/upload`) | Upload your own transactions (`ts, src_bank, src_account, dst_bank, dst_account, amount_uzs, rail`); they are scored as a separate graph. A sample file is linked on the page. |
| **Evidence** (`/evidence`) | Measured results: PR-AUC, precision and recall at alert budgets, calibration, the ablation LightGBM vs GNN vs ensemble, the Elliptic check, live latency and CPU cost per transaction. |
| **Audit** (`/audit`) | Every officer decision, hash-chained (SHA-256) so that editing a past entry is detected. |

## How it works

```
transactions ─▶ ingest: one schema for AMLworld and the Uzbek data (amounts in USD, rails mapped)
             ─▶ features that only look backwards in time (src/features.py)
                  · the payment: amount, rail, hour, weekday, same bank
                  · sender and receiver history before this payment: payments in/out and money in/out
                    over 24 h and 7 days, number of new counterparties (fan-in / fan-out)
                  · pass-through: how soon money leaves after it arrived; first payment between the pair
             ─▶ LightGBM in two stages (src/baseline.py): stage 2 adds each account's earlier risk,
                the stage-1 scores of the payments it recently sent and received (layering)
             ─▶ LightGBM stage 2                      ─┐
             ─▶ GraphSAGE over the account graph      ├─▶ stacking (logistic regression fitted on
                (src/gnn.py): accounts are nodes,     │   the validation period) ─▶ probability
                payments are edges, messages flow    ─┘
                both ways (money in and money out)
             ─▶ explanation: TreeSHAP per feature, grouped into readable reasons; evidence subgraph
             ─▶ ranked queue ─▶ officer decides ─▶ hash-chained audit log
```

- **Time-aware evaluation.** Data is split by time (first 60% train, next 20% validation, last 20%
  test). Early stopping, calibration, the stacking weights and the decision threshold use the
  validation period only; the test period is touched once, for the reported numbers.
- **GNN protocol.** The graph that scores a period contains the transactions up to the end of that
  period with labels hidden, as in the AMLworld benchmark (Egressy et al., 2024). The LightGBM
  features are stricter: they only use the past of each transaction.
- **Two stages without leakage.** Stage 2's risk features need a stage-1 score for every earlier
  payment. Training payments get out-of-fold scores (stage 1 refitted without each of three time
  blocks), so stage 2 learns from scores like the ones it sees on new data.
- **Calibration.** Isotonic regression on the validation period turns scores into probabilities.

## Results

All numbers are measured on data the models never saw during training; the Evidence page shows
them live from `models/metrics.json`.

**IBM AMLworld HI-Small, test period** (1,015,669 transactions from 2022-09-08 16:12 to 2022-09-18 16:18, laundering share 0.18%):

| Model | PR-AUC | ROC-AUC | F1 | Precision / recall at F1 threshold | Precision / recall in the top 1% |
|---|---|---|---|---|---|
| LightGBM, two stages | 0.725 | 0.989 | 0.764 | 88.3% / 67.3% | 14.6% / 82.7% |
| GraphSAGE alone | 0.218 | 0.951 | 0.215 | 69.6% / 12.7% | 8.3% / 46.8% |
| Stacking ensemble | **0.734** | **0.990** | **0.762** | **90.8%** / 65.6% | 14.7% / 83.1% |

In the riskiest 0.1% of test transactions (1,016 alerts) the ensemble is right 969 times: **precision 95.4%**, recall 53.9%.
Reviewing the riskiest 1% finds 83.1% of all laundering.

The second stage was chosen on the validation period (PR-AUC 0.620 → 0.649; other changes we tried,
more negatives and larger trees, moved it by less than 0.002, inside run-to-run noise) and then
evaluated on the test period once. The one-stage LightGBM had test PR-AUC 0.658 and the ensemble 0.679.

Stacking weights (fitted on validation): LightGBM 0.75, GNN 0.01. The GNN alone is much weaker than LightGBM here and adds about +0.01 PR-AUC to the ensemble.

**Live, on the demo server (2 ARM cores):** a new transaction is scored in 274 ms (median, p95 378 ms, 26 live requests) including the graph refresh; CPU cost about $2.10 per million transactions (Oracle A1 list price). Training on 5.1M transactions takes about 38 minutes on a laptop CPU (LightGBM in two stages with three out-of-fold refits, 22 min; GNN, 16 min).

**Transfer check on the Uzbek demo bank** (synthetic, our own generator, so not a headline metric): LightGBM PR-AUC 0.195 (recall in top 2%: 53%), GNN PR-AUC 0.005 (recall in top 2%: 4%), ensemble PR-AUC 0.199 (recall in top 2%: 55%). Fan-out schemes injected live are usually flagged, often above 50%, but not for every account profile; the mule fan-in pattern of our generator is scored low. This is why a pilot retrains on the bank's own data.

**Elliptic (external check of the approach, real Bitcoin transactions, test time steps 35–49):**

| Model | PR-AUC | Illicit F1 | Precision | Recall | PR-AUC, steps 35–42 | PR-AUC, steps 43–49 |
|---|---|---|---|---|---|---|
| LightGBM | 0.765 | 0.784 | 84.5% | 73.2% | 0.898 | 0.028 |
| GraphSAGE alone | 0.257 | 0.388 | 37.9% | 39.8% | 0.328 | 0.034 |
| Stacking ensemble | **0.794** | **0.803** | 91.5% | 71.6% | **0.919** | 0.040 |

A large dark market was shut down at time step 43 (Weber et al., 2019). After it the illicit
transactions look unlike anything in the training steps, and every model fails on them. Before the
shutdown the ensemble reaches PR-AUC 0.919 (F1 0.895, precision 95.2%); after it, 0.040. A deployed
model therefore needs regular retraining and an officer in the loop. The GNN alone generalises
poorly over time on Elliptic, but stacked with LightGBM it improves on LightGBM alone.

## Data and licences

| Dataset | Used for | Source and licence |
|---|---|---|
| IBM AMLworld, HI-Small (Altman et al., 2023) | training and held-out test | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, **CDLA-Sharing-1.0**. Our copy came from the Hugging Face mirror `bbfizp/AMLSim-HI-Small`; it has the published counts (5,078,345 transactions, 5,177 laundering), SHA-256 `b19d39f5…`. Not redistributed here. |
| Elliptic Data Set (Weber et al., 2019) | external check only | Kaggle `ellipticco/elliptic-data-set`, **CC BY-NC-ND 4.0** (non-commercial). Downloaded through PyTorch Geometric and used only to evaluate the approach. It is not part of the product model and is not redistributed. |
| FlowLens Uzbek demo bank | live demo only | generated by `src/generator.py` in this repo; **CC0 1.0** |

All data is synthetic or public. There is no real personal data. In the demo bank, "MKBANK" is the
partner bank and "Bank-A" to "Bank-E" are fictitious; names of people are randomly generated.

**Modelling assumptions (declared):**
- Uzbek amounts are converted at a fixed 12,800 UZS per USD.
- Uzbek rails are mapped to the nearest AMLworld payment format:

  | Uzbek rails | AMLworld format |
  |---|---|
  | Humo, Uzcard | Credit Card |
  | Click, Payme, Uzum, bank transfer | ACH |
  | cash | Cash |
  | SWIFT | Wire |
  | crypto | Bitcoin |

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

python -m src.generator --out data/demo --accounts 2000   # the Uzbek demo bank (deterministic, seed 7)
python train.py        # needs data/raw/amlworld/HI-Small_Trans.csv; writes models/
python elliptic.py     # optional external check; writes models/elliptic.json
uvicorn src.app:app --reload
```

`./deploy.sh` deploys the pushed commit to the demo server (Docker behind Caddy) and records it as
a GitHub deployment. `./deploy.sh --data` also copies `models/`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/score` | `{"transactions": [{ts, src_bank, src_account, dst_bank, dst_account, amount_uzs, rail}, …]}` → probabilities of each model, the ensemble, and reasons for the top alerts |
| GET | `/api/metrics` | evaluation metrics and live latency |
| GET | `/health` | liveness |

## Third-party software

Python libraries: PyTorch (BSD-3-Clause), PyTorch Geometric (MIT), LightGBM (MIT), scikit-learn
(BSD-3-Clause), NumPy and pandas (BSD-3-Clause), Polars (MIT), FastAPI (MIT), Uvicorn
(BSD-3-Clause), Jinja2 (BSD-3-Clause).

Front end, served locally with no CDN: htmx (0BSD) and Chart.js (MIT).

## Limitations

- **Unseen schemes.** The models learn the typologies present in AMLworld. New schemes may be missed.
- **New accounts.** Accounts with no history (cold start) have empty history features.
- **Synthetic data.** Both the training data and the demo bank are synthetic. Real performance has to
  be measured on the bank's own data in a pilot.
- **GNN view.** The GNN sees the graph as it stands. In live use, a scheme becomes clearer as more of
  its transactions arrive, so scores are refreshed whenever the graph changes.

## Repository layout

```
src/ingest.py      data loading, one schema for AMLworld and the Uzbek data
src/features.py    backward-looking transaction features
src/baseline.py    LightGBM in two stages + calibration
src/gnn.py         GraphSAGE edge classifier
src/evaluate.py    PR-AUC, precision/recall at budget, calibration
src/explain.py     TreeSHAP reasons in plain English, evidence subgraph
src/generator.py   Uzbek synthetic demo bank (CC0)
src/service.py     scoring, live injection, CSV scoring, audit
src/app.py         FastAPI routes; templates/ and static/ hold the UI
train.py           trains both models, evaluates, writes models/
elliptic.py        external check on Elliptic
```
