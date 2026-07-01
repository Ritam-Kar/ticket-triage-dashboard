# Ticket Triage & Forecast Dashboard

A Program Manager-facing data app that triages IT/support tickets in real time — showing which open tickets are at risk of breaching SLA, forecasting incoming volume per team for the next 7 days, and proving GPU acceleration via NVIDIA cuDF speeds up the processing pipeline vs plain CPU pandas.

Built as a hackathon submission using **Google Cloud** and **NVIDIA** acceleration tools.

---

## 🔗 Live Links

| Resource | URL |
|---|---|
| **Live App (Cloud Run)** | https://ticket-triage-demo-326806126135.us-central1.run.app |
| **Full Dashboard (Looker Studio)** | https://datastudio.google.com/reporting/82124fce-4aa0-4530-828a-8f3f41e4aab7 |
| **GitHub Repo** | https://github.com/Ritam-Kar/ticket-triage-dashboard |

---

## ⚡ GPU Acceleration Result

| Method | Time (5M rows) | Speedup |
|---|---|---|
| Pandas (CPU) | 6.2797s | 1.0x |
| cuDF / cudf.pandas (GPU) | 3.6053s | **1.74x** |

Feature engineering (groupby, rolling aggregations, time-delta calculations) on 5,000,000 rows, run on an NVIDIA L4 GPU (GCE `g2-standard-4`). Identical code both runs — only `cudf.pandas.install()` added for the GPU run. No code rewrite required.

---

## 🏗️ Architecture

```
Synthetic Data Generator
        ↓
Cloud Storage (GCS) — raw Parquet
        ↓
GCE VM (NVIDIA L4 GPU)
  → cuDF / cudf.pandas feature engineering
  → CPU vs GPU benchmark logged
        ↓
BigQuery (ticket_analytics dataset)
  → raw_tickets
  → processed_tickets
  → ticket_risk_scores
  → volume_forecast
        ↓
┌─────────────────────┬──────────────────────┐
│  Looker Studio      │  Cloud Run (Streamlit)│
│  (full dashboard)   │  (public demo app)   │
└─────────────────────┴──────────────────────┘
```

---

## 📊 Dashboard Views

1. **Risk-Ranked Open Tickets Table** — 502,636 open tickets sorted by risk score (0–100), with team, category, priority, and risk band
2. **Risk Band Distribution** — bar chart of ticket counts by band (Critical / High / Medium / Low)
3. **Backlog Heatmap** — team × day-of-week pivot table showing volume hotspots
4. **7-Day Volume Forecast** — per-team predicted ticket count, Jul 1–7 2026
5. **CPU vs GPU Benchmark** — processing time comparison bar chart with 1.74x speedup callout

---

## 🛠️ Tech Stack

**Google Cloud**
- Cloud Storage — raw data landing zone
- BigQuery — analytics warehouse (4 tables, 5M+ rows)
- Cloud Run — public deployment of the Streamlit app
- Looker Studio — full interactive dashboard

**NVIDIA**
- `cudf.pandas` (RAPIDS) — drop-in GPU acceleration for pandas, zero code rewrite
- NVIDIA L4 GPU on Google Cloud (GCE `g2-standard-4`) — GPU compute for benchmarking

**Application**
- Python, Streamlit, Plotly, google-cloud-bigquery
- Docker, Google Cloud Build, Artifact Registry

---

## 📁 Repository Structure

```
├── generate_tickets.py      # Stage 2: synthetic 5M-row dataset generator
├── ingest_tickets.sh        # Stage 2: GCS ingestion script
├── benchmark_features.py    # Stage 3: CPU vs GPU benchmark (identical pipeline, two runs)
├── benchmark_results.csv    # Stage 3: actual timing results
├── risk_and_forecast.py     # Stage 4: risk scoring + 7-day volume forecast
├── app.py                   # Stage 6: Streamlit Cloud Run app
├── Dockerfile               # Stage 6: container config
├── requirements.txt         # Stage 6: Python dependencies
└── .dockerignore            # Stage 6: keeps build context lightweight
```

---

## 🔢 Dataset

- **5,000,000** synthetic IT/project support tickets
- 12-month date range (Jul 2025 – Jun 2026)
- Fields: `ticket_id`, `created_at`, `resolved_at`, `category`, `priority`, `team`, `assignee_id`, `customer_tier`, `status`, `sla_target_hours`, `first_response_minutes`, `reopened_count`
- ~5% intentional messiness injected (nulls, missing fields) to simulate real cleaning requirements
- Open ticket distribution uses exponential decay weighting — recent tickets are far more likely to be open than old ones (realistic behaviour)

---

## 📈 Risk Scoring Methodology

Risk score (0–100) for each open ticket computed as:

| Component | Formula | Max Points |
|---|---|---|
| Urgency | `min(hours_since_created / sla_target_hours, 3) / 3 × 55` | 55 |
| Priority weight | P1=25, P2=15, P3=8, P4=3 | 25 |
| Reopened count | `reopened_count × 5`, capped | 20 |

Risk bands assigned via **percentile-based thresholds** (not fixed cutoffs) for a realistic distribution:

| Band | Share | Count |
|---|---|---|
| Critical | 5.2% | ~26,098 |
| High | 19.8% | ~99,633 |
| Medium | 43.4% | ~217,941 |
| Low | 31.6% | ~158,964 |

---

## 🚀 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Requires GCP credentials with BigQuery read access (`roles/bigquery.dataViewer`) on the `ticket-triage-dashboard` project.
