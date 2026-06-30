"""
Stage 4: SLA Breach Risk Scoring + 7-Day Volume Forecast
- Reads processed/processed_tickets.parquet from GCS
- Computes risk_score (0-100) and risk_band for open tickets
- Computes 7-day forecast of daily ticket volume per team
- Loads both into BigQuery
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery, storage
import io
import datetime

PROJECT = "ticket-triage-dashboard"
DATASET = "ticket_analytics"
BUCKET  = "ticket-triage-dashboard-ticket-data"
PARQUET_PATH = "processed/processed_tickets.parquet"

RISK_TABLE     = f"{PROJECT}.{DATASET}.ticket_risk_scores"
FORECAST_TABLE = f"{PROJECT}.{DATASET}.volume_forecast"

# ---------------------------------------------------------------------------
# 1. Load processed parquet from GCS
# ---------------------------------------------------------------------------
print("Loading processed_tickets.parquet from GCS...")
storage_client = storage.Client(project=PROJECT)
blob = storage_client.bucket(BUCKET).blob(PARQUET_PATH)
data = blob.download_as_bytes()
df = pd.read_parquet(io.BytesIO(data))
print(f"Loaded {len(df):,} rows.  Columns: {list(df.columns)}")

# Ensure datetime columns are proper timestamps
for col in ["created_at", "resolved_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

# ---------------------------------------------------------------------------
# 2. Risk Scoring (open tickets only)
# ---------------------------------------------------------------------------
print("\n--- Building risk scores for open tickets ---")

open_mask = df["status"].isin(["Open", "In Progress"])
open_df = df[open_mask].copy()
print(f"Open tickets: {len(open_df):,}")

# hours_since_created: compute from created_at relative to now
NOW = datetime.datetime(2026, 7, 1, 0, 0, 0)  # fixed reference matching benchmark
open_df["hours_since_created"] = (NOW - open_df["created_at"]).dt.total_seconds() / 3600
open_df["sla_target_hours"] = pd.to_numeric(open_df["sla_target_hours"], errors="coerce").fillna(48)

# Component 1: urgency ratio capped at 3x SLA → max 55 pts
# A ticket at exactly its SLA limit scores 55/3 ≈ 18 pts (not the full 55)
urgency_ratio = (open_df["hours_since_created"] / open_df["sla_target_hours"]).clip(0, 3)
urgency_score = (urgency_ratio / 3 * 55).clip(0, 55)   # max 55 pts

# Component 2: priority weight — reduced to avoid double-penalising recent P1s
priority_map = {"P1": 25, "P2": 15, "P3": 8, "P4": 3}
priority_score = open_df["priority"].map(priority_map).fillna(3)   # max 25 pts

# Component 3: reopened_count — increased weight (each reopen = +5, capped at 20 pts)
reopen_score = (open_df["reopened_count"].fillna(0) * 5).clip(0, 20)   # max 20 pts

open_df["risk_score"] = (urgency_score + priority_score + reopen_score).clip(0, 100).round(2)

# Risk band buckets based on percentile rank:
# - Critical: top 5% of scores (95th percentile and above)
# - High: next 15% (80th-95th percentile)
# - Medium: next 35% (45th-80th percentile)
# - Low: bottom 45% (below 45th percentile)
open_df["pct_rank"] = open_df["risk_score"].rank(pct=True)
open_df["risk_band"] = pd.cut(
    open_df["pct_rank"],
    bins=[0.0, 0.45, 0.80, 0.95, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
    include_lowest=True
).astype(str)
open_df["computed_at"] = pd.Timestamp(NOW)

risk_out = open_df[["ticket_id", "risk_score", "risk_band", "computed_at"]].copy()
risk_out.to_parquet("ticket_risk_scores.parquet", index=False)
print(risk_out["risk_band"].value_counts().to_string())
print(f"\nSample risk scores:\n{risk_out.head(10).to_string(index=False)}")

# ---------------------------------------------------------------------------
# 3. 7-Day Volume Forecast (moving average / linear trend per team)
# ---------------------------------------------------------------------------
print("\n--- Building 7-day volume forecast ---")

# Use last 90 days of created_at
CUTOFF = NOW - datetime.timedelta(days=90)
hist = df[df["created_at"] >= CUTOFF].copy()
hist["date"] = hist["created_at"].dt.date

daily = (
    hist.groupby(["team", "date"])
    .size()
    .reset_index(name="ticket_count")
)
daily["date"] = pd.to_datetime(daily["date"])

teams = daily["team"].unique()
forecast_rows = []
forecast_start = datetime.date(2026, 7, 1)
forecast_dates = [forecast_start + datetime.timedelta(days=i) for i in range(7)]

for team in teams:
    team_data = daily[daily["team"] == team].sort_values("date")
    counts = team_data["ticket_count"].values

    if len(counts) < 7:
        # Fallback: use mean
        trend_pred = np.full(7, counts.mean())
    else:
        # Linear trend fit on the last 30 data points
        recent = counts[-30:] if len(counts) >= 30 else counts
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        next_x = np.arange(len(recent), len(recent) + 7)
        trend_pred = np.polyval(coeffs, next_x)

    # Blend with 7-day moving average for stability
    ma7 = np.mean(counts[-7:]) if len(counts) >= 7 else np.mean(counts)
    blended = (0.6 * trend_pred + 0.4 * ma7).clip(0)

    for d, pred in zip(forecast_dates, blended):
        forecast_rows.append({
            "team": team,
            "forecast_date": d,
            "predicted_ticket_count": int(round(pred))
        })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"])
forecast_df.to_parquet("volume_forecast.parquet", index=False)
print(f"\nForecast rows: {len(forecast_df)}")
print(forecast_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Load into BigQuery
# ---------------------------------------------------------------------------
bq = bigquery.Client(project=PROJECT)

def bq_load(df, table_id, schema):
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = bq.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {job.output_rows} rows → {table_id}")

print("\n--- Loading ticket_risk_scores into BigQuery ---")
risk_schema = [
    bigquery.SchemaField("ticket_id",   "STRING"),
    bigquery.SchemaField("risk_score",  "FLOAT64"),
    bigquery.SchemaField("risk_band",   "STRING"),
    bigquery.SchemaField("computed_at", "TIMESTAMP"),
]
bq_load(risk_out, RISK_TABLE, risk_schema)

print("\n--- Loading volume_forecast into BigQuery ---")
forecast_schema = [
    bigquery.SchemaField("team",                   "STRING"),
    bigquery.SchemaField("forecast_date",          "DATE"),
    bigquery.SchemaField("predicted_ticket_count", "INT64"),
]
forecast_bq = forecast_df.copy()
forecast_bq["forecast_date"] = forecast_bq["forecast_date"].dt.date
bq_load(forecast_bq, FORECAST_TABLE, forecast_schema)

print("\n=== Stage 4 complete! ===")
