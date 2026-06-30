"""
Synthetic Support Ticket Data Generator
Generates ~1.5M realistic IT/project support tickets for the
Ticket Triage & Forecast Dashboard project.

Run this locally or have Antigravity run it as part of Stage 2.
Output: tickets.parquet (and a smaller tickets_sample.csv for quick inspection)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

# ---------- CONFIG ----------
N_ROWS = 5_000_000
START_DATE = datetime(2025, 7, 1)
END_DATE = datetime(2026, 6, 30)
SEED = 42
OUTPUT_PARQUET = "tickets.parquet"
OUTPUT_SAMPLE_CSV = "tickets_sample.csv"
# -----------------------------

rng = np.random.default_rng(SEED)

CATEGORIES = ["bug", "feature_request", "access_issue", "outage", "billing", "other"]
CATEGORY_WEIGHTS = [0.30, 0.15, 0.20, 0.05, 0.15, 0.15]

PRIORITIES = ["P1", "P2", "P3", "P4"]
PRIORITY_WEIGHTS = [0.05, 0.20, 0.45, 0.30]
SLA_HOURS = {"P1": 4, "P2": 12, "P3": 48, "P4": 96}

TEAMS = ["Platform", "Mobile", "Data", "Infra", "Support"]
TEAM_WEIGHTS = [0.25, 0.20, 0.15, 0.20, 0.20]

CUSTOMER_TIERS = ["Free", "Pro", "Enterprise"]
TIER_WEIGHTS = [0.50, 0.35, 0.15]

STATUSES = ["Open", "In Progress", "Resolved", "Closed", "Reopened"]


def random_timestamps(n, start, end):
    """Generate created_at timestamps with weekday seasonality + occasional spike days."""
    total_days = (end - start).days
    day_offsets = rng.integers(0, total_days, size=n)
    base_dates = np.array([start + timedelta(days=int(d)) for d in day_offsets])

    # Weekday seasonality: fewer tickets on weekends
    weekday_mask = np.array([d.weekday() < 5 for d in base_dates])
    keep_prob = np.where(weekday_mask, 1.0, 0.4)
    keep = rng.random(n) < keep_prob

    # Re-roll dropped weekend tickets onto weekdays to keep N consistent
    while not keep.all():
        n_redo = (~keep).sum()
        redo_offsets = rng.integers(0, total_days, size=n_redo)
        redo_dates = np.array([start + timedelta(days=int(d)) for d in redo_offsets])
        base_dates[~keep] = redo_dates
        weekday_mask = np.array([d.weekday() < 5 for d in base_dates])
        keep_prob = np.where(weekday_mask, 1.0, 0.4)
        keep = rng.random(n) < keep_prob

    # add random time-of-day, biased toward business hours
    hours = rng.normal(loc=13, scale=4, size=n).clip(0, 23).astype(int)
    minutes = rng.integers(0, 60, size=n)
    timestamps = [
        d.replace(hour=int(h), minute=int(m))
        for d, h, m in zip(base_dates, hours, minutes)
    ]
    return pd.to_datetime(timestamps)


def generate():
    n = N_ROWS
    print(f"Generating {n:,} synthetic tickets...")

    created_at = random_timestamps(n, START_DATE, END_DATE)
    category = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    priority = rng.choice(PRIORITIES, size=n, p=PRIORITY_WEIGHTS)
    team = rng.choice(TEAMS, size=n, p=TEAM_WEIGHTS)
    customer_tier = rng.choice(CUSTOMER_TIERS, size=n, p=TIER_WEIGHTS)

    sla_target_hours = np.array([SLA_HOURS[p] for p in priority])

    # Resolution behavior: P1 resolved faster on average but with more variance under load
    base_resolution_hours = {
        "P1": 5, "P2": 14, "P3": 50, "P4": 90,
    }
    resolution_noise = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    resolution_hours = np.array(
        [base_resolution_hours[p] for p in priority]
    ) * resolution_noise

    # ~12% of tickets still open (not yet resolved)
    still_open_mask = rng.random(n) < 0.12
    resolved_at = created_at + pd.to_timedelta(resolution_hours, unit="h")
    resolved_at = pd.Series(resolved_at)
    resolved_at[still_open_mask] = pd.NaT

    status = np.where(
        still_open_mask,
        rng.choice(["Open", "In Progress"], size=n, p=[0.5, 0.5]),
        rng.choice(["Resolved", "Closed", "Reopened"], size=n, p=[0.55, 0.35, 0.10]),
    )

    first_response_minutes = rng.exponential(scale=45, size=n).clip(1, 2000)
    reopened_count = rng.poisson(lam=0.15, size=n)
    assignee_id = rng.integers(1000, 1200, size=n)  # ~200 simulated agents

    ticket_id = [f"TCK-{uuid.uuid4().hex[:10]}" for _ in range(n)]

    df = pd.DataFrame({
        "ticket_id": ticket_id,
        "created_at": created_at,
        "resolved_at": resolved_at,
        "category": category,
        "priority": priority,
        "team": team,
        "assignee_id": [f"AGT-{a}" for a in assignee_id],
        "customer_tier": customer_tier,
        "status": status,
        "sla_target_hours": sla_target_hours,
        "first_response_minutes": first_response_minutes.round(1),
        "reopened_count": reopened_count,
    })

    # ---- inject ~5% messiness for realistic cleaning step ----
    n_messy = int(n * 0.05)
    messy_idx = rng.choice(n, size=n_messy, replace=False)
    half = n_messy // 2
    df.loc[messy_idx[:half], "first_response_minutes"] = np.nan
    df.loc[messy_idx[half:], "customer_tier"] = None

    # Floor timestamps to seconds to avoid precision issues
    df["created_at"] = df["created_at"].dt.floor("s")
    df["resolved_at"] = df["resolved_at"].dt.floor("s")
    df["sla_target_hours"] = df["sla_target_hours"].astype(float)

    print("Done. Sample:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Open/unresolved tickets: {df['resolved_at'].isna().sum():,}")

    df.to_parquet(OUTPUT_PARQUET, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    df.sample(5000, random_state=SEED).to_csv(OUTPUT_SAMPLE_CSV, index=False)
    print(f"\nSaved full dataset to {OUTPUT_PARQUET}")
    print(f"Saved 5,000-row sample to {OUTPUT_SAMPLE_CSV} for quick inspection")


if __name__ == "__main__":
    generate()
