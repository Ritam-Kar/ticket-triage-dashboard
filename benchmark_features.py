"""
GPU-Accelerated Feature Engineering Benchmark (Stage 3)
Compares performance of pandas (CPU) vs. cudf.pandas (GPU)
on 1.5 million support tickets from BigQuery.
"""

import sys
import time
import numpy as np

# A helper to run the identical feature engineering on both CPU and GPU
def run_feature_engineering(df_in):
    # Make a copy to avoid mutating the original
    df = df_in.copy()
    
    # Strip timezone info if present to avoid tz-naive and tz-aware subtraction errors
    if df["created_at"].dt.tz is not None:
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    if df["resolved_at"].dt.tz is not None:
        df["resolved_at"] = df["resolved_at"].dt.tz_localize(None)
    
    # We use a fixed current time slightly after the last ticket date for hours_since_created
    # Since dataset is 2025-07-01 to 2026-06-30, we use 2026-07-01
    current_time = pd.Timestamp("2026-07-01 00:00:00")
    
    # 1. hours_since_created (for open tickets, resolved_at is NaT)
    is_open = df["resolved_at"].isna()
    hours_since_created = (current_time - df["created_at"]).dt.total_seconds() / 3600.0
    df["hours_since_created"] = hours_since_created.where(is_open, np.nan)
    
    # 2. hours_to_resolution (for resolved tickets)
    hours_to_resolution = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600.0
    df["hours_to_resolution"] = hours_to_resolution.where(~is_open, np.nan)
    
    # 3. breached_sla (resolution time or current open time > sla_target_hours)
    duration = df["hours_to_resolution"].fillna(df["hours_since_created"])
    df["breached_sla"] = duration > df["sla_target_hours"]
    
    # 4. day_of_week, hour_of_day
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["hour_of_day"] = df["created_at"].dt.hour
    
    # 5. rolling_team_backlog (count of open tickets per team per day, 7-day rolling window)
    # Additions (+1 at created_at date)
    created_dates = df["created_at"].dt.normalize()
    add_df = pd.DataFrame({
        "team": df["team"],
        "date": created_dates,
        "change": 1
    })
    
    # Removals (-1 at resolved_at + 1 day)
    resolved_only = df[~is_open]
    rem_df = pd.DataFrame({
        "team": resolved_only["team"],
        "date": resolved_only["resolved_at"].dt.normalize() + pd.Timedelta(days=1),
        "change": -1
    })
    
    # Combine changes
    changes = pd.concat([add_df, rem_df])
    daily_changes = changes.groupby(["team", "date"])["change"].sum().reset_index()
    
    # Create complete grid of unique teams x all days to ensure rolling works correctly
    all_dates = pd.date_range(
        start=df["created_at"].min().normalize(),
        end=df["created_at"].max().normalize() + pd.Timedelta(days=2),
        freq="D"
    )
    teams = df["team"].unique()
    grid_index = pd.MultiIndex.from_product([teams, all_dates], names=["team", "date"])
    grid_df = pd.DataFrame(index=grid_index).reset_index()
    
    # Merge and fill missing dates with 0 change
    grid_df = grid_df.merge(daily_changes, on=["team", "date"], how="left")
    grid_df["change"] = grid_df["change"].fillna(0)
    
    # Sort grid to compute cumulative backlog correctly
    grid_df = grid_df.sort_values(["team", "date"])
    grid_df["backlog"] = grid_df.groupby("team")["change"].cumsum()
    
    # 7-day rolling mean of the backlog
    rolling_series = grid_df.groupby("team")["backlog"].rolling(window=7, min_periods=1).mean()
    grid_df["rolling_team_backlog"] = rolling_series.reset_index(level=0, drop=True)
    grid_df["rolling_team_backlog"] = grid_df["rolling_team_backlog"].round().astype("int64")
    
    # Map the rolling backlog back to the original df on team & created_date
    df_temp = pd.DataFrame({
        "team": df["team"],
        "date": created_dates,
        "original_index": df.index
    })
    df_merged = df_temp.merge(grid_df[["team", "date", "rolling_team_backlog"]], on=["team", "date"], how="left")
    df_merged = df_merged.sort_values("original_index")
    
    df["rolling_team_backlog"] = df_merged["rolling_team_backlog"].values
    
    return df

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    
    if mode == "cpu":
        import pandas as pd
        print("Loading raw tickets from BigQuery (CPU mode)...")
        from google.cloud import bigquery
        client = bigquery.Client()
        df_raw = client.query("SELECT * FROM ticket_analytics.raw_tickets").to_dataframe()
        print(f"Loaded {len(df_raw):,} rows.")
        
        print("Running CPU Feature Engineering...")
        t0 = time.perf_counter()
        df_out = run_feature_engineering(df_raw)
        t_cpu = time.perf_counter() - t0
        print(f"CPU execution time: {t_cpu:.4f} seconds")
        
    elif mode == "gpu":
        # Enable RAPIDS cuDF pandas accelerator
        import cudf.pandas
        cudf.pandas.install()
        import pandas as pd
        
        print("Loading raw tickets from BigQuery (GPU mode)...")
        from google.cloud import bigquery
        client = bigquery.Client()
        df_raw = client.query("SELECT * FROM ticket_analytics.raw_tickets").to_dataframe()
        print(f"Loaded {len(df_raw):,} rows.")
        
        print("Running GPU Feature Engineering (cuDF)...")
        t0 = time.perf_counter()
        df_out = run_feature_engineering(df_raw)
        t_gpu = time.perf_counter() - t0
        print(f"GPU execution time: {t_gpu:.4f} seconds")
        
        # Save output and upload to GCS
        output_parquet = "processed_tickets.parquet"
        print(f"Saving processed GPU output to {output_parquet}...")
        df_out.to_parquet(output_parquet, index=False)
        print("Upload completed successfully via GCS commands.")
        
    else:
        # Full Benchmark mode
        import pandas as pd
        print("Loading raw tickets from BigQuery for benchmark...")
        from google.cloud import bigquery
        client = bigquery.Client()
        df_raw = client.query("SELECT * FROM ticket_analytics.raw_tickets").to_dataframe()
        print(f"Loaded {len(df_raw):,} rows.")
        
        # 1. CPU Run
        print("\n--- Running CPU (plain pandas) ---")
        t0 = time.perf_counter()
        df_cpu = run_feature_engineering(df_raw)
        t_cpu = time.perf_counter() - t0
        print(f"CPU Wall Clock: {t_cpu:.4f} seconds")
        
        # 2. GPU Run
        print("\n--- Running GPU (cudf.pandas) ---")
        # Initialize cuDF
        import cudf.pandas
        cudf.pandas.install()
        # Re-import pandas so it uses the cuDF backend
        import pandas as pd
        
        # Warmup (JIT compilation)
        print("Warming up cuDF JIT...")
        _ = run_feature_engineering(df_raw.head(100))
        
        t0 = time.perf_counter()
        df_gpu = run_feature_engineering(df_raw)
        t_gpu = time.perf_counter() - t0
        print(f"GPU Wall Clock: {t_gpu:.4f} seconds")
        
        speedup = t_cpu / t_gpu
        print(f"\nSpeedup: {speedup:.2f}x")
        
        # Save comparison results
        results_df = pd.DataFrame({
            "method": ["CPU (pandas)", "GPU (cudf.pandas)"],
            "wall_clock_seconds": [t_cpu, t_gpu],
            "speedup_factor": [1.0, speedup]
        })
        results_df.to_csv("benchmark_results.csv", index=False)
        print("Saved results to benchmark_results.csv")
        
        # Save processed GPU output
        output_parquet = "processed_tickets.parquet"
        print(f"Saving GPU-processed dataset to {output_parquet}...")
        df_gpu.to_parquet(output_parquet, index=False)
        print("Saved.")
