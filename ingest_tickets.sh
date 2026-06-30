#!/bin/bash
# End-to-end Support Ticket Ingestion Script (Stage 2)
# Generates synthetic data, uploads it to GCS, and loads it into BigQuery.
set -e

# Set local paths for tools
export PATH="/Users/ritam/google-cloud-sdk/bin:/Users/ritam/homebrew/bin:$PATH"
export CLOUDSDK_PYTHON="/Users/ritam/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11"

BUCKET_NAME="ticket-triage-dashboard-ticket-data"
DATASET_NAME="ticket_analytics"
TABLE_NAME="raw_tickets"

echo "=== 1. Generating 1.5 Million Support Tickets ==="
~/.local/bin/uv run --with numpy --with pandas --with pyarrow generate_tickets.py

echo "=== 2. Uploading tickets.parquet to Cloud Storage ==="
gcloud storage cp tickets.parquet "gs://${BUCKET_NAME}/raw/tickets.parquet"

echo "=== 3. Loading Parquet file into BigQuery table ${DATASET_NAME}.${TABLE_NAME} ==="
bq load --source_format=PARQUET "${DATASET_NAME}.${TABLE_NAME}" "gs://${BUCKET_NAME}/raw/tickets.parquet"

echo "=== Ingestion Completed Successfully ==="
