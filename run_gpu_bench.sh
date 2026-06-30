#!/bin/bash
# Local Orchestrator for GPU Feature Engineering Benchmark (Stage 3)
set -e

# Setup paths for gcloud
export PATH="/Users/ritam/google-cloud-sdk/bin:/Users/ritam/homebrew/bin:$PATH"
export CLOUDSDK_PYTHON="/Users/ritam/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11"

VM_NAME="gpu-benchmark-instance"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
PROJECT_ID="ticket-triage-dashboard"

VM_CREATED=false
ACTIVE_ZONE=""

# Deletion Trap - guarantees cleanup regardless of success or failure
cleanup() {
    if [ "$VM_CREATED" = "true" ]; then
        echo "--------------------------------------------------"
        echo "CLEANUP: Deleting VM instance ${VM_NAME} in zone ${ACTIVE_ZONE}..."
        gcloud compute instances delete "${VM_NAME}" --zone="${ACTIVE_ZONE}" --quiet || true
        echo "CLEANUP: Done."
    fi
}
trap cleanup EXIT INT TERM

# We define candidate search runs:
# Each run try spot or standard, L4 or T4, and a list of candidate zones.
# Spot (preemptible) has separate resource pools and often higher availability.
RUNS=(
    "g2-standard-4|nvidia-l4|--preemptible|us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east1-d us-east4-a us-west1-a us-west1-b"
    "n1-standard-4|nvidia-tesla-t4|--preemptible|us-central1-a us-central1-c us-central1-f us-east1-c us-east1-d us-east4-a us-west1-a us-west1-b"
    "g2-standard-4|nvidia-l4||us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east1-d us-east4-a us-west1-a us-west1-b"
    "n1-standard-4|nvidia-tesla-t4||us-central1-a us-central1-c us-central1-f us-east1-c us-east1-d us-east4-a us-west1-a us-west1-b"
)

echo "=== 1. Provisioning VM instance ${VM_NAME} ==="

for RUN in "${RUNS[@]}"; do
    IFS="|" read -r MACHINE_TYPE ACCEL_TYPE EXTRA_FLAGS ZONES_STR <<< "$RUN"
    read -r -a CONFIG_ZONES <<< "$ZONES_STR"
    
    echo "Trying: Machine=${MACHINE_TYPE}, GPU=${ACCEL_TYPE}, Flags=${EXTRA_FLAGS}..."
    
    for ZONE in "${CONFIG_ZONES[@]}"; do
        echo "Attempting creation in zone ${ZONE}..."
        if [ -n "$EXTRA_FLAGS" ]; then
            CREATE_CMD="gcloud compute instances create ${VM_NAME} --zone=${ZONE} --machine-type=${MACHINE_TYPE} --accelerator=type=${ACCEL_TYPE},count=1 --image-family=${IMAGE_FAMILY} --image-project=${IMAGE_PROJECT} --maintenance-policy=TERMINATE --boot-disk-size=100GB --scopes=cloud-platform ${EXTRA_FLAGS} --quiet"
        else
            CREATE_CMD="gcloud compute instances create ${VM_NAME} --zone=${ZONE} --machine-type=${MACHINE_TYPE} --accelerator=type=${ACCEL_TYPE},count=1 --image-family=${IMAGE_FAMILY} --image-project=${IMAGE_PROJECT} --maintenance-policy=TERMINATE --boot-disk-size=100GB --scopes=cloud-platform --quiet"
        fi
        
        if eval "$CREATE_CMD"; then
            VM_CREATED=true
            ACTIVE_ZONE="${ZONE}"
            echo "VM successfully created in zone ${ACTIVE_ZONE}."
            break 2
        else
            echo "Could not create VM in zone ${ZONE}."
        fi
    done
done

if [ "$VM_CREATED" = "false" ]; then
    echo "ERROR: Failed to provision VM with GPU in all attempted configurations/zones."
    echo "Please check your project GPU quota or try again later."
    exit 1
fi

echo "=== 2. Waiting 60 seconds for VM initialization and SSH service ==="
sleep 60

# Setup remote dependencies
echo "=== 3. Installing dependencies on remote VM ==="
gcloud compute ssh "${VM_NAME}" --zone="${ACTIVE_ZONE}" --ssh-flag="-oStrictHostKeyChecking=no" --command="
    sudo apt-get update && \
    sudo apt-get install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install google-cloud-bigquery db-dtypes pyarrow && \
    pip3 install --extra-index-url https://pypi.nvidia.com cudf-cu12
"

# Copy benchmark script to VM (using valid arguments)
echo "=== 4. Copying benchmark_features.py to VM ==="
gcloud compute scp benchmark_features.py "${VM_NAME}":~/benchmark_features.py --zone="${ACTIVE_ZONE}"

# Run benchmark
echo "=== 5. Running Benchmark Script on VM ==="
gcloud compute ssh "${VM_NAME}" --zone="${ACTIVE_ZONE}" --ssh-flag="-oStrictHostKeyChecking=no" --command="python3 ~/benchmark_features.py"

# Copy the generated output file to Cloud Storage
echo "=== 6. Uploading processed_tickets.parquet to GCS ==="
gcloud compute ssh "${VM_NAME}" --zone="${ACTIVE_ZONE}" --ssh-flag="-oStrictHostKeyChecking=no" --command="gcloud storage cp processed_tickets.parquet gs://${PROJECT_ID}-ticket-data/processed/processed_tickets.parquet"

# Download the CSV results locally (using valid arguments)
echo "=== 7. Downloading benchmark results CSV ==="
gcloud compute scp "${VM_NAME}":~/benchmark_results.csv ./benchmark_results.csv --zone="${ACTIVE_ZONE}"

echo "=== 8. Verification query against uploaded Parquet in GCS ==="
gcloud storage ls "gs://${PROJECT_ID}-ticket-data/processed/processed_tickets.parquet"

echo "=== Execution Finished! ==="
