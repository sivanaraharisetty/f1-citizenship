#!/bin/bash
# Full Pipeline Execution Script
# Runs the complete Reddit Visa Discourse Analysis pipeline end-to-end

set -e  # Exit on any error

echo "Starting Full Reddit Visa Discourse Analysis Pipeline..."

# Set up logging
LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Pipeline started"

# Step 1: Data Sampling
log "Step 1: Data Sampling"
python src/preprocessing/data_sampling.py \
    --config config/data_config.yaml \
    --output data/interim/sampled_data.parquet \
    --log-file "$LOG_FILE"

# Step 2: Data Cleaning
log "Step 2: Data Cleaning"
python src/preprocessing/data_cleaning.py \
    --input data/interim/sampled_data.parquet \
    --output data/processed/cleaned_data.parquet \
    --config config/data_config.yaml \
    --log-file "$LOG_FILE"

# Step 3: Annotation (if needed)
if [ ! -f "data/processed/annotated_data.parquet" ]; then
    log "Step 3: Annotation"
    python src/preprocessing/annotation_system.py \
        --input data/processed/cleaned_data.parquet \
        --output data/processed/annotated_data.parquet \
        --sample-size 1000 \
        --log-file "$LOG_FILE"
else
    log "Step 3: Annotation already exists, skipping"
fi

# Step 4: Complete Dataset Analysis
log "Step 4: Complete Dataset Analysis"
python src/analysis/complete_no_sampling_analysis.py \
    --log-file "$LOG_FILE"

# Step 5: Temporal Analysis
log "Step 5: Temporal Analysis"
python src/analysis/temporal_analysis.py \
    --config config/analysis_config.yaml \
    --output results/reports/temporal_analysis.json \
    --log-file "$LOG_FILE"

# Step 6: Visualization
log "Step 6: Visualization"
python src/visualization/visualization_tools.py \
    --analysis-results results/reports/temporal_analysis.json \
    --output results/visualizations/ \
    --log-file "$LOG_FILE"

log "Pipeline completed successfully!"
log "Results saved to: results/"
log "Logs saved to: $LOG_FILE"

echo ""
echo "Full pipeline completed!"
echo "Check results in: results/"
echo "Check logs in: $LOG_FILE"
