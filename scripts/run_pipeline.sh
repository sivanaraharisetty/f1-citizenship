#!/bin/bash
# Full Pipeline Execution Script
# Runs the complete BERT classifier pipeline end-to-end

set -e  # Exit on any error

echo "Starting Full BERT Classifier Pipeline..."

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

# Step 4: Model Training
log "Step 4: Model Training"
python src/classification/bert_classifier.py \
    --input data/processed/annotated_data.parquet \
    --config config/model_config.yaml \
    --output models/checkpoints/ \
    --log-file "$LOG_FILE"

# Step 5: Model Evaluation
log "Step 5: Model Evaluation"
python src/analysis/evaluation_metrics.py \
    --model-path models/checkpoints/ \
    --test-data data/processed/annotated_data.parquet \
    --output results/metrics/evaluation_results.json \
    --log-file "$LOG_FILE"

# Step 6: Full Dataset Classification
log "Step 6: Full Dataset Classification"
python src/classification/inference.py \
    --model-path models/checkpoints/ \
    --input data/processed/cleaned_data.parquet \
    --output data/processed/classified_data.parquet \
    --log-file "$LOG_FILE"

# Step 7: Temporal Analysis
log "Step 7: Temporal Analysis"
python src/analysis/temporal_analysis.py \
    --input data/processed/classified_data.parquet \
    --config config/analysis_config.yaml \
    --output results/reports/temporal_analysis.json \
    --log-file "$LOG_FILE"

# Step 8: Visualization
log "Step 8: Visualization"
python src/visualization/visualization_tools.py \
    --input data/processed/classified_data.parquet \
    --analysis-results results/reports/temporal_analysis.json \
    --output results/visualizations/ \
    --log-file "$LOG_FILE"

# Step 9: Report Generation
log "Step 9: Report Generation"
python src/visualization/report_builder.py \
    --data data/processed/classified_data.parquet \
    --metrics results/metrics/evaluation_results.json \
    --analysis results/reports/temporal_analysis.json \
    --output results/reports/comprehensive_report.html \
    --log-file "$LOG_FILE"

log "Pipeline completed successfully!"
log "Results saved to: results/"
log "Logs saved to: $LOG_FILE"

echo ""
echo "Full pipeline completed!"
echo "Check results in: results/"
echo "Check logs in: $LOG_FILE"
