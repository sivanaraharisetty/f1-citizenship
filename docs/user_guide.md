# User Guide

## Overview

This guide provides step-by-step instructions for using the Immigration Classification Pipeline.

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

### 2. Basic Usage

```bash
# Run the complete pipeline
python scripts/run_enhanced_pipeline.py

# Or run individual components
python scripts/collect_data.py
python scripts/preprocess_data.py
python scripts/run_analysis.py
python scripts/train_model.py
```

### 3. Configuration

Edit `configs/config.yaml` to customize:

```yaml
data:
  s3:
    bucket: "your-bucket-name"
    comments_path: "reddit/comments/"
    posts_path: "reddit/posts/"
  chunking:
    files_per_chunk: 1
    rows_per_chunk: 1000

model:
  parameters:
    learning_rate: 5e-5
    batch_size: 32
    epochs: 3
```

## Detailed Usage

### Data Collection

```bash
# Collect data with custom parameters
python scripts/collect_data.py --config configs/config.yaml --output data/raw --max-chunks 10
```

### Data Preprocessing

```bash
# Preprocess collected data
python scripts/preprocess_data.py --input data/raw --output data/processed
```

### Analysis

```bash
# Run descriptive analysis
python scripts/run_analysis.py --input data/processed --output results/analysis
```

### Model Training

```bash
# Train BERT classifier
python scripts/train_model.py --input data/processed --output models --epochs 5
```

## Results

The pipeline generates organized results in the following structure:

```
results/
├── data/                    # Raw and processed data
├── analysis/                # Analysis results (JSON)
├── models/                  # Trained models
├── reports/                 # Text reports
└── visualizations/         # Charts and plots
```

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS credentials are configured
2. **Memory Issues**: Reduce `rows_per_chunk` in config
3. **Import Errors**: Check Python path and dependencies

### Getting Help

- Check the logs in `logs/` directory
- Review error messages in terminal output
- Consult the API documentation in `docs/api/`
