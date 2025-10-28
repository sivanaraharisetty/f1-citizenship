# Makefile for BERT Classifier Project
# Professional CLI commands for reproducible workflows

.PHONY: help setup train analyze visualize clean test lint format

# Default target
help:
	@echo "BERT Classifier - Professional Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup          - Setup development environment"
	@echo "  make install        - Install dependencies"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make sample         - Run data sampling"
	@echo "  make clean-data     - Run data cleaning"
	@echo "  make annotate       - Run annotation system"
	@echo ""
	@echo "Machine Learning:"
	@echo "  make train          - Train BERT classifier"
	@echo "  make evaluate       - Evaluate model performance"
	@echo "  make predict        - Run inference on new data"
	@echo ""
	@echo "Analysis:"
	@echo "  make analyze        - Run temporal analysis"
	@echo "  make visualize      - Generate visualizations"
	@echo "  make report         - Generate comprehensive report"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  make pipeline       - Run complete end-to-end pipeline"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test           - Run unit tests"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean temporary files"
	@echo "  make clean-data     - Clean processed data"
	@echo "  make clean-models   - Clean model checkpoints"

# Setup & Environment
setup:
	@echo "Setting up development environment..."
	./scripts/setup_env.sh

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Data Pipeline
sample:
	@echo "Running data sampling..."
	python src/preprocessing/data_sampling.py --config config/data_config.yaml

clean-data:
	@echo "Running data cleaning..."
	python src/preprocessing/data_cleaning.py --config config/data_config.yaml

annotate:
	@echo "Running annotation system..."
	python src/preprocessing/annotation_system.py --config config/data_config.yaml

# Machine Learning
train:
	@echo "Training BERT classifier..."
	python src/classification/bert_classifier.py --config config/model_config.yaml

evaluate:
	@echo "Evaluating model performance..."
	python src/analysis/evaluation_metrics.py --config config/model_config.yaml

predict:
	@echo "Running inference..."
	python src/classification/inference.py --config config/model_config.yaml

# Analysis
analyze:
	@echo "Running temporal analysis..."
	python src/analysis/temporal_analysis.py --config config/analysis_config.yaml

visualize:
	@echo "Generating visualizations..."
	python src/visualization/visualization_tools.py --config config/analysis_config.yaml

report:
	@echo "Generating comprehensive report..."
	python src/visualization/report_builder.py --config config/analysis_config.yaml

# Full Pipeline
pipeline:
	@echo "Running complete end-to-end pipeline..."
	./scripts/run_pipeline.sh

# Testing & Quality
test:
	@echo "Running unit tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running code linting..."
	flake8 src/ --max-line-length=100
	pylint src/

format:
	@echo "Formatting code..."
	black src/ --line-length=100
	isort src/

# Maintenance
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name ".DS_Store" -delete

clean-data:
	@echo "Cleaning processed data..."
	rm -rf data/interim/*
	rm -rf data/processed/*

clean-models:
	@echo "Cleaning model checkpoints..."
	rm -rf models/checkpoints/*
	rm -rf models/tokenizer/*

# Development helpers
dev-setup: setup install
	@echo "Development environment ready!"

quick-test: sample clean-data train evaluate
	@echo "Quick test pipeline completed!"

full-test: pipeline
	@echo "Full test pipeline completed!"
