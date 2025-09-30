# Developer Guide

## Architecture

The Immigration Classification Pipeline follows a modular architecture:

```
src/
├── data/              # Data loading and preprocessing
├── model/             # Model training and evaluation
├── analysis/          # Descriptive and comparative analysis
└── utils/             # Utility functions
```

## Adding New Features

### 1. New Data Sources

To add support for new data sources:

1. Create a new loader in `src/data/`
2. Implement the same interface as existing loaders
3. Update the main pipeline to use the new loader

### 2. New Analysis Methods

To add new analysis methods:

1. Add methods to `src/analysis/descriptive_analysis.py`
2. Update the comprehensive analysis to include new methods
3. Add corresponding tests in `tests/test_analysis.py`

### 3. New Model Types

To add support for new model types:

1. Create a new trainer class in `src/model/`
2. Implement the same interface as `BertClassifierTrainer`
3. Update the training script to support the new model

## Code Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions with docstrings

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use pytest for testing framework

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions and classes
- Update this guide for architectural changes

## Development Workflow

### 1. Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### 2. Making Changes

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# Run tests
python -m pytest tests/

# Run linting
black src/ tests/
flake8 src/ tests/

# Commit changes
git commit -m "Add new feature"
```

### 3. Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_preprocessing.py

# Run with coverage
python -m pytest --cov=src tests/
```

## API Reference

### Data Module

#### `iter_load_data(bucket_name, comments_prefix, posts_prefix, **kwargs)`

Loads data from S3 in chunks.

**Parameters:**
- `bucket_name`: S3 bucket name
- `comments_prefix`: Prefix for comments data
- `posts_prefix`: Prefix for posts data
- `files_per_chunk`: Number of files per chunk
- `rows_per_chunk`: Number of rows per chunk

**Returns:** Generator of (DataFrame, keys) tuples

#### `preprocess_data(df)`

Preprocesses Reddit data for classification.

**Parameters:**
- `df`: Input DataFrame with text data

**Returns:** Processed DataFrame with text and labels

### Model Module

#### `BertClassifierTrainer(num_labels)`

BERT-based text classifier trainer.

**Parameters:**
- `num_labels`: Number of classification labels

**Methods:**
- `train_on_dataframe(df, epochs, save_dir)`: Train on DataFrame
- `evaluate(test_data)`: Evaluate on test data

### Analysis Module

#### `ImmigrationDataAnalyzer()`

Comprehensive data analysis for immigration content.

**Methods:**
- `extract_keywords(texts, top_n)`: Extract top keywords
- `analyze_immigration_keywords(texts)`: Analyze immigration terms
- `cluster_topics(texts, n_clusters)`: Cluster topics
- `comprehensive_analysis(df)`: Run full analysis

## Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Install production dependencies
   pip install -r requirements.txt
   
   # Configure environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```

2. **Configuration**
   ```bash
   # Update production config
   cp configs/config.yaml configs/production.yaml
   # Edit production.yaml with production settings
   ```

3. **Run Pipeline**
   ```bash
   # Run with production config
   python scripts/run_enhanced_pipeline.py --config configs/production.yaml
   ```

### Monitoring

- Check logs in `logs/` directory
- Monitor S3 data access
- Track model performance metrics
- Set up alerts for failures

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Address review feedback

### Code Review

- All code must be reviewed
- Tests must pass
- Documentation must be updated
- Performance impact considered
