# Immigration Classification Pipeline

A comprehensive machine learning pipeline for classifying immigration-related content from Reddit data using BERT-based models.

## Features

- **Enhanced Data Collection**: Automated S3 data loading with configurable chunking
- **Advanced Preprocessing**: Immigration-specific text cleaning and labeling
- **Descriptive Analysis**: Comprehensive data analysis with topic modeling and clustering
- **BERT Classification**: State-of-the-art text classification using Hugging Face Transformers
- **Pre/Post Analysis**: Detailed performance evaluation and comparison
- **Production Ready**: Scalable, configurable, and well-documented

## Project Structure

```
s3-classifier-project/
├── README.md              # Project overview & setup instructions
├── .gitignore             # Ignore unnecessary files
├── requirements.txt       # Dependencies
├── setup.py               # (optional) Packaging setup
├── src/                   # Source code
│   ├── __init__.py
│   ├── data/              # Data loading and preprocessing
│   ├── model/             # Model training and evaluation
│   ├── analysis/          # Descriptive and comparative analysis
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter/Colab notebooks
├── data/                  # Small sample data (not large raw data)
│   ├── raw/               # Raw data samples
│   └── processed/         # Processed data samples
├── tests/                 # Unit and integration tests
├── scripts/               # Automation, ETL, training jobs, etc.
├── configs/               # YAML/JSON config files
├── docs/                  # Documentation
└── infra/                 # Deployment/infra (Terraform, CloudFormation, etc.)
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- AWS credentials configured
- Access to S3 bucket with Reddit data

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd s3-classifier-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```

4. **Update configuration**
   ```bash
   # Edit configs/config.yaml with your S3 bucket details
   ```

## Quick Start

### Run Enhanced Pipeline

```bash
# Run the complete enhanced pipeline
python scripts/run_enhanced_pipeline.py

# Run with specific configuration
python scripts/run_enhanced_pipeline.py --config configs/production.yaml
```

### Run Individual Components

```bash
# Data collection only
python scripts/collect_data.py

# Preprocessing only
python scripts/preprocess_data.py

# Analysis only
python scripts/run_analysis.py

# Model training only
python scripts/train_model.py
```

## Usage Examples

### Basic Usage

```python
from src.data.loader import iter_load_data
from src.data.preprocess import preprocess_data
from src.model.train import BertClassifierTrainer

# Load data
for df, keys in iter_load_data(bucket_name, comments_prefix, posts_prefix):
    # Preprocess
    processed = preprocess_data(df)
    
    # Train model
    trainer = BertClassifierTrainer(num_labels=3)
    metrics = trainer.train_on_dataframe(processed)
```

### Enhanced Pipeline

```python
from scripts.run_enhanced_pipeline import CleanEnhancedPipeline

# Initialize pipeline
pipeline = CleanEnhancedPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline()
```

## Results

The pipeline generates comprehensive results including:

- **Data Analysis**: Descriptive statistics, keyword analysis, topic modeling
- **Model Performance**: Accuracy, precision, recall, F1 scores
- **Visualizations**: Word clouds, confusion matrices, performance charts
- **Reports**: Detailed analysis reports in JSON and text formats

## Configuration

### Main Configuration (`configs/config.yaml`)

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
    num_labels: 3

analysis:
  topics:
    n_topics: 5
    n_clusters: 5
  keywords:
    top_n: 50
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_preprocessing.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Documentation

- **API Documentation**: `docs/api/`
- **User Guide**: `docs/user_guide.md`
- **Developer Guide**: `docs/developer_guide.md`
- **Deployment Guide**: `docs/deployment.md`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:

- Create an issue in the repository
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`

---

**Status**: Production Ready