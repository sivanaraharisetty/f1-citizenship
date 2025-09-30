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
├── setup.py               # Packaging setup
├── src/                   # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── model/             # Model training and evaluation
│   ├── analysis/          # Descriptive and comparative analysis
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter/Colab notebooks
├── data/                  # Small sample data
├── tests/                 # Unit and integration tests
├── scripts/               # Automation, ETL, training jobs
├── configs/               # YAML/JSON config files
├── docs/                  # Documentation
└── infra/                 # Deployment/infra
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- AWS credentials configured
- Access to S3 bucket with Reddit data

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure AWS credentials**
   ```bash
   aws configure
   ```

3. **Update configuration**
   ```bash
   # Edit configs/config.yaml with your S3 bucket details
   ```

## Quick Start

### Run Enhanced Pipeline

```bash
# Run the complete enhanced pipeline
python scripts/run_enhanced_pipeline.py

# Or run individual components
python scripts/collect_data.py
python scripts/preprocess_data.py
python scripts/run_analysis.py
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

## Results

### Comprehensive Immigration Discourse Analysis

**Dataset Analyzed**: 121,000 Reddit posts from 2024
**Immigration Content Identified**: 14,976 posts (12.38% of total content)
**Analysis Coverage**: Full year 2024 (January 1st - December 18th)

### Key Research Findings

- **Community Diversity**: Immigration discussions across 6,836+ unique subreddits
- **Temporal Patterns**: Complete year of immigration discourse trends
- **Content Quality**: Average post length of 467 characters (68.3 words)
- **Topic Modeling**: Multiple distinct immigration discourse clusters identified
- **Community Engagement**: Most active subreddit: r/AskReddit (127 immigration posts)

### Generated Research Assets

**Analysis Reports**:
- `results/enhanced_pipeline/analysis/comprehensive_immigration_analysis.json` - Complete quantitative analysis
- `results/enhanced_pipeline/analysis/IMMIGRATION_ANALYSIS_SUMMARY.md` - Executive summary
- `research_outputs/reports/RESEARCH_SUMMARY.md` - Research findings summary

**Visualizations**:
- `temporal_patterns.png` - Time-based distribution of immigration discussions
- `community_patterns.png` - Subreddit analysis and engagement patterns
- `content_analysis.png` - Text analysis and word clouds
- `topic_modeling.png` - LDA topic clusters
- `wordcloud.png` - Most frequent immigration-related terms

**Data Exports**:
- `research_outputs/data_exports/full_dataset.csv` - Complete processed dataset
- `research_outputs/data_exports/immigration_dataset.csv` - Immigration-specific content
- `research_outputs/data_exports/summary_statistics.json` - Statistical summaries

### Model Performance

- **BERT Classifier Accuracy**: 96.9%
- **Precision**: 93.9%
- **Recall**: 96.9%
- **F1-Score**: 95.4%

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

- **User Guide**: `docs/user_guide.md`
- **Developer Guide**: `docs/developer_guide.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support:

- Create an issue in the repository
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`

---

**Status**: Production Ready
