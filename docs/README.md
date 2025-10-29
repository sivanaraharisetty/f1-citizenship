# Reddit Visa Discourse Analysis Pipeline

A comprehensive analysis pipeline for studying visa-related discourse patterns on Reddit, with focus on fear, Q&A, and fear-driven questions across different visa stages.

## Overview

This pipeline analyzes large-scale Reddit datasets to understand discourse patterns related to U.S. visa processes across stages (F1 → H1B+OPT → Green Card → Citizenship). It uses keyword-based pattern matching for classification and provides comprehensive analysis including temporal trends and visa stage distributions.

## Features

- **Stratified Sampling**: 1% sampling with oversampling for rare events
- **Text Preprocessing**: Comprehensive cleaning, normalization, and tokenization
- **Keyword-Based Classification**: Pattern matching for multi-label classification
- **Temporal Analysis**: Pre/post policy change analysis and trend detection
- **Interactive Visualizations**: Comprehensive dashboards and reports
- **Evaluation Metrics**: Precision, recall, F1-score, confusion matrices

## Project Structure

```
Project_Reddit_Analysis/

 raw_data/                  # Original S3 data
 sampled_data/              # 1% sample + oversampled rare events
 cleaned_data/              # Preprocessed posts/comments
 descriptive_analysis/       # Keywords, topic clusters, distributions
 annotation/                # Labeled subset for classifier training
 classifier/                
    models/                # Saved classification models
    predictions/           # Predicted labels
    metrics/               # Precision, recall, F1, confusion matrix
 pre_post_analysis/         # Temporal and policy change analysis
 visualizations/            # Figures, charts, plots for paper
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd bert_classifier
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download additional models** (optional):
```bash
python -m spacy download en_core_web_sm
```

## Configuration

Edit `config.py` to set your parameters:

```python
# S3 Configuration
s3_bucket = "your-reddit-dataset-bucket"
s3_prefix = "reddit-data/"

# Sampling parameters
sample_rate = 0.01  # 1% sampling
min_samples_per_file = 100

# Classification model configuration
model_name = "keyword-based"  # Pattern matching approach
batch_size = 16
learning_rate = 2e-5
num_epochs = 3
```

## Usage

### 1. Complete Pipeline

Run the entire analysis pipeline:

```bash
python main_pipeline.py
```

### 2. Specific Steps

Run only specific steps:

```bash
# Data sampling and cleaning only
python main_pipeline.py --steps sampling cleaning

# Annotation and classification only
python main_pipeline.py --steps annotation classifier_training
```

### 3. Individual Modules

Run individual analysis modules:

```bash
# Data sampling
python data_sampling.py

# Data cleaning
python data_cleaning.py

# Descriptive analysis
python descriptive_analysis.py

# Annotation system
python annotation_system.py

# Classification training
python src/classification/bert_classifier.py

# Evaluation metrics
python evaluation_metrics.py

# Temporal analysis
python temporal_analysis.py

# Visualization tools
python visualization_tools.py
```

## Pipeline Steps

### 1. Data Sampling
- Stratified sampling (1% per file)
- Oversampling for rare events
- Metadata preservation

### 2. Data Cleaning
- Text normalization and preprocessing
- Duplicate removal
- Missing value handling
- Tokenization and lemmatization

### 3. Descriptive Analysis
- Keyword frequency analysis
- Topic modeling (LDA)
- Temporal distribution analysis
- Sentiment analysis

### 4. Annotation System
- Manual annotation interface
- Semi-automated batch annotation
- Label validation and quality control

### 5. Classification
- Keyword-based pattern matching
- Multi-label classification
- Prediction on full dataset

### 6. Evaluation Metrics
- Precision, recall, F1-score
- Confusion matrices
- ROC curves and PR curves
- Error analysis

### 7. Temporal Analysis
- Policy impact analysis
- Seasonal pattern detection
- Anomaly detection
- Trend analysis

### 8. Visualization
- Interactive dashboards
- Comprehensive reports
- Network visualizations
- Word clouds

## Visa Stages and Keywords

### Student Visa Stage (F1, CPT, OPT, STEM OPT)
- **Subreddits**: r/F1Visa, r/OPT, r/stemopt, r/immigration, r/visa
- **Keywords**: F1, CPT, OPT, STEM OPT, Visa interview, Work authorization, I-765

### Work Visa Stage (H1B, Employer Sponsorship)
- **Subreddits**: r/h1b, r/WorkVisas, r/immigrationlaw, r/immigrationattorney
- **Keywords**: H1B, H-1B, Employer sponsor, Job search visa issues, Visa denial

### Permanent Residency Stage (PERM, I-140, Green Card)
- **Subreddits**: r/greencard, r/greencardprocess, r/immigration, r/USCIS
- **Keywords**: I-140, PERM, Green Card, GC, Adjustment of status, Priority date

### Citizenship Stage
- **Subreddits**: r/citizenship, r/immigration, r/USCIS, r/immigrationlaw
- **Keywords**: Citizenship, Naturalization, Immigration reform, Travel ban

## Classification Labels

- **fear**: Posts expressing fear, anxiety, worry about visa processes
- **question**: Posts asking questions or seeking information
- **fear_driven_question**: Questions motivated by fear or anxiety
- **other**: Posts that don't fit the above categories

## Output Files

### Data Files
- `sampled_reddit_data.parquet`: Sampled dataset
- `cleaned_reddit_data.parquet`: Cleaned dataset
- `exported_annotations.parquet`: Annotated subset

### Model Files
- `classifier/models/`: Saved classification models
- `classifier/predictions/`: Model predictions
- `classifier/metrics/`: Evaluation metrics

### Analysis Results
- `descriptive_analysis_results.json`: Descriptive analysis
- `temporal_analysis_results.json`: Temporal analysis
- `comprehensive_evaluation.json`: Model evaluation

### Visualizations
- `comprehensive_dashboard.html`: Main dashboard
- `visualizations/`: All charts and plots

## Customization

### Adding New Labels
Edit `config.py` to add new classification labels:

```python
labels = ["fear", "question", "fear_driven_question", "other", "new_label"]
```

### Adding New Visa Stages
Add new visa stages to the configuration:

```python
visa_stages = {
    "new_stage": {
        "subreddits": ["subreddit1", "subreddit2"],
        "keywords": ["keyword1", "keyword2"]
    }
}
```

### Custom Model Configuration
Modify classification parameters:

```python
# Keyword patterns can be customized in config files
fear_keywords = ['afraid', 'scared', 'worried', ...]
qa_keywords = ['?', 'how', 'what', ...]
```

## Troubleshooting

### Common Issues

1. **S3 Connection Issues**
   - Verify AWS credentials
   - Check bucket permissions
   - Ensure bucket exists

2. **Memory Issues**
   - Process data in chunks
   - Use parallel processing
   - Optimize file reading

3. **Annotation Issues**
   - Check annotation guidelines
   - Validate label consistency
   - Review annotation quality

### Performance Optimization

1. **GPU Usage**
   - Ensure CUDA is available
   - Use appropriate batch sizes
   - Monitor GPU memory usage

2. **Data Processing**
   - Use parallel processing
   - Optimize data loading
   - Cache intermediate results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{reddit_visa_analysis,
  title={Reddit Visa Discourse Analysis Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/bert_classifier}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

## Acknowledgments

- Reddit API for data access
- AWS for data infrastructure
- Community contributors
