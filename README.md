# Reddit Visa Discourse Analysis

A comprehensive analysis pipeline for studying visa-related discourse patterns on Reddit, with focus on fear, Q&A, and fear-driven questions across different visa stages.

## Project Overview

This pipeline analyzes large-scale Reddit datasets to understand discourse patterns related to U.S. visa processes across stages (F1 → H1B+OPT → Green Card → Citizenship). It uses keyword-based pattern matching for classification and provides comprehensive analysis including temporal trends and visa stage distributions.

## Project Structure

```
bert_classifier/

 README.md
 requirements.txt
 .gitignore

 config/                        # Configuration files
    data_config.yaml
    model_config.yaml
    analysis_config.yaml
    credentials_template.py

 src/                           # Source code
    classification/            # Classification models
    preprocessing/             # Data processing
    analysis/                  # Statistical analysis
    visualization/             # Charts & dashboards
    ingestion/                 # Data acquisition
    utils/                     # Shared utilities

 data/                          # Data lifecycle
    raw/                       # Immutable raw data
    interim/                   # Intermediate processing
    processed/                 # Analysis-ready data
    external/                  # External reference data

 models/                        # ML assets
    checkpoints/               # Model weights
    tokenizer/                 # Preprocessing components

 results/                       # Outputs
    metrics/                   # Evaluation metrics
    visualizations/            # Charts & plots
    reports/                   # Human-readable outputs
    exports/                   # Shareable files

 notebooks/                     # Jupyter notebooks
 logs/                          # Process logs
 scripts/                       # Automation scripts
 docs/                          # Documentation
```

## Key Features

- **Keyword-Based Classification**: Pattern matching for 4 labels (fear, question, fear_driven_question, other)
- **Complete Dataset Analysis**: Processes 192+ million records without sampling
- **Visa Stage Classification**: Identifies discourse patterns across 6 visa stages
- **Parallel Processing**: Efficient multi-threaded analysis architecture
- **Production Ready**: Robust error handling and logging

## Quick Start

1. **Setup Environment**:
```bash
./scripts/setup_env.sh
```

2. **Run Full Pipeline**:
```bash
./scripts/run_pipeline.sh
```

3. **Run Individual Components**:
```bash
python src/preprocessing/data_sampling.py
python src/analysis/complete_no_sampling_analysis.py
python src/analysis/temporal_analysis.py
```

## Classification Labels

- **fear**: Posts expressing fear, anxiety, worry about visa processes
- **question**: Posts asking questions or seeking information
- **fear_driven_question**: Questions motivated by fear or anxiety
- **other**: Posts that don't fit the above categories

## Visa Stages Analyzed

- **Student Visa**: F1, CPT, OPT, STEM OPT
- **Work Visa**: H1B, Employer Sponsorship
- **Permanent Residency**: PERM, I-140, Green Card
- **Citizenship**: Naturalization process
- **General Immigration**: General immigration topics

## Current Status

- **2024 Analysis**: Complete dataset analysis (115M+ records)
- **2025 Analysis**: Complete dataset analysis (77M+ records)
- **Classification**: Keyword-based pattern matching implemented
- **Visualization**: Analysis results available

## Configuration

Edit `config/` files to customize:
- Model parameters (`config/model_config.yaml`)
- Data sources (`config/data_config.yaml`)
- Analysis settings (`config/analysis_config.yaml`)

## Documentation

- [Technical Documentation](docs/TECHNICAL_DOCS.md)
- [Architecture Overview](docs/ARCHITECTURE_DIAGRAM.png)
- [Sampling Methodology](docs/SAMPLING_METHODOLOGY.md)
- [Model Documentation](docs/MODEL_DOCUMENTATION.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Reddit API for data access
- AWS for data infrastructure
- Community contributors
