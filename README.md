# Reddit Visa Discourse Analysis

A comprehensive analysis pipeline for studying visa-related discourse patterns on Reddit, using keyword-based pattern matching to identify fear expressions, question patterns, and visa stage classifications across 192+ million records.

## Project Overview

This pipeline analyzes large-scale Reddit datasets to understand discourse patterns related to U.S. visa processes across stages (F1 → H1B+OPT → Green Card → Citizenship). It uses keyword-based pattern matching for classification and provides comprehensive analysis including temporal trends and visa stage distributions.

## Project Structure

```
bert_classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/                        # Configuration files
│   ├── data_config.yaml
│   ├── analysis_config.yaml
│   └── credentials_template.py
│
├── src/                           # Source code
│   ├── preprocessing/             # Data processing
│   ├── analysis/                  # Statistical analysis
│   ├── visualization/             # Charts & dashboards
│   ├── ingestion/                 # Data acquisition
│   └── utils/                     # Shared utilities
│
├── data/                          # Data lifecycle
│   ├── raw/                       # Immutable raw data
│   ├── interim/                   # Intermediate processing
│   ├── processed/                 # Analysis-ready data
│   └── external/                  # External reference data
│
├── results/                       # Outputs
│   ├── metrics/                   # Evaluation metrics
│   ├── visualizations/            # Charts & plots
│   ├── reports/                   # Human-readable outputs
│   └── exports/                   # Shareable files
│
├── notebooks/                     # Jupyter notebooks
├── logs/                          # Process logs
├── scripts/                       # Automation scripts
└── docs/                          # Documentation
```

## Key Features

- **Keyword-Based Classification**: Pattern matching for fear, questions, and visa stages
- **Complete Dataset Analysis**: Full 192+ million records processed without sampling
- **Temporal Analysis**: Policy impact analysis and trend detection
- **Interactive Visualizations**: Comprehensive dashboards and reports
- **Production Ready**: Robust error handling and logging

## Quick Start

1. **Setup Environment**:
```bash
./scripts/setup_env.sh
```

2. **Run Analysis**:
```bash
python src/analysis/complete_no_sampling_analysis.py
```

3. **Run Individual Components**:
```bash
python src/preprocessing/data_sampling.py
python src/analysis/temporal_analysis.py
python src/visualization/visualization_tools.py
```

## Classification Labels

- **fear**: Posts expressing fear, anxiety, worry about visa processes (detected via keyword matching)
- **question**: Posts asking questions or seeking information (detected via keyword matching)
- **fear_driven_question**: Questions that contain both fear keywords and question indicators
- **visa_stage**: Classification by visa stage using keyword matching

## Visa Stages Analyzed

- **Student Visa**: F1, CPT, OPT, STEM OPT
- **Work Visa**: H1B, Employer Sponsorship
- **Permanent Residency**: PERM, I-140, Green Card
- **Citizenship**: Naturalization process
- **General Immigration**: General immigration topics

## Current Status

- **2024 Analysis**: Complete dataset analysis (115,289,778 records)
- **2025 Analysis**: Complete dataset analysis (77,073,526 records)
- **Total Records**: 192,363,304 records analyzed
- **Visualization**: Interactive dashboards available

## Configuration

Edit `config/` files to customize:
- Data sources (`config/data_config.yaml`)
- Analysis settings (`config/analysis_config.yaml`)

Set environment variables for AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

## Documentation

- [Production Pipeline Guide](docs/PRODUCTION_PIPELINE_GUIDE.md)
- [Enhanced Sampling Guide](docs/ENHANCED_SAMPLING_GUIDE.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Reddit data via Arctic Shift
- AWS infrastructure for data processing
- Community contributors
