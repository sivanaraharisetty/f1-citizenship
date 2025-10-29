# Complete Methodology: BERT-Based Multi-Label Classification for Reddit Visa Discourse Analysis

## Abstract

This document describes the complete methodology for analyzing large-scale Reddit data to understand discourse patterns related to U.S. visa processes. The pipeline employs BERT-based multi-label classification to identify fear, questions, and fear-driven questions across different visa stages (F1 → H1B+OPT → Green Card → Citizenship). The methodology encompasses data collection, stratified sampling, preprocessing, annotation, model training, evaluation, and temporal analysis.

---

## 1. Research Design and Objectives

### 1.1 Research Questions

1. What are the patterns of fear and anxiety expressed in visa-related discourse on Reddit?
2. How do discourse patterns vary across different visa stages?
3. What is the relationship between fear-driven questions and general questions in visa discourse?
4. How do policy changes affect discourse patterns over time?
5. Can BERT-based models effectively classify complex emotional and informational content in visa discourse?

### 1.2 Classification Framework

The research employs a multi-label classification framework with four mutually non-exclusive labels:

- **Fear**: Posts expressing fear, anxiety, or worry about visa processes
- **Question**: Posts asking questions or seeking information
- **Fear-Driven Question**: Questions specifically motivated by fear or anxiety
- **Other**: Posts that do not fit the above categories

### 1.3 Visa Stage Categorization

Discourse is analyzed across five visa stage categories:

1. **Student Visa (F1)**: F1, CPT, OPT, STEM OPT
2. **Work Visa**: H1B, Employer Sponsorship
3. **Permanent Residency**: PERM, I-140, Green Card
4. **Citizenship**: Naturalization process
5. **General Immigration**: General immigration topics

---

## 2. Data Collection and Sources

### 2.1 Data Source

- **Platform**: Reddit (social media platform)
- **Data Type**: Posts and comments from visa-related subreddits
- **Storage**: Amazon S3 bucket
- **Format**: JSON and Parquet files
- **Time Period**: 2024-2025 (comprehensive analysis)
- **Total Volume**: 
  - **2024**: 115,289,778 records (complete dataset, no sampling)
  - **2025**: 77,073,526 records
  - **Combined Total**: 192+ million records
- **Data Provider**: Arctic Shift (Reddit data archive)

### 2.2 Data Collection Process

#### 2.2.1 Initial Data Acquisition

The raw Reddit data is obtained from Arctic Shift, a public Reddit data archive, through the following process:

1. **Torrent Download**:
   - Data source: Arctic Shift Reddit torrent files
   - Tool: `aria2c` (multi-protocol download utility, version 1.37.0+)
   - Process: Files downloaded to EC2 instance in compressed format
   - Command: `aria2c ~/reddit.torrent --dir=~/downloads/reddit`
   - Background execution: `nohup aria2c ~/reddit.torrent --dir=~/downloads/reddit > ~/aria2.log 2>&1 &`

2. **File Format**:
   - Initial format: `.zst` (Zstandard compressed files)
   - Monthly structure: 
     - Posts: `RS_YYYY-MM.zst` (e.g., `RS_2024-01.zst`)
     - Comments: `RC_YYYY-MM.zst` (e.g., `RC_2024-01.zst`)

3. **Data Volume**:
   - **Posts**: ~150-200 GB per monthly file
   - **Comments**: ~400-500 GB per monthly file
   - **Total Monthly Input**: ~600-700 GB
   - **Files per Month**: 2 files (1 posts + 1 comments)

#### 2.2.2 Decompression Process

Raw `.zst` files are decompressed to JSON format:

1. **Decompression Tool**: `unzstd` (Zstandard decompression utility)
2. **Process**:
   - Command: `unzstd -c input.zst > output.json`
   - Batch processing: Decompressed in background with logging
   - Directory structure: `~/reddit_data/zstfiles/` and `~/reddit_data/jsonfiles/`
   - Example: `nohup unzstd -c ~/reddit_data/zstfiles/RS_2024-01.zst > ~/reddit_data/jsonfiles/RS_2024-01.json 2> ~/reddit_data/nohup.log &`
3. **Validation**: File size monitoring and content verification via log files

### 2.3 ETL Pipeline Architecture

The data processing pipeline follows a four-zone architecture designed for efficient data transformation and analytics:

#### 2.3.1 Zone Architecture Overview

**Raw Zone** (Landing Zone):
- **Purpose**: Immutable landing zone for incoming JSON files
- **Input**: Raw JSON files from decompression
- **Process**: No transformation applied
- **Output**: Raw JSON files for archive and replay capability
- **Characteristics**:
  - Full data preservation
  - Very large files (high storage costs)
  - Challenge: Storage scalability

**Staging Zone** (Initial ETL):
- **Purpose**: Initial transformation and normalization
- **Input**: Raw JSON files from Raw zone
- **Process**:
  - Flatten nested JSON structures
  - Partition by year and month
  - Convert timestamps (`created_utc` → `created_at`)
  - Write to Parquet format
- **Output**: Clean, partitioned Parquet datasets
- **Characteristics**:
  - Efficient columnar format (Parquet)
  - Improved query performance
  - Challenge: Schema validation and compute costs

**Refining Zone** (Data Quality):
- **Purpose**: Data cleansing, enrichment, and validation
- **Input**: Staged Parquet files
- **Process**:
  - Trim and normalize key fields (author, title, selftext, body)
  - Validate required fields (`id`, `body`, `created_utc`)
  - Flag deleted/removed posts and corrupt dates
  - Add `is_valid_record` flag
  - Drop exact duplicates
  - Convert arrays to JSON (link_flair_richtext, author_flair_richtext)
  - Add full JSON copy as `raw_record` column
- **Output**: High-quality, validated datasets
- **Characteristics**:
  - Improved data quality
  - Richer analytics capabilities
  - Challenge: Validation rules and scaling

**Publishing Zone** (Production Ready):
- **Purpose**: Curated, production-ready datasets for analytics
- **Input**: Refined Parquet files
- **Process**:
  - Ensure partition columns (`year`, `month`) present
  - Repartition by year and month
  - Filter to valid records (`is_valid_record = true`)
  - Write in Snappy Parquet format (optimized compression)
  - Update AWS Glue Catalog partitions
  - Manage table permissions
- **Output**: Analytic-ready, optimized datasets
- **Characteristics**:
  - Performance optimized for AWS Athena queries
  - Easy access via Glue Catalog
  - Challenge: Metadata consistency and security

#### 2.3.2 AWS Glue Processing

**Job Architecture**:
- **Approach**: Separate Glue jobs for posts and comments (recommended initial approach)
- **Rationale**:
  - Simpler job logic and easier maintenance
  - Isolated failure handling
  - Flexible resource allocation
  - Easier debugging
- **Alternative**: Combined job (for optimization after stabilization)
  - Pros: Reduced scheduling complexity, improved resource use
  - Cons: Increased complexity, higher memory/CPU requirements

**Job Configuration**:
- **Input Parameters**: Year and month (for incremental processing)
- **Processing Mode**: Monthly incremental data processing
- **Output Format**: Partitioned Parquet with year/month partitioning
- **Catalog Updates**: Automatic Glue Catalog partition updates
- **Validation**: Row count validation and success/failure logging

#### 2.3.3 Stage-Specific Processing Details

**Raw → Staging**:
- Reads raw JSON with full schema
- Applies schema validation
- Adds `created_at`, `year`, `month` columns
- Writes partitioned Parquet to staging path
- Uses overwrite mode
- No validation or deduplication at this stage

**Staging → Refining**:
- **Posts Processing**:
  - Trims and normalizes: `author`, `title`, `selftext`
  - Flags deleted/removed posts
  - Converts arrays to JSON (link_flair_richtext, author_flair_richtext)
  
- **Comments Processing**:
  - Validates required fields: `id`, `body`, `created_utc`
  - Trims and normalizes: `author`, `body`
  - Flags corrupt dates
  
- **Common Operations**:
  - Adds `is_valid_record` flag
  - Drops exact duplicates
  - Skips existing partitions using S3 partition check
  - Retains partition structure
  - Adds full JSON copy as `raw_record` column
  - Writes valid data in append mode

**Refining → Publishing**:
- Adds partition columns if missing
- Repartitions by year and month
- Filters to valid records (`is_valid_record = true`)
- Writes to published zone in Snappy Parquet format
- Uses overwrite mode
- Drops duplicates again
- Skips already published partitions using S3 checks

### 2.4 Data Access

Data is accessed via AWS S3 with the following configuration:

- **Bucket**: `[REDACTED]` (AWS S3 bucket name)
- **Prefix**: `[REDACTED]` (data prefix path)
- **File Structure**: Year-based partitioning (`2024/`, `2025/`)
- **Access Method**: boto3 and s3fs libraries
- **Authentication**: AWS credentials configured via environment variables
- **Query Interface**: AWS Athena for SQL-based analytics on published zone

### 2.5 Infrastructure and Tools

#### 2.5.1 Download Infrastructure
- **EC2 Instance**: Amazon EC2 for data download and initial processing
- **SSH Access**: Secure shell access for remote management
- **Storage**: EBS volumes for temporary data storage

#### 2.5.2 Data Processing Tools
- **aria2c**: Multi-protocol download utility for torrent files
  - Version: 1.37.0+
  - Installation: Compiled from source or via package manager
  - Features: Background download, progress tracking, resume capability
  
- **unzstd**: Zstandard decompression utility
  - Purpose: Decompress `.zst` files to JSON
  - Usage: `unzstd -c input.zst > output.json`
  - Process: Batch decompression with logging

- **AWS Glue**: Serverless ETL service
  - Purpose: Transform JSON to Parquet, apply schema, partition data
  - Features: Spark-based processing, automatic catalog updates
  - Monitoring: Job logs, validation, success/failure tracking

#### 2.5.3 Data Storage Formats
- **JSON**: Initial format from decompression (Raw zone)
- **Parquet**: Columnar format for efficient storage and querying (Staging/Refining/Publishing zones)
- **Compression**: Snappy compression for optimized Parquet files (Publishing zone)

### 2.6 Subreddit Coverage

Data is collected from multiple visa-related subreddits including:

- r/F1Visa, r/OPT, r/stemopt
- r/h1b, r/WorkVisas
- r/greencard, r/greencardprocess
- r/citizenship
- r/immigration, r/immigrationlaw, r/USCIS
- r/immigrationusa, r/USCISquestions

---

## 3. Data Sampling Strategy

### 3.1 Stratified Sampling Approach

Due to the large-scale nature of the dataset (192+ million records across 2024-2025), a stratified sampling strategy is employed to ensure statistical representativeness while maintaining computational feasibility. For comprehensive analyses, the complete dataset without sampling is processed: 115.3 million records for 2024 and 77.1 million records for 2025.

#### 3.1.1 Sampling Parameters

- **Base Sample Rate**: 1% per file
- **Minimum Samples per File**: 100 records
- **Stratification Variables**:
  - Subreddit
  - Month/Year
  - Content type (posts vs. comments)
  - Visa stage (when identifiable)

#### 3.1.2 Sampling Procedure

1. **File-Level Sampling**:
   - List all files in S3 bucket by year and month
   - Apply 1% random sampling to each file
   - Ensure minimum of 100 samples per file

2. **Stratification**:
   - Maintain proportional representation across subreddits
   - Preserve temporal distribution (monthly/yearly patterns)
   - Balance content types (posts vs. comments)

3. **Metadata Preservation**:
   - Retain original file source information
   - Preserve timestamp metadata
   - Maintain subreddit and author information

### 3.2 Oversampling for Rare Events

To address class imbalance in rare event detection:

- **Rare Event Threshold**: 5% frequency threshold
- **Oversampling Factor**: 20x for events below threshold
- **Application**: Applied to fear-driven questions and specific visa stage combinations

### 3.3 Sampling Validation

- **Random Seed**: Fixed seed (42) for reproducibility
- **Quality Checks**: Verify sample representativeness
- **Metadata Tracking**: Log sampling decisions and parameters

---

## 4. Data Preprocessing and Cleaning

### 4.1 Text Normalization

#### 4.1.1 Reddit-Specific Formatting Removal

- Remove Reddit markdown formatting:
  - Bold text (`**text**`)
  - Italic text (`*text*`)
  - Code blocks (```code```)
  - Inline code (`code`)
  - Headers (`# Header`)
  - Lists (`- item`)
  - Quotes (`> quote`)

- Remove Reddit-specific elements:
  - User mentions (`u/username`)
  - Subreddit mentions (`r/subreddit`)
  - Reddit links (`[text](url)`)

#### 4.1.2 URL and Link Removal

- Remove HTTP/HTTPS URLs
- Remove Reddit permalink structures
- Preserve domain information when relevant

#### 4.1.3 Emoji and Special Character Handling

- Convert emojis to text descriptions
- Normalize Unicode characters
- Handle special punctuation and symbols
- Preserve meaningful punctuation

### 4.2 Text Cleaning Pipeline

#### 4.2.1 Basic Cleaning Steps

1. **Whitespace Normalization**:
   - Remove multiple spaces
   - Normalize line breaks
   - Trim leading/trailing whitespace

2. **Case Normalization**:
   - Convert to lowercase for BERT processing
   - Preserve case for domain-specific terms (e.g., "H1B", "F1")

3. **Character Encoding**:
   - Normalize Unicode to UTF-8
   - Handle encoding errors gracefully
   - Preserve accented characters

#### 4.2.2 Language Processing

- **Tokenization**: NLTK word tokenization
- **Stopword Removal**: English stopwords from NLTK
- **Lemmatization**: WordNet lemmatizer for root forms
- **Part-of-Speech Tagging**: spaCy English model (when available)

### 4.3 Quality Filtering

#### 4.3.1 Length Filtering

- **Minimum Length**: 10 characters
- **Maximum Length**: 512 tokens (BERT limit)
- **Truncation**: Applied at token level for BERT processing

#### 4.3.2 Content Filtering

- Remove empty or whitespace-only texts
- Filter out deleted/removed posts
- Remove posts with insufficient information

#### 4.3.3 Duplicate Detection

- Exact duplicate removal
- Near-duplicate detection using text similarity
- Preserve temporal information when duplicates found

### 4.4 Feature Extraction

#### 4.4.1 Temporal Features

- Year, month, day, weekday
- Hour of day
- Week number, quarter
- Time bins for temporal analysis

#### 4.4.2 Metadata Features

- Subreddit name
- Post type (post vs. comment)
- Author information (when available)
- Score and engagement metrics

#### 4.4.3 Visa Stage Identification

- Keyword-based identification
- Subreddit-based classification
- Manual validation for ambiguous cases

---

## 5. Annotation System

### 5.1 Annotation Strategy

#### 5.1.1 Multi-Label Annotation

Each record can have multiple labels simultaneously:
- A post can be both "fear" and "question"
- "Fear-driven question" requires both "fear" and "question"
- "Other" is mutually exclusive with other labels

#### 5.1.2 Annotation Sample Size

- **Initial Sample**: 1,000 records for manual annotation
- **Sampling Strategy**: Stratified across subreddits and visa stages
- **Quality Assurance**: Inter-annotator agreement checks

### 5.2 Annotation Methods

#### 5.2.1 Manual Annotation

- **Interface**: Streamlit interactive web application
- **Annotator Training**: Guidelines and examples provided
- **Validation**: Multiple annotators for subset of data
- **Consensus**: Discussion for disagreements

#### 5.2.2 Semi-Automated Annotation

- **Keyword-Based Pre-annotation**:
  - Fear keywords: "worried", "scared", "anxious", "fear", "panic"
  - Question keywords: "?", "how", "what", "why", "when"
  - Question phrases: "can someone help", "what should I do"

- **Rule-Based Labeling**:
  - Regex patterns for common structures
  - Sentiment analysis for fear detection
  - Question mark detection for question classification

- **Human Review**: All automated annotations reviewed manually

### 5.3 Annotation Guidelines

#### 5.3.1 Fear Label

**Criteria**:
- Explicit expressions of worry, anxiety, or fear
- Uncertainty about visa status or process
- Concerns about delays or denials
- Emotional distress related to immigration

**Examples**:
- "I'm really worried about my H1B application"
- "Panicking about my visa interview"
- "Can't sleep because of visa uncertainty"

#### 5.3.2 Question Label

**Criteria**:
- Explicit questions seeking information
- Requests for advice or guidance
- Information-seeking statements

**Examples**:
- "How long does visa processing take?"
- "What documents do I need?"
- "Can someone explain the process?"

#### 5.3.3 Fear-Driven Question Label

**Criteria**:
- Questions motivated by fear or anxiety
- Urgent information-seeking due to concerns
- Combination of fear expression and question

**Examples**:
- "I'm panicking - what should I do if my visa is denied?"
- "Worried about my status, can someone help?"
- "Anxious about interview - what to expect?"

#### 5.3.4 Other Label

**Criteria**:
- General immigration discussions
- Information sharing without questions
- Experiences and stories
- Does not fit other categories

### 5.4 Annotation Quality Control

- **Inter-Annotator Agreement**: Cohen's Kappa calculated
- **Review Process**: Regular review of annotation quality
- **Guideline Updates**: Refinement based on edge cases
- **Validation Set**: Held-out validation set for quality checks

---

## 6. Model Architecture

### 6.1 BERT-Based Multi-Label Classifier

#### 6.1.1 Base Model Architecture

- **Base Model**: BERT-base-uncased (110M parameters)
- **Alternative Models**: DistilBERT, RoBERTa (configurable)
- **Input Format**: Tokenized text with special tokens
- **Maximum Sequence Length**: 512 tokens
- **Hidden Size**: 768 dimensions

#### 6.1.2 Model Components

1. **BERT Encoder**:
   - Pre-trained transformer encoder
   - 12 transformer layers
   - 12 attention heads per layer
   - Contextualized word representations

2. **Pooling Layer**:
   - CLS token pooling
   - Average pooling (alternative)
   - Max pooling (alternative)

3. **Classification Head**:
   - Dropout layer (0.1 dropout rate)
   - Linear layer: 768 → 4 (number of labels)
   - Sigmoid activation for multi-label output

#### 6.1.3 Multi-Label Classification Design

- **Output Format**: 4 independent binary classifiers
- **Activation Function**: Sigmoid for each label
- **Threshold**: 0.5 (configurable per label)
- **Loss Function**: Binary Cross-Entropy Loss

### 6.2 Dataset Class

#### 6.2.1 Custom Dataset Implementation

- **Text Tokenization**: BERT tokenizer with truncation/padding
- **Label Encoding**: Multi-hot binary encoding
- **Batch Processing**: Dynamic batching with padding
- **Data Augmentation**: Not applied (preserve original text)

#### 6.2.2 Data Loading

- **Batch Size**: 16 (configurable)
- **Shuffle**: Enabled for training
- **Num Workers**: 4 (for parallel loading)
- **Pin Memory**: Enabled for GPU acceleration

---

## 7. Training Methodology

### 7.1 Training Configuration

#### 7.1.1 Hyperparameters

- **Learning Rate**: 2e-5 (BERT standard)
- **Batch Size**: 16
- **Number of Epochs**: 3
- **Warmup Steps**: 500
- **Weight Decay**: 0.01
- **Gradient Accumulation Steps**: 1
- **Max Gradient Norm**: 1.0

#### 7.1.2 Optimizer

- **Optimizer**: AdamW
- **Beta1**: 0.9
- **Beta2**: 0.999
- **Epsilon**: 1e-8
- **Learning Rate Schedule**: Linear with warmup

#### 7.1.3 Regularization

- **Dropout**: 0.1 in classification head
- **Weight Decay**: 0.01
- **Early Stopping**: Based on validation loss (patience: 3 epochs)

### 7.2 Data Splitting

#### 7.2.1 Train-Validation-Test Split

- **Training Set**: 70% of annotated data
- **Validation Set**: 15% of annotated data
- **Test Set**: 15% of annotated data
- **Splitting Strategy**: Stratified by labels and visa stages
- **Random Seed**: 42 for reproducibility

#### 7.2.2 Cross-Validation

- **K-Fold**: 5-fold cross-validation (optional)
- **Stratification**: Maintains label distribution
- **Purpose**: Model selection and hyperparameter tuning

### 7.3 Training Procedure

#### 7.3.1 Training Loop

1. **Forward Pass**:
   - Tokenize input text
   - Pass through BERT encoder
   - Apply classification head
   - Compute predictions

2. **Loss Computation**:
   - Binary cross-entropy loss per label
   - Average across labels
   - Weighted loss for class imbalance (optional)

3. **Backward Pass**:
   - Compute gradients
   - Apply gradient clipping
   - Update model parameters

4. **Validation**:
   - Evaluate on validation set
   - Compute metrics
   - Save best model

#### 7.3.2 Model Checkpointing

- **Checkpoint Frequency**: End of each epoch
- **Best Model Selection**: Based on validation F1-score
- **Model Saving**: Full model state and tokenizer
- **Checkpoint Location**: `models/checkpoints/`

### 7.4 Hardware and Software

#### 7.4.1 Hardware

- **GPU**: CUDA-compatible GPU (when available)
- **CPU**: Multi-core CPU for parallel processing
- **Memory**: Sufficient RAM for batch processing

#### 7.4.2 Software

- **Deep Learning Framework**: PyTorch
- **Transformers Library**: Hugging Face Transformers
- **Other Libraries**: NumPy, Pandas, scikit-learn

---

## 8. Evaluation Metrics

### 8.1 Multi-Label Classification Metrics

#### 8.1.1 Per-Label Metrics

For each label (fear, question, fear_driven_question, other):

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Support**: Number of true instances per label

#### 8.1.2 Overall Metrics

- **Macro-Averaged F1**: Average F1 across all labels
- **Micro-Averaged F1**: F1 calculated globally
- **Weighted F1**: F1 weighted by label support
- **Exact Match Ratio**: Percentage of perfectly predicted samples

#### 8.1.3 Threshold Optimization

- **ROC Curves**: Per-label ROC curves
- **Precision-Recall Curves**: Per-label PR curves
- **Optimal Threshold**: Selected to maximize F1-score
- **Threshold Range**: 0.3 to 0.7 with 0.05 increments

### 8.2 Confusion Matrices

- **Per-Label Confusion Matrix**: TP, FP, TN, FN for each label
- **Multi-Label Confusion Matrix**: Comprehensive confusion matrix
- **Visualization**: Heatmaps for clarity

### 8.3 Error Analysis

#### 8.3.1 Error Categories

- **False Positives**: Incorrectly predicted labels
- **False Negatives**: Missed true labels
- **Label Confusion**: Confusion between similar labels

#### 8.3.2 Error Analysis Procedure

1. Identify common error patterns
2. Analyze text characteristics of errors
3. Investigate visa stage-specific errors
4. Review temporal patterns in errors

### 8.4 Statistical Significance Testing

- **Bootstrap Resampling**: 1000 iterations
- **Confidence Intervals**: 95% CI for metrics
- **Significance Testing**: Compare model variations

---

## 9. Temporal Analysis Methodology

### 9.1 Temporal Feature Extraction

#### 9.1.1 Time-Based Features

- **Granularity**: Year, month, week, day, hour
- **Cyclical Encoding**: Sin/cos encoding for cyclical features
- **Time Bins**: Equal-width bins for trend analysis

#### 9.1.2 Temporal Aggregation

- **Daily Aggregation**: Counts and rates per day
- **Weekly Aggregation**: Weekly averages and trends
- **Monthly Aggregation**: Monthly patterns and seasonality
- **Yearly Aggregation**: Year-over-year comparisons

### 9.2 Policy Event Analysis

#### 9.2.1 Policy Events Identified

1. **Trump Travel Ban**: January 27, 2017
2. **H1B Suspension**: June 22, 2020
3. **Biden Immigration Reform**: January 20, 2021
4. **DACA Renewal**: January 20, 2021
5. **H1B Cap Increase**: December 6, 2021

#### 9.2.2 Pre/Post Analysis

- **Analysis Windows**:
  - Short-term: 30 days before/after
  - Medium-term: 90 days before/after
  - Long-term: 180 days before/after

- **Statistical Tests**:
  - Mann-Whitney U test for pre/post comparison
  - Chi-square test for categorical changes
  - Regression analysis for trend changes

### 9.3 Trend Analysis

#### 9.3.1 Time Series Analysis

- **Trend Detection**: Linear regression on time series
- **Seasonality Detection**: Fourier analysis
- **Anomaly Detection**: Statistical outlier detection

#### 9.3.2 Temporal Pattern Identification

- **Daily Patterns**: Hourly variation in discourse
- **Weekly Patterns**: Day-of-week effects
- **Monthly Patterns**: Seasonal variations
- **Yearly Patterns**: Long-term trends

### 9.4 Cohort Analysis

- **Visa Stage Cohorts**: Compare patterns across visa stages
- **Time-Based Cohorts**: Compare users by registration period
- **Event-Based Cohorts**: Compare users affected by policy changes

---

## 10. Validation and Reproducibility

### 10.1 Reproducibility Measures

#### 10.1.1 Random Seed Control

- **Random Seed**: Fixed seed (42) for all random operations
- **NumPy**: `np.random.seed(42)`
- **Python Random**: `random.seed(42)`
- **PyTorch**: `torch.manual_seed(42)`

#### 10.1.2 Configuration Management

- **Configuration Files**: YAML-based configuration
- **Version Control**: Git for code versioning
- **Parameter Logging**: All hyperparameters logged

#### 10.1.3 Environment Management

- **Dependencies**: `requirements.txt` with versions
- **Python Version**: Python 3.8+
- **Library Versions**: Pinned versions for reproducibility

### 10.2 Validation Strategies

#### 10.2.1 Hold-Out Validation

- **Test Set**: Held-out test set (15% of data)
- **No Training Leakage**: Test set never used in training
- **Final Evaluation**: Metrics reported on test set only

#### 10.2.2 Cross-Validation

- **K-Fold CV**: 5-fold cross-validation for hyperparameter tuning
- **Stratified Splits**: Maintains label distribution
- **Model Selection**: Best model selected via CV

#### 10.2.3 Temporal Validation

- **Time-Based Splits**: Train on earlier data, test on later data
- **Purpose**: Evaluate temporal generalization
- **Challenges**: Address distribution shift over time

### 10.3 Documentation

#### 10.3.1 Code Documentation

- **Docstrings**: Comprehensive function documentation
- **Comments**: Inline comments for complex logic
- **Type Hints**: Type annotations for clarity

#### 10.3.2 Experiment Logging

- **Logging**: Comprehensive logging of all steps
- **Experiment Tracking**: Log all hyperparameters and results
- **Error Logging**: Detailed error logs for debugging

#### 10.3.3 Results Documentation

- **Metrics Reports**: Detailed metrics reports
- **Visualizations**: Charts and plots for analysis
- **Methodology Documentation**: This document

---

## 11. Analysis Pipeline

### 11.1 Complete Pipeline Steps

1. **Data Sampling** (Section 3)
   - Stratified sampling from S3
   - Oversampling for rare events
   - Metadata preservation

2. **Data Cleaning** (Section 4)
   - Text normalization
   - Quality filtering
   - Feature extraction

3. **Descriptive Analysis** (Section 4.4)
   - Keyword frequency analysis
   - Topic modeling
   - Temporal distribution analysis

4. **Annotation** (Section 5)
   - Manual annotation
   - Semi-automated annotation
   - Quality control

5. **Model Training** (Sections 6-7)
   - BERT model training
   - Hyperparameter tuning
   - Model validation

6. **Evaluation** (Section 8)
   - Metrics computation
   - Error analysis
   - Statistical testing

7. **Temporal Analysis** (Section 9)
   - Policy impact analysis
   - Trend detection
   - Cohort analysis

8. **Visualization** (Section 11.2)
   - Dashboard generation
   - Report creation
   - Export results

### 11.2 Visualization and Reporting

#### 11.2.1 Visualizations Generated

- **Label Distribution**: Bar charts and pie charts
- **Temporal Trends**: Time series plots
- **Confusion Matrices**: Heatmaps
- **ROC/PR Curves**: Performance curves
- **Word Clouds**: Keyword visualizations
- **Network Graphs**: Relationship visualizations

#### 11.2.2 Reports Generated

- **Executive Summary**: High-level findings
- **Detailed Analysis**: Comprehensive results
- **Methodology Report**: This document
- **Technical Documentation**: Implementation details

---

## 12. Ethical Considerations

### 12.1 Data Privacy

- **Anonymization**: User identifiers removed/hashed
- **Data Minimization**: Only necessary data collected
- **Access Control**: Secure access to sensitive data

### 12.2 Research Ethics

- **Public Data**: Only publicly available Reddit data
- **Consent**: Reddit's Terms of Service compliance
- **Bias Awareness**: Acknowledgment of potential biases

### 12.3 Responsible AI

- **Model Limitations**: Clear documentation of limitations
- **Bias Mitigation**: Efforts to address dataset biases
- **Transparency**: Open methodology and reproducible results

---

## 13. Limitations and Future Work

### 13.1 Limitations

- **Complete Dataset Analysis**: The full dataset (192+ million records) has been analyzed without sampling, providing comprehensive coverage. However, initial exploratory analyses used stratified sampling (1%) for computational efficiency.
- **Reddit Bias**: Reddit users may not represent all visa applicants; findings are specific to Reddit discourse patterns.
- **Temporal Scope**: Comprehensive analysis covers 2024 (complete year) and 2025 (partial year with full coverage available).
- **Language**: English-only analysis; non-English discourse is excluded.
- **Label Ambiguity**: Some posts may have ambiguous labels, requiring manual review for edge cases.
- **Keyword-Based Detection**: Initial fear and Q&A detection uses keyword matching; BERT classification provides more nuanced understanding.

### 13.2 Future Work

- **Expansion**: Larger samples and longer time periods
- **Multi-Language**: Support for other languages
- **Advanced Models**: Try newer transformer architectures
- **Real-Time Analysis**: Deploy for real-time monitoring
- **External Validation**: Validate findings with other data sources

## 14. Technical Stack and Tools

### 14.1 Programming Language and Core Framework

- **Python**: 3.8+ (primary programming language)
- **Python Standard Library**: 
  - `json`, `logging`, `datetime`, `pathlib`, `typing`, `re`, `collections`, `concurrent.futures`

### 14.2 Deep Learning and Machine Learning

#### 14.2.1 Deep Learning Framework
- **PyTorch**: Primary deep learning framework
  - `torch`: Core tensor operations
  - `torch.nn`: Neural network modules
  - `torch.utils.data`: Dataset and DataLoader classes
  - CUDA support for GPU acceleration

#### 14.2.2 Transformers Library
- **Hugging Face Transformers**: Pre-trained BERT models
  - `AutoTokenizer`: Tokenization for BERT models
  - `AutoModel`: BERT model architectures
  - `AutoConfig`: Model configuration
  - `TrainingArguments`: Training configuration
  - `Trainer`: Model training utilities
  - `EarlyStoppingCallback`: Early stopping mechanism
  - `get_linear_schedule_with_warmup`: Learning rate scheduling

#### 14.2.3 Scikit-learn
- **scikit-learn**: Machine learning utilities
  - `train_test_split`: Data splitting
  - `TfidfVectorizer`: Text vectorization
  - `LogisticRegression`: Baseline classification
  - `precision_recall_fscore_support`: Evaluation metrics
  - `confusion_matrix`: Confusion matrix calculation
  - `classification_report`: Classification metrics
  - `roc_auc_score`: ROC-AUC calculation
  - `average_precision_score`: Average precision
  - `LinearRegression`: Trend analysis
  - `StandardScaler`: Feature scaling

### 14.3 Data Processing and Analysis

#### 14.3.1 Data Manipulation
- **Pandas**: Data manipulation and analysis
  - DataFrames for structured data handling
  - Time series analysis
  - Data aggregation and grouping
  - File I/O (CSV, Parquet, JSON)

- **NumPy**: Numerical computing
  - Array operations
  - Mathematical functions
  - Statistical operations

#### 14.3.2 Statistical Analysis
- **SciPy**: Scientific computing
  - `stats`: Statistical functions (Mann-Whitney U test, etc.)
  - Statistical hypothesis testing

### 14.4 Natural Language Processing

#### 14.4.1 NLTK (Natural Language Toolkit)
- `nltk.corpus.stopwords`: English stopwords
- `nltk.tokenize.word_tokenize`: Word tokenization
- `nltk.stem.WordNetLemmatizer`: Word lemmatization
- Data downloads: `punkt`, `stopwords`, `wordnet`, `omw-1.4`

#### 14.4.2 spaCy
- **spaCy**: Advanced NLP library
  - `en_core_web_sm`: English language model
  - Part-of-speech tagging
  - Named entity recognition
  - Dependency parsing

#### 14.4.3 TextBlob
- **TextBlob**: Text processing library
  - Sentiment analysis
  - Text normalization

#### 14.4.4 Other NLP Tools
- **emoji**: Emoji handling and conversion
- **unicodedata**: Unicode normalization

### 14.5 Cloud Infrastructure and Data Storage

#### 14.5.1 AWS Services
- **Amazon S3**: Data storage
  - `boto3`: AWS SDK for Python
  - `s3fs`: S3 filesystem interface
  - Object storage for large datasets
  - Year-based partitioning (`2024/`, `2025/`)

#### 14.5.2 Data Formats
- **Parquet**: Columnar storage format (via PyArrow)
  - `pyarrow`: Apache Arrow Python bindings
  - `pyarrow.parquet`: Parquet file I/O
  - Efficient data serialization
- **JSON**: Configuration and results storage
- **CSV**: Data export and summaries

### 14.6 Visualization and Reporting

#### 14.6.1 Plotly
- **Plotly**: Interactive visualizations
  - `plotly.graph_objects`: Custom graphs
  - `plotly.express`: High-level plotting
  - `plotly.subplots`: Subplot creation
  - `plotly.figure_factory`: Advanced figures
  - `plotly.offline`: Offline plotting
  - Interactive HTML dashboards

#### 14.6.2 Matplotlib
- **Matplotlib**: Static visualization
  - `matplotlib.pyplot`: Plotting interface
  - Publication-quality figures

#### 14.6.3 Seaborn
- **Seaborn**: Statistical visualization
  - Statistical plotting
  - Heatmaps
  - Advanced plot styling

#### 14.6.4 Other Visualization Tools
- **WordCloud**: Word cloud generation
- **NetworkX**: Network graph visualization

### 14.7 Web Framework and User Interface

#### 14.7.1 Streamlit
- **Streamlit**: Interactive web applications
  - Annotation interface
  - Dashboard creation
  - User interaction components

#### 14.7.2 IPython/Jupyter (Optional)
- **ipywidgets**: Interactive widgets
- **IPython.display**: Display utilities

### 14.8 Parallel Processing and Performance

#### 14.8.1 Concurrent Processing
- **concurrent.futures**: Parallel execution
  - `ThreadPoolExecutor`: Multi-threaded execution
  - `as_completed`: Future completion handling
  - Parallel file processing

#### 14.8.2 Progress Tracking
- **tqdm**: Progress bars
  - Iteration progress tracking
  - Time estimation

### 14.9 Configuration and Environment Management

#### 14.9.1 Configuration Management
- **YAML**: Configuration files (`config/*.yaml`)
- **Python dataclasses**: Configuration classes
- **Environment Variables**: AWS credentials and settings configured via environment variables (not hardcoded)

#### 14.9.2 Version Control
- **Git**: Code version control
- **GitHub**: Repository hosting

### 14.10 Development and Deployment Tools

#### 14.10.1 Build and Automation
- **Makefile**: Build automation
- **Bash Scripts**: Automation scripts
  - `setup_env.sh`: Environment setup
  - `run_pipeline.sh`: Pipeline execution
  - `monitor.sh`: Process monitoring

#### 14.10.2 Logging and Monitoring
- **Python logging**: Comprehensive logging
- **Custom monitoring scripts**: Process status tracking

### 14.11 Data Validation and Quality

#### 14.11.1 Type Checking
- **Python typing**: Type hints for code clarity
  - `Dict`, `List`, `Tuple`, `Optional`, `Any`

#### 14.11.2 Error Handling
- **Python warnings**: Warning management
- **Exception handling**: Robust error handling

### 14.12 Model Architecture Specifications

#### 14.12.1 BERT Models
- **bert-base-uncased**: Primary model (110M parameters)
  - 12 transformer layers
  - 12 attention heads per layer
  - 768 hidden dimensions
  - 512 maximum sequence length

#### 14.12.2 Alternative Models (Configurable)
- **DistilBERT**: Lightweight alternative
- **RoBERTa**: Robustly optimized BERT

### 14.13 Hardware Requirements

#### 14.13.1 Compute Resources
- **GPU**: CUDA-compatible GPU (optional, recommended)
  - NVIDIA GPU with CUDA support
  - GPU memory: 8GB+ recommended
- **CPU**: Multi-core CPU for parallel processing
- **RAM**: Sufficient memory for batch processing (16GB+ recommended)

#### 14.13.2 Storage
- **Local Storage**: For intermediate data and models
- **Cloud Storage**: AWS S3 for raw data and results

### 14.14 Software Environment

#### 14.14.1 Operating System
- **macOS**: Development environment (Darwin 25.0.0)
- **Linux**: Production deployment (compatible)
- **Shell**: Zsh/Bash for automation

#### 14.14.2 Python Environment
- **Virtual Environment**: Python venv
- **Package Management**: pip
- **Dependencies**: Pinned versions for reproducibility

### 14.15 Third-Party APIs and Services

#### 14.15.1 Data Sources
- **Reddit API**: Social media data access
- **AWS S3**: Cloud data storage

### 14.16 Code Organization and Structure

#### 14.16.1 Project Structure
- **Modular Design**: Separate modules for each component
- **Configuration Files**: Centralized configuration
- **Scripts**: Automation and utility scripts
- **Documentation**: Comprehensive documentation

#### 14.16.2 Code Quality
- **Docstrings**: Comprehensive function documentation
- **Type Hints**: Type annotations throughout
- **Logging**: Structured logging across modules

---

## 15. References and Resources

### 15.1 Key Libraries

- Hugging Face Transformers: BERT implementation
- PyTorch: Deep learning framework
- scikit-learn: Machine learning utilities
- Pandas: Data manipulation
- NLTK: Natural language processing
- spaCy: Advanced NLP library
- Plotly: Interactive visualizations
- Matplotlib: Static visualizations
- Seaborn: Statistical visualizations

### 15.2 Data Sources

- Reddit API: Social media data
- AWS S3: Data storage

### 15.3 Model Resources

- BERT Paper: Devlin et al. (2019)
- Multi-label Classification: Literature on multi-label methods
- Hugging Face Model Hub: Pre-trained transformer models

---

## Appendix A: Configuration Parameters

See `config/config.py` and `config/*.yaml` files for complete configuration parameters.

## Appendix B: Implementation Details

See source code in `src/` directory for detailed implementation.

## Appendix C: Results and Findings

### C.1 Complete Dataset Analysis Results

#### C.1.1 2024 Analysis Results (Complete Dataset, No Sampling)

- **Total Records Analyzed**: 115,289,778
- **Files Processed**: Complete dataset across all 12 months
- **Fear Detection**:
  - Total Fear Cases: 21,793,076
  - Fear Rate: 18.9% (0.189)
- **Question Detection**:
  - Total Q&A Cases: 47,759,518
  - Q&A Rate: 41.4% (0.414)
- **Visa Stage Distribution**:
  - F1 (Student Visa): 14,502,378 records (12.6%)
  - OPT: 40,422,496 records (35.1%)
  - H1B: 206,204 records (0.2%)
  - Green Card: 7,208,835 records (6.3%)
  - Citizenship: 1,280,921 records (1.1%)
  - General Immigration: 1,086,146 records (0.9%)

#### C.1.2 2025 Analysis Results (Complete Dataset)

- **Total Records Analyzed**: 77,073,526
- **Files Processed**: 6,923 files (all successfully processed)
- **Fear Detection**:
  - Total Fear Cases: 12,160,505
  - Fear Rate: 15.8% (estimated, based on 2024 patterns)
- **Question Detection**:
  - Total Q&A Cases: 51,897,196
  - Q&A Rate: 67.4% (estimated)
- **Visa Stage Distribution**:
  - F1 (Student Visa): 19,131,575 records (24.8%)
  - OPT: 24,036,491 records (31.2%)
  - H1B: 154,421 records (0.2%)
  - Green Card: 4,235,052 records (5.5%)
  - Citizenship: 1,010,571 records (1.3%)
  - General Immigration: 812,636 records (1.1%)

#### C.1.3 Key Findings

1. **Fear Expression Prevalence**: Approximately 18.9% of visa-related discourse contains expressions of fear or anxiety, indicating significant emotional distress in the immigration process.

2. **Question Density**: Over 41% of discourse involves questions, highlighting the information-seeking nature of visa-related discussions.

3. **Visa Stage Focus**: OPT-related discussions dominate the discourse (35.1% in 2024, 31.2% in 2025), followed by F1 student visa discussions (12.6% in 2024, increasing to 24.8% in 2025). This shift indicates growing student visa concerns in 2025.

4. **Temporal Coverage**: Analysis covers complete years (2024: 12 months, 2025: partial year with comprehensive coverage).

5. **Dataset Scale**: Combined dataset of 192+ million records provides robust statistical power for analysis.

### C.2 Detailed Results Files

See `results/` directory for detailed analysis results and visualizations:
- `complete_2024_analysis_NO_SAMPLING.json`: Complete 2024 analysis
- `complete_2025_analysis.json`: Complete 2025 analysis
- Monthly breakdowns and stage-specific analyses
- Visualization files and summary reports

---

**Document Version**: 1.2  
**Last Updated**: October 2025  
**Author**: BERT Classifier Research Team  
**Status**: Complete and Current

