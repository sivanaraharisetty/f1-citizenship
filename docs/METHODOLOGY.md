# Complete Methodology: Large-Scale Reddit Visa Discourse Analysis

## Abstract

This document describes the complete methodology for analyzing large-scale Reddit data to understand discourse patterns related to U.S. visa processes. The analysis employs keyword-based pattern matching to identify fear expressions and question patterns across different visa stages (F1 → H1B+OPT → Green Card → Citizenship) in a complete dataset of 192+ million records from 2024-2025. The methodology encompasses data collection from Arctic Shift, ETL processing through AWS Glue, complete dataset analysis without sampling, and keyword-based classification. A BERT-based classifier framework has been implemented for future enhancement but was not used for the reported results.

---

## 1. Research Design and Objectives

### 1.1 Research Questions

1. What are the patterns of fear and anxiety expressed in visa-related discourse on Reddit?
2. How do discourse patterns vary across different visa stages?
3. What is the relationship between fear-driven questions and general questions in visa discourse?
4. How do discourse patterns vary across time periods (2024 vs 2025)?

### 1.2 Classification Framework

The research employs a keyword-based pattern matching approach to identify:

- **Fear**: Posts expressing fear, anxiety, or worry about visa processes (detected via keyword matching)
- **Question**: Posts asking questions or seeking information (detected via keyword matching)
- **Fear-Driven Question**: Questions that contain both fear keywords and question indicators (detected via combination of patterns)
- **Visa Stage**: Classification of posts by visa stage using keyword matching

### 1.3 Visa Stage Categorization

Discourse is analyzed across six visa stage categories using keyword matching:

1. **F1 (Student Visa)**: Keywords: 'f1', 'student visa', 'student', 'f-1', 'f1 visa', 'student status', 'f1 status'
2. **OPT (Optional Practical Training)**: Keywords: 'opt', 'optional practical training', 'stem opt', 'cpt', 'work authorization', 'opt status'
3. **H1B (Work Visa)**: Keywords: 'h1b', 'h-1b', 'work visa', 'h1-b', 'h1b visa', 'employment visa', 'h1b status'
4. **Green Card (Permanent Residency)**: Keywords: 'green card', 'greencard', 'permanent resident', 'gc', 'i-140', 'i-485', 'permanent residency'
5. **Citizenship**: Keywords: 'citizenship', 'naturalization', 'citizen', 'n-400', 'naturalized', 'citizen status'
6. **General Immigration**: Keywords: 'immigration', 'visa', 'immigrant', 'uscis', 'immigration law', 'immigration status'

---

## 2. Data Collection and Sources

### 2.1 Data Source

- **Platform**: Reddit (social media platform)
- **Data Type**: Posts and comments from visa-related subreddits
- **Storage**: Amazon S3 bucket
- **Format**: Parquet files (processed from JSON)
- **Time Period**: 2024-2025 (comprehensive analysis)
- **Total Volume**: 
  - **2024**: 115,289,778 records (complete dataset, no sampling)
  - **2025**: 77,073,526 records
  - **Combined Total**: 192+ million records
- **Data Provider**: Arctic Shift (Reddit data archive)
- **Data Availability**: Reddit data is available as monthly dumps on Arctic Shift; the torrent file was downloaded directly from Arctic Shift to obtain these dumps.

### 2.2 Data Collection Process

#### 2.2.1 Initial Data Acquisition

The raw Reddit data was obtained from Arctic Shift through the following process:

1. **Torrent Download**:
   - Data source: Arctic Shift Reddit monthly dumps
   - Tool: `aria2c` (multi-protocol download utility, version 1.37.0+)
   - Process: Monthly `.zst` files downloaded via torrent to an EC2 instance
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

Raw `.zst` files were decompressed to JSON format:

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
- **File Structure**: Year-based partitioning (`2024/`, `2025/`) with month sub-partitions
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

---

## 3. Data Analysis Methodology

### 3.1 Complete Dataset Analysis (No Sampling)

The analysis was performed on the complete dataset without any sampling to ensure full coverage and statistical validity:

- **2024**: All 115,289,778 records processed
- **2025**: All 77,073,526 records processed
- **Processing Method**: Parallel processing using `ThreadPoolExecutor` with 30 workers
- **File Processing**: All Parquet files in S3 bucket processed sequentially by month
- **No Sampling**: Every record in every file was analyzed

### 3.2 Keyword-Based Pattern Matching

The classification approach uses regex-based keyword pattern matching for detection rather than machine learning models. This method was chosen for computational efficiency when processing 192+ million records.

#### 3.2.1 Fear Detection

**Keywords Used**:
- `'afraid'`, `'scared'`, `'worried'`, `'anxious'`, `'panic'`, `'terrified'`, `'fear'`, `'concern'`, `'nervous'`

**Implementation**:
```python
fear_keywords = ['afraid', 'scared', 'worried', 'anxious', 'panic', 'terrified', 'fear', 'concern', 'nervous']
fear_pattern = '|'.join([re.escape(keyword) for keyword in fear_keywords])
df['has_fear'] = df[text_col].str.lower().str.contains(fear_pattern, na=False, regex=True)
```

**Method**:
- Case-insensitive matching
- Regex pattern matching
- Any occurrence of fear keywords flags the record as containing fear

**Limitations**:
- Simple keyword matching may miss nuanced expressions of fear
- May produce false positives (e.g., "not afraid" would still match)
- Does not capture context or sentiment intensity

#### 3.2.2 Question Detection

**Keywords Used**:
- `'?'`, `'how'`, `'what'`, `'when'`, `'where'`, `'why'`, `'can i'`, `'should i'`, `'help'`

**Implementation**:
```python
qa_keywords = ['\\?', 'how', 'what', 'when', 'where', 'why', 'can i', 'should i', 'help']
qa_pattern = '|'.join([re.escape(keyword) for keyword in qa_keywords])
df['is_question'] = df[text_col].str.lower().str.contains(qa_pattern, na=False, regex=True)
```

**Method**:
- Case-insensitive matching
- Regex pattern matching
- Question mark detection or question word detection

**Limitations**:
- May miss rhetorical questions or statements phrased as questions
- May produce false positives (e.g., "how-to" guides)
- Does not distinguish between genuine questions and statements

#### 3.2.3 Fear-Driven Question Detection

**Method**:
- Records flagged as containing both fear keywords AND question indicators
- Logical AND operation: `is_fear_driven_question = has_fear & is_question`

**Limitations**:
- Does not verify that fear and question are semantically related
- May flag coincidental co-occurrence

#### 3.2.4 Visa Stage Classification

**Method**: Keyword pattern matching for each visa stage category

**Keywords by Stage**:
- **F1**: `'f1'`, `'student visa'`, `'student'`, `'f-1'`, `'f1 visa'`, `'student status'`, `'f1 status'`
- **OPT**: `'opt'`, `'optional practical training'`, `'stem opt'`, `'cpt'`, `'work authorization'`, `'opt status'`
- **H1B**: `'h1b'`, `'h-1b'`, `'work visa'`, `'h1-b'`, `'h1b visa'`, `'employment visa'`, `'h1b status'`
- **Green Card**: `'green card'`, `'greencard'`, `'permanent resident'`, `'gc'`, `'i-140'`, `'i-485'`, `'permanent residency'`
- **Citizenship**: `'citizenship'`, `'naturalization'`, `'citizen'`, `'n-400'`, `'naturalized'`, `'citizen status'`
- **General Immigration**: `'immigration'`, `'visa'`, `'immigrant'`, `'uscis'`, `'immigration law'`, `'immigration status'`

**Implementation**:
```python
stages = {
    'F1': ['f1', 'student visa', 'student', 'f-1', 'f1 visa', 'student status', 'f1 status'],
    'OPT': ['opt', 'optional practical training', 'stem opt', 'cpt', 'work authorization', 'opt status'],
    # ... other stages
}
for stage, keywords in stages.items():
    pattern = '|'.join([re.escape(keyword) for keyword in keywords])
    count = df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
```

**Limitations**:
- Multiple visa stages may match a single post (not mutually exclusive)
- May miss posts that discuss visa stages without using exact keywords
- No validation of actual visa stage accuracy

### 3.3 Text Column Detection

The analysis automatically detects text columns by searching for columns containing keywords:
- `'text'`, `'body'`, `'content'`, `'comment'`, `'post'`

The first matching column is used for analysis. If multiple text columns exist, only the first is processed.

### 3.4 Processing Pipeline

#### 3.4.1 File Processing

1. **File Discovery**: List all Parquet files in S3 bucket organized by year and month
2. **Parallel Processing**: Process files using `ThreadPoolExecutor` with 30 workers
3. **Per-File Analysis**:
   - Load complete Parquet file into memory
   - Detect text column
   - Apply keyword pattern matching
   - Calculate statistics (fear rate, Q&A rate, stage counts)
   - Return analysis results
4. **Error Handling**: Failed files are logged but do not stop processing

#### 3.4.2 Aggregation

1. **Monthly Aggregation**: Sum all metrics from files within each month
2. **Yearly Aggregation**: Sum all monthly metrics for each year
3. **Cross-Year Comparison**: Compare 2024 vs 2025 patterns

### 3.5 Computational Efficiency

- **Parallel Processing**: 30 concurrent workers for file processing
- **Memory Efficiency**: Process one file at a time per worker, avoiding full dataset loading
- **No Sampling**: All records processed to ensure complete coverage
- **Processing Time**: Efficient regex matching enables rapid processing of large datasets

---

## 4. BERT Classifier Framework (Future Enhancement)

### 4.1 Implementation Status

A BERT-based multi-label classifier framework has been implemented (`src/classification/bert_classifier.py`) but was **not used** for the reported results. This framework is intended for future enhancement and provides more nuanced classification than keyword matching.

### 4.2 Framework Components

**Implemented Components**:
- `RedditDataset`: Custom PyTorch dataset for text classification
- `BERTMultiLabelClassifier`: BERT-based neural network architecture
- `RedditBERTClassifier`: Main classifier class with training and evaluation methods
- Training pipeline with early stopping and model checkpointing
- Evaluation metrics calculation

**Framework Capabilities**:
- Multi-label classification (fear, question, fear_driven_question, other)
- Customizable BERT models (bert-base-uncased, DistilBERT, RoBERTa)
- Training with validation and test splits
- Model evaluation with precision, recall, F1-score
- Model saving and loading

### 4.3 Requirements for BERT Usage

To use BERT classification for future results:
1. **Annotation**: Manually annotate training data (1000+ records recommended)
2. **Training**: Train BERT model on annotated data
3. **Evaluation**: Validate model performance on held-out test set
4. **Inference**: Run trained model on complete dataset
5. **Comparison**: Compare BERT results with keyword-based results

### 4.4 Why Keyword Matching Was Used

Keyword matching was chosen for the initial analysis because:
1. **Scalability**: Process 192+ million records efficiently
2. **Transparency**: Simple, interpretable classification rules
3. **Speed**: Fast processing without GPU requirements
4. **Baseline**: Provides baseline results for comparison with future BERT models
5. **No Training Data**: Does not require annotated training data

---

## 5. Results and Findings

### 5.1 2024 Analysis Results (Complete Dataset)

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

### 5.2 2025 Analysis Results (Complete Dataset)

- **Total Records Analyzed**: 77,073,526
- **Files Processed**: Complete dataset
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

### 5.3 Key Findings

1. **Fear Expression Prevalence**: Approximately 18.9% of visa-related discourse contains expressions of fear or anxiety, indicating significant emotional distress in the immigration process.

2. **Question Density**: Over 41% of discourse involves questions, highlighting the information-seeking nature of visa-related discussions.

3. **Visa Stage Focus**: OPT-related discussions dominate the discourse (35.1% in 2024, 31.2% in 2025), followed by F1 student visa discussions (12.6% in 2024, increasing to 24.8% in 2025). This shift indicates growing student visa concerns in 2025.

4. **Temporal Coverage**: Analysis covers complete years (2024: 12 months, 2025: partial year with comprehensive coverage).

5. **Dataset Scale**: Combined dataset of 192+ million records provides robust statistical power for analysis.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Keyword-Based Classification**:
   - Simple pattern matching may miss nuanced expressions
   - No context understanding
   - Potential false positives and negatives
   - No confidence scores or uncertainty quantification

2. **Visa Stage Classification**:
   - Multiple stages may match single posts (not mutually exclusive)
   - May miss posts discussing visa stages without keyword matches
   - No validation of classification accuracy

3. **Fear Detection**:
   - Does not capture intensity or context
   - May flag non-fearful uses of fear words (e.g., "not afraid")
   - Does not distinguish between different types of fear

4. **Question Detection**:
   - May miss rhetorical questions or indirect questions
   - May flag non-question uses of question words
   - Does not distinguish question types or urgency

5. **No Temporal Analysis**: Current analysis does not examine temporal trends or policy impacts

6. **No Sentiment Analysis**: No sentiment scoring or emotional intensity measurement

### 6.2 Future Enhancements

1. **BERT Classification**:
   - Train BERT model on annotated data
   - Replace keyword matching with BERT predictions
   - Compare BERT results with keyword-based results
   - Provide confidence scores for predictions

2. **Advanced NLP**:
   - Named entity recognition for visa types
   - Sentiment analysis for emotional intensity
   - Topic modeling for thematic analysis
   - Question type classification

3. **Temporal Analysis**:
   - Pre/post policy change analysis
   - Seasonal pattern detection
   - Trend analysis over time
   - Anomaly detection

4. **Validation**:
   - Manual validation of keyword-based classifications
   - Inter-annotator agreement studies
   - Cross-validation with external data sources

5. **Scalability Improvements**:
   - Distributed processing for even larger datasets
   - Incremental processing for new data
   - Real-time analysis capabilities

---

## 7. Technical Stack and Tools

### 7.1 Programming Language

- **Python**: 3.8+ (primary programming language)
- **Python Standard Library**: 
  - `json`, `logging`, `datetime`, `pathlib`, `typing`, `re`, `collections`, `concurrent.futures`

### 7.2 Data Processing

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **PyArrow**: Parquet file I/O
- **boto3**: AWS SDK for Python (S3 access)
- **s3fs**: S3 filesystem interface

### 7.3 Pattern Matching

- **re (regex)**: Regular expression pattern matching
- **String operations**: Case-insensitive matching, pattern search

### 7.4 Parallel Processing

- **concurrent.futures**: Parallel execution
  - `ThreadPoolExecutor`: Multi-threaded execution
  - `as_completed`: Future completion handling
- **Threading**: Multi-threaded file processing

### 7.5 Cloud Infrastructure

- **AWS S3**: Data storage
- **AWS Glue**: ETL processing
- **AWS Athena**: SQL-based analytics (planned)
- **EC2**: Compute infrastructure for data download

### 7.6 Data Storage Formats

- **Parquet**: Columnar storage format (via PyArrow)
- **JSON**: Configuration and results storage
- **Snappy Compression**: Optimized Parquet compression

### 7.7 Development Tools

- **Git**: Version control
- **Logging**: Python logging module for process tracking
- **Pathlib**: File path management

### 7.8 BERT Framework (Future)

- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: Pre-trained BERT models
- **scikit-learn**: Machine learning utilities
- **CUDA**: GPU acceleration (optional)

---

## 8. Reproducibility and Validation

### 8.1 Reproducibility

- **Random Seeds**: Fixed seed (42) for any random operations
- **Deterministic Processing**: Sequential file processing ensures reproducible results
- **Code Versioning**: Git version control for code tracking
- **Configuration Files**: Centralized configuration management

### 8.2 Data Validation

- **File Count Validation**: Verify expected number of files processed
- **Record Count Validation**: Verify expected number of records analyzed
- **Error Logging**: Comprehensive error logging for failed files
- **Statistics Tracking**: Track processing statistics (files processed, records analyzed, bytes processed)

### 8.3 Results Validation

- **Cross-Validation**: Compare results across months and years
- **Statistical Consistency**: Verify statistical consistency of results
- **Output Verification**: JSON output validation and structure verification

---

## 9. Data Privacy and Ethics

### 9.1 Data Privacy

- **Public Data**: Only publicly available Reddit data analyzed
- **No Personal Information**: Analysis focuses on aggregate patterns, not individual users
- **Anonymization**: User identifiers not included in analysis results

### 9.2 Research Ethics

- **Reddit Terms of Service**: Compliance with Reddit's Terms of Service
- **Data Minimization**: Only necessary data collected and analyzed
- **Responsible Use**: Analysis focused on understanding patterns, not identifying individuals

---

## 10. Conclusion

This methodology document describes a large-scale analysis of Reddit visa discourse using keyword-based pattern matching on a complete dataset of 192+ million records. The analysis provides foundational insights into fear expressions, question patterns, and visa stage distributions across 2024-2025. While keyword matching has limitations, it provides a scalable baseline for understanding large-scale discourse patterns. Future enhancements with BERT-based classification will provide more nuanced understanding while maintaining computational efficiency.

**Document Version**: 2.0  
**Last Updated**: October 2025  
**Author**: BERT Classifier Research Team  
**Status**: Complete and Current - Based on Actual Implementation
