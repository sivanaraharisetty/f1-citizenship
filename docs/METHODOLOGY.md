# Technical Methodology: Large-Scale Reddit Visa Discourse Analysis

## Document Information

- **Version**: 2.1
- **Last Updated**: October 2025
- **Classification**: Technical Documentation
- **Status**: Production

---

## 1. Executive Summary

This document specifies the technical methodology for analyzing 192,363,304 records of Reddit discourse data to identify fear expressions, question patterns, and visa stage classifications using deterministic keyword-based pattern matching algorithms. The analysis pipeline processes complete datasets without sampling, utilizing parallel processing architectures and AWS cloud infrastructure.

**Key Specifications**:
- **Dataset**: 115,289,778 records (2024) + 77,073,526 records (2025)
- **Classification Method**: Regular expression pattern matching
- **Processing Architecture**: Multi-threaded parallel execution (30 workers)
- **Storage Format**: Parquet (Snappy compression)
- **Compute Infrastructure**: AWS S3, AWS Glue, EC2

---

## 2. Data Acquisition and Ingestion

### 2.1 Data Source Specification

| Attribute | Value |
|-----------|-------|
| **Source Platform** | Reddit (public API) |
| **Data Provider** | Arctic Shift (Reddit archive) |
| **Data Type** | User-generated content (posts, comments) |
| **Temporal Coverage** | 2024-01-01 to 2025-12-31 |
| **Initial Format** | Zstandard-compressed JSON (`.zst`) |
| **Distribution Method** | BitTorrent protocol |

### 2.2 Data Acquisition Workflow

#### 2.2.1 Torrent Download Process

**Tool**: `aria2c` v1.37.0+  
**Protocol**: BitTorrent  
**Target**: EC2 instance (Amazon Linux 2)

**Execution Parameters**:
```bash
aria2c ~/reddit.torrent --dir=~/downloads/reddit
nohup aria2c ~/reddit.torrent --dir=~/downloads/reddit > ~/aria2.log 2>&1 &
```

**File Naming Convention**:
- Posts: `RS_YYYY-MM.zst`
- Comments: `RC_YYYY-MM.zst`

**Volume Specifications**:
- Posts: 150-200 GB per monthly file
- Comments: 400-500 GB per monthly file
- Total monthly input: 600-700 GB compressed

#### 2.2.2 Decompression Pipeline

**Tool**: `unzstd` (Zstandard decompression)  
**Operation**: Stream decompression to JSON

**Execution**:
```bash
unzstd -c ~/reddit_data/zstfiles/RS_YYYY-MM.zst > ~/reddit_data/jsonfiles/RS_YYYY-MM.json
```

**Directory Structure**:
```
~/reddit_data/
├── zstfiles/      # Compressed source files
└── jsonfiles/     # Decompressed JSON output
```

**Validation**: File size monitoring via log analysis

### 2.3 ETL Pipeline Architecture

#### 2.3.1 Zone Architecture Specification

The pipeline implements a four-zone data lake architecture:

**Zone 1: Raw (Landing Zone)**
- **Format**: JSON (uncompressed)
- **Schema**: Unvalidated, raw schema from source
- **Retention**: Immutable archive
- **Access Pattern**: Write-only during ingestion
- **Storage**: S3 Standard storage class

**Zone 2: Staging**
- **Format**: Parquet (GZIP compression)
- **Schema**: Normalized, validated
- **Partitioning**: `year/month` hierarchical partitioning
- **Transformations**:
  - Flatten nested JSON structures
  - Convert `created_utc` (Unix timestamp) → `created_at` (datetime)
  - Extract partition columns (`year`, `month`)
- **Storage**: S3 Standard storage class

**Zone 3: Refining**
- **Format**: Parquet (GZIP compression)
- **Schema**: Enriched with quality flags
- **Transformations**:
  - Field normalization (trim whitespace, lowercase)
  - Validation (`is_valid_record` boolean flag)
  - Deduplication (exact match on `id` field)
  - Type conversion (arrays → JSON strings)
- **Storage**: S3 Standard storage class

**Zone 4: Publishing**
- **Format**: Parquet (Snappy compression)
- **Schema**: Production-ready, curated
- **Filters**: `is_valid_record = true`
- **Partitioning**: `year/month` with Glue Catalog integration
- **Optimization**: Snappy compression for query performance
- **Storage**: S3 Standard storage class

#### 2.3.2 AWS Glue Job Specifications

**Job Architecture**: Separate jobs for posts and comments

**Rationale**:
- Isolation of failure domains
- Independent resource allocation
- Simplified debugging
- Schema-specific optimization

**Job Configuration**:
```python
{
    "JobType": "Spark",
    "InputParameters": ["year", "month"],
    "ProcessingMode": "Incremental",
    "OutputFormat": "Parquet",
    "PartitionColumns": ["year", "month"],
    "CatalogUpdates": "Automatic"
}
```

**Processing Stages**:

1. **Raw → Staging**:
   - Read: JSON from Raw zone
   - Transform: Schema validation, timestamp conversion, partition extraction
   - Write: Parquet to Staging zone (overwrite mode)

2. **Staging → Refining**:
   - Read: Parquet from Staging zone
   - Transform: Field normalization, validation, deduplication
   - Write: Parquet to Refining zone (append mode, partition check)

3. **Refining → Publishing**:
   - Read: Parquet from Refining zone
   - Transform: Filter valid records, repartition
   - Write: Snappy Parquet to Publishing zone (overwrite mode)

### 2.4 Data Access Configuration

**Storage Location**:
- **Bucket**: `[REDACTED]`
- **Prefix**: `[REDACTED]`
- **Partition Structure**: `year=YYYY/month=MM/`

**Access Methods**:
- **Python**: `boto3` SDK v1.28.0+
- **Filesystem**: `s3fs` v2023.10.0+
- **Query Interface**: AWS Athena (planned)

**Authentication**:
- **Method**: IAM role-based authentication
- **Credentials**: Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- **Region**: `us-east-1` (default)

---

## 3. Classification Methodology

### 3.1 Classification Algorithm Specification

**Algorithm Type**: Deterministic pattern matching  
**Pattern Type**: Regular expression (regex)  
**Execution Mode**: In-memory batch processing  
**Determinism**: Fully deterministic (no randomization)

### 3.2 Pattern Definition

#### 3.2.1 Fear Expression Detection

**Pattern Type**: Case-insensitive regex alternation  
**Keywords**: 9 lexical patterns

**Pattern Definition**:
```python
fear_keywords = [
    'afraid', 'scared', 'worried', 'anxious', 
    'panic', 'terrified', 'fear', 'concern', 'nervous'
]
fear_pattern = '|'.join([re.escape(kw) for kw in fear_keywords])
```

**Matching Logic**:
```python
df['has_fear'] = df[text_col].str.lower().str.contains(
    fear_pattern, 
    na=False, 
    regex=True
)
```

**Output**: Boolean flag per record (`has_fear`)

**Algorithmic Properties**:
- **Time Complexity**: O(n × m) where n = records, m = pattern length
- **Space Complexity**: O(n) for boolean array
- **False Positives**: Possible (e.g., "not afraid" still matches)
- **False Negatives**: Possible (nuanced fear expressions)

#### 3.2.2 Question Pattern Detection

**Pattern Type**: Case-insensitive regex alternation  
**Keywords**: 9 lexical/interpunctuation patterns

**Pattern Definition**:
```python
qa_keywords = [
    '\\?', 'how', 'what', 'when', 'where', 'why', 
    'can i', 'should i', 'help'
]
qa_pattern = '|'.join([re.escape(kw) for kw in qa_keywords])
```

**Matching Logic**:
```python
df['is_question'] = df[text_col].str.lower().str.contains(
    qa_pattern, 
    na=False, 
    regex=True
)
```

**Output**: Boolean flag per record (`is_question`)

**Algorithmic Properties**:
- **Time Complexity**: O(n × m)
- **Space Complexity**: O(n)
- **False Positives**: Possible (e.g., "how-to" guides)
- **False Negatives**: Possible (rhetorical questions)

#### 3.2.3 Fear-Driven Question Detection

**Method**: Logical conjunction of fear and question flags

**Definition**:
```python
df['is_fear_driven_question'] = df['has_fear'] & df['is_question']
```

**Output**: Boolean flag per record

**Constraint**: Requires both conditions (AND logic)

**Limitation**: No semantic relationship validation

#### 3.2.4 Visa Stage Classification

**Method**: Multi-label classification via keyword matching  
**Output**: Count per stage (non-exclusive)

**Stage Definitions**:

| Stage | Keyword Patterns |
|-------|-----------------|
| **F1** | `'f1'`, `'student visa'`, `'student'`, `'f-1'`, `'f1 visa'`, `'student status'`, `'f1 status'` |
| **OPT** | `'opt'`, `'optional practical training'`, `'stem opt'`, `'cpt'`, `'work authorization'`, `'opt status'` |
| **H1B** | `'h1b'`, `'h-1b'`, `'work visa'`, `'h1-b'`, `'h1b visa'`, `'employment visa'`, `'h1b status'` |
| **Green Card** | `'green card'`, `'greencard'`, `'permanent resident'`, `'gc'`, `'i-140'`, `'i-485'`, `'permanent residency'` |
| **Citizenship** | `'citizenship'`, `'naturalization'`, `'citizen'`, `'n-400'`, `'naturalized'`, `'citizen status'` |
| **General Immigration** | `'immigration'`, `'visa'`, `'immigrant'`, `'uscis'`, `'immigration law'`, `'immigration status'` |

**Implementation**:
```python
stages = {
    'F1': ['f1', 'student visa', 'student', 'f-1', 'f1 visa', 'student status', 'f1 status'],
    'OPT': ['opt', 'optional practical training', 'stem opt', 'cpt', 'work authorization', 'opt status'],
    'H1B': ['h1b', 'h-1b', 'work visa', 'h1-b', 'h1b visa', 'employment visa', 'h1b status'],
    'greencard': ['green card', 'greencard', 'permanent resident', 'gc', 'i-140', 'i-485', 'permanent residency'],
    'citizenship': ['citizenship', 'naturalization', 'citizen', 'n-400', 'naturalized', 'citizen status'],
    'general_immigration': ['immigration', 'visa', 'immigrant', 'uscis', 'immigration law', 'immigration status']
}

stage_counts = {}
for stage, keywords in stages.items():
    pattern = '|'.join([re.escape(kw) for kw in keywords])
    count = df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
    stage_counts[stage] = int(count)
```

**Output**: Dictionary of stage → count mappings

**Properties**:
- **Non-exclusive**: Multiple stages can match single record
- **No validation**: No ground truth comparison
- **Aggregation**: Counts summed across all records

### 3.3 Text Column Detection Algorithm

**Method**: Case-insensitive substring matching on column names

**Search Patterns**:
```python
text_column_indicators = ['text', 'body', 'content', 'comment', 'post']
```

**Selection Logic**:
1. Iterate columns in DataFrame order
2. Test if any indicator exists in column name (case-insensitive)
3. Select first matching column
4. If no match: raise `ValueError`

**Implementation**:
```python
text_columns = [
    col for col in df.columns 
    if any(indicator in col.lower() for indicator in text_column_indicators)
]
if not text_columns:
    raise ValueError("No text column detected")
text_col = text_columns[0]
```

---

## 4. Processing Pipeline Architecture

### 4.1 Execution Model

**Parallelism Model**: Multi-threaded (I/O-bound operations)  
**Concurrency**: 30 worker threads  
**Execution Framework**: `concurrent.futures.ThreadPoolExecutor`

**Rationale**:
- S3 I/O operations are I/O-bound
- Thread-based parallelism sufficient for network I/O
- Avoids GIL limitations for I/O operations

### 4.2 Pipeline Stages

#### 4.2.1 Stage 1: File Discovery

**Operation**: List S3 objects with pagination  
**Tool**: `boto3.client('s3').list_objects_v2()`  
**Pagination**: Automatic via `get_paginator('list_objects_v2')`

**Output**: List of S3 object keys (Parquet files)

**Partitioning**: Organized by `year/month` from S3 prefix structure

#### 4.2.2 Stage 2: Parallel File Processing

**Executor**: `ThreadPoolExecutor(max_workers=30)`

**Per-File Operations**:
1. **Download**: Read Parquet file from S3 into memory buffer
2. **Parse**: Load Parquet to Pandas DataFrame via PyArrow
3. **Analyze**: Apply pattern matching algorithms
4. **Aggregate**: Compute per-file statistics
5. **Return**: File-level analysis results

**Error Handling**:
- Failed files logged with error details
- Processing continues for remaining files
- Error statistics tracked separately

**Implementation**:
```python
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = {
        executor.submit(process_file, file_key): file_key 
        for file_key in file_list
    }
    for future in as_completed(futures):
        result = future.result()
        if result.get('error'):
            log_error(result)
        else:
            aggregate_results(result)
```

#### 4.2.3 Stage 3: Aggregation

**Aggregation Levels**:
1. **File-level**: Sum metrics within single file
2. **Month-level**: Sum metrics across files in same month
3. **Year-level**: Sum metrics across all months in year

**Aggregation Operations**:
- **Sum**: Total counts (fear, Q&A, stage counts)
- **Mean**: Rates (fear rate, Q&A rate)
- **Count**: Files processed, records processed

**Output Format**: Nested JSON structure

### 4.3 Memory Management

**Strategy**: Per-file processing (streaming)

**Constraints**:
- Load single file into memory per worker
- Process file completely before loading next
- Release memory after processing

**Memory Requirements**:
- **Per-file**: ~2-4 GB (depending on Parquet file size)
- **Total**: 30 workers × 4 GB = 120 GB peak (theoretical)
- **Actual**: Lower due to staggered processing

### 4.4 Performance Characteristics

**Throughput**: ~1000-2000 files/hour (30 workers)  
**Latency**: ~2-5 seconds per file (I/O + processing)  
**Scalability**: Linear with worker count (I/O-bound)

**Bottlenecks**:
- S3 read bandwidth
- Network latency
- Parquet decompression

---

## 5. Results Specification

### 5.1 Output Schema

**Format**: JSON (UTF-8 encoding)  
**Structure**: Hierarchical (year → month → file)

**Schema Definition**:
```json
{
  "analysis_date": "ISO 8601 datetime",
  "year": 2024,
  "analysis_type": "complete_dataset_NO_SAMPLING",
  "total_records": 115289778,
  "total_fear": 21793076,
  "total_qa": 47759518,
  "fear_rate": 0.189,
  "qa_rate": 0.414,
  "stage_analysis": {
    "F1": 14502378,
    "OPT": 40422496,
    "H1B": 206204,
    "greencard": 7208835,
    "citizenship": 1280921,
    "general_immigration": 1086146
  },
  "monthly_breakdown": {
    "2024-01": { /* month-level stats */ },
    "2024-02": { /* month-level stats */ }
  },
  "summary_stats": {
    "total_records": 115289778,
    "total_fear": 21793076,
    "total_qa": 47759518,
    "fear_rate": 0.189,
    "qa_rate": 0.414,
    "months_covered": 12,
    "total_files_processed": 999,
    "total_bytes_processed": 21474836480
  }
}
```

### 5.2 2024 Results

| Metric | Value |
|--------|-------|
| **Total Records** | 115,289,778 |
| **Total Fear Cases** | 21,793,076 |
| **Fear Rate** | 18.9% (0.189) |
| **Total Q&A Cases** | 47,759,518 |
| **Q&A Rate** | 41.4% (0.414) |

**Visa Stage Distribution**:
| Stage | Count | Percentage |
|-------|-------|------------|
| F1 | 14,502,378 | 12.6% |
| OPT | 40,422,496 | 35.1% |
| H1B | 206,204 | 0.2% |
| Green Card | 7,208,835 | 6.3% |
| Citizenship | 1,280,921 | 1.1% |
| General Immigration | 1,086,146 | 0.9% |

### 5.3 2025 Results

| Metric | Value |
|--------|-------|
| **Total Records** | 77,073,526 |
| **Total Fear Cases** | 12,160,505 |
| **Fear Rate** | 15.8% (estimated) |
| **Total Q&A Cases** | 51,897,196 |
| **Q&A Rate** | 67.4% (estimated) |

**Visa Stage Distribution**:
| Stage | Count | Percentage |
|-------|-------|------------|
| F1 | 19,131,575 | 24.8% |
| OPT | 24,036,491 | 31.2% |
| H1B | 154,421 | 0.2% |
| Green Card | 4,235,052 | 5.5% |
| Citizenship | 1,010,571 | 1.3% |
| General Immigration | 812,636 | 1.1% |

---

## 6. Technical Stack

### 6.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.8+ | Runtime environment |
| **pandas** | 1.3.0+ | DataFrame operations |
| **numpy** | 1.21.0+ | Numerical computations |
| **pyarrow** | 5.0.0+ | Parquet I/O |
| **boto3** | 1.28.0+ | AWS SDK |
| **s3fs** | 2023.10.0+ | S3 filesystem interface |

### 6.2 Processing Libraries

| Library | Module | Usage |
|---------|--------|-------|
| **re** | Standard library | Regex pattern matching |
| **concurrent.futures** | Standard library | ThreadPoolExecutor |
| **io** | Standard library | BytesIO buffer management |
| **json** | Standard library | Result serialization |
| **logging** | Standard library | Process logging |

### 6.3 Infrastructure Components

| Component | Service | Purpose |
|----------|--------|---------|
| **S3** | AWS S3 | Object storage |
| **Glue** | AWS Glue | ETL processing |
| **Athena** | AWS Athena | SQL query interface (planned) |
| **EC2** | Amazon EC2 | Compute infrastructure |

### 6.4 Data Formats

| Format | Library | Compression | Usage |
|--------|---------|-------------|-------|
| **Parquet** | PyArrow | Snappy (final), GZIP (intermediate) | Columnar storage |
| **JSON** | Standard library | None | Results, configuration |
| **Zstandard** | External tool | `.zst` | Source decompression |

---

## 7. Limitations and Constraints

### 7.1 Algorithmic Limitations

**Pattern Matching Constraints**:
- **Context**: No semantic understanding
- **Negation**: Negative constructions may match (e.g., "not afraid")
- **Polysemy**: Ambiguous word meanings not disambiguated
- **Collocations**: Multi-word expressions not handled

**Classification Accuracy**:
- **No Ground Truth**: No validation dataset
- **No Confidence Scores**: Binary classification only
- **No Uncertainty Quantification**: Deterministic boolean output

### 7.2 Processing Limitations

**Scalability**:
- **Single-Instance**: Processing limited to single EC2 instance
- **Memory Constraints**: Large files may exceed available RAM
- **Network Bandwidth**: S3 download speed limits throughput

**Error Handling**:
- **Partial Failures**: Failed files skipped, not retried
- **No Recovery**: Processing must restart from beginning
- **No Checkpointing**: Cannot resume interrupted processing

### 7.3 Data Quality Limitations

**Schema Assumptions**:
- Text column detection heuristic-based
- Assumes consistent column naming
- No schema validation against source

**Data Completeness**:
- No validation of record completeness
- No detection of missing partitions
- No verification of data integrity

---

## 8. Reproducibility

### 8.1 Deterministic Execution

**Random Seed**: Fixed seed (42) for any random operations  
**Processing Order**: Deterministic file ordering (lexicographic)  
**Algorithm**: Deterministic pattern matching (no randomness)

### 8.2 Version Control

**Code Repository**: Git  
**Configuration**: Version-controlled config files  
**Dependencies**: `requirements.txt` with pinned versions

### 8.3 Execution Environment

**Operating System**: Linux (Amazon Linux 2)  
**Python Version**: 3.8+  
**Dependencies**: Pinned versions in `requirements.txt`

---

## 9. BERT Classifier Framework (Future Enhancement)

### 9.1 Implementation Status

**Status**: Implemented but not used for reported results  
**Location**: `src/classification/bert_classifier.py`  
**Purpose**: Future enhancement framework

### 9.2 Architecture

**Model**: BERT-base-uncased (110M parameters)  
**Framework**: PyTorch  
**Library**: Hugging Face Transformers

**Components**:
- `RedditDataset`: PyTorch Dataset implementation
- `BERTMultiLabelClassifier`: Neural network architecture
- `RedditBERTClassifier`: Training and inference interface

### 9.3 Prerequisites for Usage

1. **Annotation**: 1000+ manually annotated records
2. **Training**: Model training on annotated dataset
3. **Validation**: Held-out test set evaluation
4. **Inference**: Batch inference on complete dataset

### 9.4 Comparison with Keyword Matching

| Aspect | Keyword Matching | BERT Classification |
|--------|------------------|---------------------|
| **Training Data** | None required | Required (1000+ records) |
| **Computational Cost** | Low (O(n)) | High (O(n × m) where m = model size) |
| **Interpretability** | High | Low |
| **Accuracy** | Baseline | Expected higher |
| **Scalability** | High | Moderate (GPU recommended) |

---

## 10. Data Privacy and Compliance

### 10.1 Data Source

**Type**: Publicly available Reddit data  
**License**: Reddit Terms of Service compliant  
**Anonymization**: User identifiers excluded from analysis

### 10.2 Analysis Scope

**Granularity**: Aggregate statistics only  
**Personal Information**: No PII included in results  
**User Identification**: No individual user tracking

### 10.3 Compliance

**Reddit ToS**: Compliant with Terms of Service  
**Data Minimization**: Only necessary fields analyzed  
**Responsible Use**: Research purposes only

---

## Appendix A: Regular Expression Patterns

### A.1 Fear Pattern

```regex
(afraid|scared|worried|anxious|panic|terrified|fear|concern|nervous)
```

### A.2 Question Pattern

```regex
(\?|how|what|when|where|why|can i|should i|help)
```

### A.3 Visa Stage Patterns

**F1**:
```regex
(f1|student visa|student|f-1|f1 visa|student status|f1 status)
```

**OPT**:
```regex
(opt|optional practical training|stem opt|cpt|work authorization|opt status)
```

**H1B**:
```regex
(h1b|h-1b|work visa|h1-b|h1b visa|employment visa|h1b status)
```

**Green Card**:
```regex
(green card|greencard|permanent resident|gc|i-140|i-485|permanent residency)
```

**Citizenship**:
```regex
(citizenship|naturalization|citizen|n-400|naturalized|citizen status)
```

**General Immigration**:
```regex
(immigration|visa|immigrant|uscis|immigration law|immigration status)
```

---

## Appendix B: Performance Metrics

### B.1 Processing Statistics

**2024 Processing**:
- Files processed: 999
- Total bytes: 21.5 GB
- Processing time: ~24 hours (estimated)
- Throughput: ~42 files/hour

**2025 Processing**:
- Files processed: 6,923
- Total bytes: Estimated 15 GB
- Processing time: ~6 hours (estimated)
- Throughput: ~1,154 files/hour

### B.2 Resource Utilization

**Compute**:
- CPU utilization: 30-50% (I/O-bound)
- Memory usage: 2-4 GB per worker
- Network I/O: S3 download bandwidth limited

**Storage**:
- Input: ~36.5 GB compressed Parquet
- Output: <1 MB JSON results

---

**Document Version**: 2.1  
**Last Updated**: October 2025  
**Classification**: Technical Documentation  
**Status**: Production
