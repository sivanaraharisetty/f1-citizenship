# Real-Time Training Status

## Current Training Progress

**Status**: ACTIVE - Processing 2TB Immigration Dataset  
**Started**: 2025-09-23 09:17:09 UTC  
**Device**: Apple Silicon MPS (GPU Accelerated)  
**Current Progress**: 2.2% Complete (8,396/382,510 steps)

## Live Metrics

### Processing Statistics
- **Total Dataset Size**: 1.57 TB (1,573,001,799,868 bytes)
- **Current Chunk**: 0 (First chunk processing)
- **Rows in Current Chunk**: 791,847
- **Files Processed**: 1
- **Processing Speed**: 1.49 iterations/second
- **Estimated Completion**: ~69 hours 40 minutes

### Model Performance
- **Architecture**: BERT-base-uncased with custom classifier
- **Training Strategy**: Early stopping with F1-score optimization
- **Mixed Precision**: Enabled (MPS optimized)
- **Batch Size**: 16 (GPU optimized)
- **Learning Rate**: 2e-5

## Data Processing Pipeline

### Stage 1: Data Ingestion
- **Source**: S3 Reddit Comments & Posts
- **Format**: Parquet files with row-group streaming
- **Chunking Strategy**: 25,000 rows per chunk
- **Memory Management**: Bounded by row-group size

### Stage 2: Text Preprocessing
- **Text Cleaning**: Lowercase, URL removal, special character handling
- **Label Classification**: 5-stage immigration journey mapping
- **Subreddit Integration**: Community-based labeling hints

### Stage 3: Model Training
- **Tokenization**: BERT tokenizer with 128 max length
- **Train/Validation Split**: 80/20 with stratification
- **Early Stopping**: Patience of 2 epochs
- **Checkpointing**: Per-chunk model saves

## Immigration Journey Classification

### Target Categories
1. **Student Visa Stage** (F1, CPT, OPT, STEM OPT)
2. **Work Visa Stage** (H1B, Employer Sponsorship)
3. **Permanent Residency Stage** (PERM, I-140, Green Card)
4. **Citizenship Stage** (Naturalization)
5. **General Immigration & Legal Issues**

### Keyword Patterns
- **Student**: F1, CPT, OPT, STEM OPT, visa interview, I-765
- **Work**: H1B, H-1B, employer sponsor, job search visa
- **Permanent**: I-140, PERM, green card, GC, I-485
- **Citizenship**: naturalization, citizenship, immigration reform
- **General**: visa denial, delays, backlog, legal help

## Real-Time Monitoring

### Log Files
- **Main Log**: `logs/classifier.log`
- **Training Logs**: `results/YYYYMMDD/chunk_*/logs/`
- **Metrics**: `results/YYYYMMDD/evaluation_results.json`

### Key Metrics Tracked
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1 across all classes
- **Precision/Recall**: Per-class performance
- **Processing Speed**: MB/s throughput
- **Memory Usage**: GPU/CPU utilization
- **ETA**: Estimated completion time

## Current Results Structure

```
results/20250923/
├── chunk_0/                    # Current chunk processing
│   ├── config.json            # Model configuration
│   ├── model.safetensors      # Model weights
│   ├── training_args.bin      # Training parameters
│   └── logs/                  # Training logs
├── evaluation_results.json    # Global metrics (JSONL)
├── label_mapping.json         # Class mapping
├── state.json                 # Resume state
└── processed_keys.sqlite      # Progress tracking
```

## Performance Optimization

### Hardware Utilization
- **GPU**: Apple Silicon MPS (Metal Performance Shaders)
- **Memory**: Optimized for 16GB+ systems
- **Storage**: SSD recommended for checkpoint I/O
- **Network**: S3 streaming with retry logic

### Scalability Features
- **Row-Group Streaming**: Memory-bounded processing
- **External Key Tracking**: SQLite-based progress persistence
- **Resume Capability**: Automatic checkpoint recovery
- **Fault Tolerance**: Exponential backoff retry logic

## Expected Outcomes

### Model Performance Targets
- **Accuracy**: >85% on immigration stage classification
- **F1-Score**: >0.80 weighted average
- **Processing Time**: ~70 hours for 2TB dataset
- **Memory Usage**: <8GB peak during processing

### Research Applications
- **Journey Mapping**: Track user progression through immigration stages
- **Pain Point Analysis**: Identify common issues at each stage
- **Policy Impact**: Measure effects of policy changes on discussions
- **Community Support**: Analyze help-seeking patterns

## Monitoring Commands

### Check Progress
```bash
# View real-time logs
tail -f logs/classifier.log

# Check current metrics
cat results/20250923/evaluation_results.json | tail -1

# Monitor system resources
top -pid $(pgrep -f "python -m src.main")
```

### Resume Training
```bash
# Training automatically resumes from last checkpoint
python -m src.main
```

## Next Steps

1. **Complete Current Chunk**: ~69 hours remaining
2. **Process Remaining Chunks**: Iterative processing of 2TB dataset
3. **Generate Final Model**: Best model selection and evaluation
4. **Export Results**: Comprehensive analytics and insights
5. **Research Analysis**: Immigration journey mapping and trends

---

**Last Updated**: 2025-09-23 12:30:00 UTC  
**Next Update**: Every 30 minutes during active training
