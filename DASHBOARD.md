# Immigration Journey Analyzer - Executive Dashboard

## Live Training Status

**Current Status**: ACTIVE - Processing 2TB Immigration Dataset  
**Started**: 2025-09-23 09:17:09 UTC  
**Progress**: 2.2% Complete (8,396/382,510 steps)  
**ETA**: ~69 hours 40 minutes  
**Device**: Apple Silicon MPS (GPU Accelerated)

## Key Performance Indicators

### Processing Metrics
- **Dataset Size**: 1.57 TB
- **Current Chunk**: 0 (791,847 rows)
- **Processing Speed**: 1.49 iterations/second
- **Memory Usage**: Optimized for 16GB+ systems
- **Fault Tolerance**: Enterprise-grade with resume capability

### Model Performance
- **Architecture**: BERT-base-uncased with 6-class classifier
- **Training Strategy**: Early stopping with F1-score optimization
- **Mixed Precision**: Enabled (MPS optimized)
- **Batch Size**: 16 (GPU optimized)
- **Learning Rate**: 2e-5

## Immigration Journey Classification

### Target Categories
1. **Citizenship** - Naturalization processes
2. **General Immigration** - Cross-cutting legal issues
3. **Green Card** - Permanent residency processes
4. **Irrelevant** - Non-immigration content
5. **Student Visa** - F1, CPT, OPT, STEM OPT
6. **Work Visa** - H1B, employer sponsorship

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

## Business Value

### Research Applications
- **Immigration Policy Analysis**: Measure impact of policy changes
- **User Journey Mapping**: Track progression through immigration stages
- **Community Support Analysis**: Identify help-seeking patterns
- **Trend Analysis**: Monitor discussion topics over time

### Technical Achievements
- **Enterprise-Scale Processing**: 2TB+ dataset handling
- **Real-Time Intelligence**: ETA tracking and progress monitoring
- **Production-Grade**: Fault tolerance and resume capability
- **Comprehensive Analytics**: Per-stage metrics and insights

## Expected Outcomes

### Model Performance Targets
- **Accuracy**: >85% on immigration stage classification
- **F1-Score**: >0.80 weighted average across all classes
- **Processing Time**: ~70 hours for complete 2TB dataset
- **Memory Usage**: <8GB peak during processing

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

1. **Complete Current Training**: ~69 hours remaining
2. **Process Remaining Chunks**: Iterative processing of 2TB dataset
3. **Generate Final Model**: Best model selection and evaluation
4. **Export Results**: Comprehensive analytics and insights
5. **Research Analysis**: Immigration journey mapping and trends

---

**Last Updated**: 2025-09-23 12:30:00 UTC  
**Next Update**: Every 30 minutes during active training  
**Repository**: [https://github.com/sivanaraharisetty/f1-citizenship](https://github.com/sivanaraharisetty/f1-citizenship)
