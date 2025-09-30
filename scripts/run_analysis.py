#!/usr/bin/env python3
"""
Analysis Script
Runs comprehensive descriptive analysis on processed data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import glob
from pathlib import Path
from src.analysis.descriptive_analysis import ImmigrationDataAnalyzer

def run_analysis(input_dir="data/processed", output_dir="results/analysis"):
    """Run comprehensive analysis on processed data"""
    print(f" Running analysis on data from {input_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all processed files
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not input_files:
        print(f" No processed files found in {input_dir}")
        return False
    
    # Load and combine all data
    all_data = []
    for input_file in input_files:
        print(f"📂 Loading {input_file}...")
        df = pd.read_parquet(input_file)
        all_data.append(df)
    
    if not all_data:
        print(" No data to analyze")
        return False
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f" Total samples: {len(combined_df)}")
    print(f" Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    # Run comprehensive analysis
    analyzer = ImmigrationDataAnalyzer()
    analysis_results, df_with_clusters = analyzer.comprehensive_analysis(combined_df)
    
    # Save analysis results
    import json
    with open(os.path.join(output_dir, "descriptive_analysis.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Save analysis report
    analyzer.save_analysis_report(os.path.join(output_dir, "descriptive_analysis_report.txt"))
    
    # Save processed data with clusters
    df_with_clusters.to_parquet(os.path.join(output_dir, "processed_data_with_clusters.parquet"))
    
    print(" Analysis complete!")
    print(f" Key findings:")
    print(f"   - Total samples: {analysis_results['basic_stats']['total_samples']}")
    print(f"   - Label distribution: {analysis_results['basic_stats']['label_distribution']}")
    print(f"   - Top keywords: {', '.join([word for word, _ in analysis_results['top_keywords'][:5]])}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run descriptive analysis")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="results/analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # Run analysis
    success = run_analysis(args.input, args.output)
    
    if success:
        print(" Analysis completed successfully!")
    else:
        print(" Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
