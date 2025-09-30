#!/usr/bin/env python3
"""
Data Preprocessing Script
Cleans and preprocesses Reddit data with immigration-specific patterns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import glob
from pathlib import Path
from src.data.preprocess import preprocess_data

def preprocess_data_files(input_dir="data/raw", output_dir="data/processed"):
    """Preprocess all data files in input directory"""
    print(f" Preprocessing data from {input_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not input_files:
        print(f" No parquet files found in {input_dir}")
        return False
    
    total_original = 0
    total_processed = 0
    
    for input_file in input_files:
        print(f" Processing {input_file}...")
        
        # Load data
        df_raw = pd.read_parquet(input_file)
        original_count = len(df_raw)
        total_original += original_count
        
        # Preprocess
        processed = preprocess_data(df_raw)
        processed = processed[["text", "label"]].dropna()
        
        processed_count = len(processed)
        total_processed += processed_count
        
        # Save processed data
        output_file = os.path.join(output_dir, f"processed_{os.path.basename(input_file)}")
        processed.to_parquet(output_file)
        
        retention_rate = processed_count/original_count if original_count > 0 else 0
        print(f"   {original_count} → {processed_count} rows ({retention_rate*100:.1f}% retained)")
        print(f"   Saved to: {output_file}")
    
    overall_retention = total_processed/total_original if total_original > 0 else 0
    print(f" Preprocessing complete: {total_original} → {total_processed} rows ({overall_retention*100:.1f}% retained)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess Reddit data")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    # Preprocess data
    success = preprocess_data_files(args.input, args.output)
    
    if success:
        print(" Data preprocessing completed successfully!")
    else:
        print(" Data preprocessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
