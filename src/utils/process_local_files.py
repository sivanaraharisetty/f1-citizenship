#!/usr/bin/env python3
"""
Simple Local File Processor
Processes local parquet files from your f1_s3_files directory
"""

import os
import pandas as pd
import time
from pathlib import Path

def process_local_files():
    """Process local parquet files"""
    print(" Starting Local File Processing")
    print("=" * 50)
    
    # Configuration
    data_dir = "/Users/comm-naraharisetty/Documents/f1_s3_files"
    output_file = "sampled_data/local_sample.parquet"
    
    # Ensure output directory exists
    os.makedirs('sampled_data', exist_ok=True)
    
    # Find all parquet files
    print(" Finding parquet files...")
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.parquet', '.csv')):
                parquet_files.append(Path(root) / file)
    
    print(f" Found {len(parquet_files)} files")
    
    # Process first 10 files as a test
    test_files = parquet_files[:10]
    print(f" Processing {len(test_files)} files for testing")
    
    all_records = []
    successful_files = 0
    
    for i, file_path in enumerate(test_files):
        try:
            print(f" Processing {i+1}/{len(test_files)}: {file_path.name}")
            
            # Read file
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            print(f" Loaded {len(df)} rows")
            
            if len(df) > 0:
                # Sample 100 rows
                sample_size = min(100, len(df))
                sample_df = df.sample(n=sample_size, random_state=42)
                
                # Add source info
                sample_df['_source_file'] = str(file_path)
                sample_df['_total_rows'] = len(df)
                
                # Convert to records
                records = sample_df.to_dict('records')
                all_records.extend(records)
                
                successful_files += 1
                print(f" Sampled {len(records)} records")
            else:
                print(" Empty file")
                
        except Exception as e:
            print(f" Error processing {file_path.name}: {e}")
    
    # Save results
    if all_records:
        print(f" Saving {len(all_records)} records...")
        result_df = pd.DataFrame(all_records)
        result_df.to_parquet(output_file, index=False)
        print(f" Results saved to {output_file}")
        
        # Print summary
        print("=" * 50)
        print(" PROCESSING COMPLETED!")
        print(f" Files processed: {len(test_files)}")
        print(f" Successful: {successful_files}")
        print(f" Total records: {len(all_records)}")
        print(f" Output: {output_file}")
        print("=" * 50)
    else:
        print(" No records to save")

if __name__ == "__main__":
    process_local_files()

