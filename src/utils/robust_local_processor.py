#!/usr/bin/env python3
"""
Robust Local File Processor
Handles CSV parsing issues and data type problems
"""

import os
import pandas as pd
import time
from pathlib import Path

def process_local_files_robust():
    """Process local files with robust error handling"""
    print(" Starting Robust Local File Processing")
    print("=" * 50)
    
    # Configuration
    data_dir = "/Users/comm-naraharisetty/Documents/f1_s3_files"
    output_file = "sampled_data/robust_local_sample.parquet"
    
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
    
    # Process first 20 files as a test
    test_files = parquet_files[:20]
    print(f" Processing {len(test_files)} files for testing")
    
    all_records = []
    successful_files = 0
    failed_files = 0
    
    for i, file_path in enumerate(test_files):
        try:
            print(f" Processing {i+1}/{len(test_files)}: {file_path.name}")
            
            # Read file with robust error handling
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                # Handle CSV parsing issues
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
                except:
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin-1')
                    except:
                        print(f" Could not read {file_path.name}, skipping")
                        failed_files += 1
                        continue
            
            print(f" Loaded {len(df)} rows")
            
            if len(df) > 0:
                # Sample 50 rows (reduced for testing)
                sample_size = min(50, len(df))
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
                failed_files += 1
                
        except Exception as e:
            print(f" Error processing {file_path.name}: {e}")
            failed_files += 1
    
    # Save results with robust handling
    if all_records:
        print(f" Saving {len(all_records)} records...")
        try:
            result_df = pd.DataFrame(all_records)
            
            # Convert all columns to string to avoid type issues
            for col in result_df.columns:
                result_df[col] = result_df[col].astype(str)
            
            # Save as CSV first (more robust)
            csv_output = output_file.replace('.parquet', '.csv')
            result_df.to_csv(csv_output, index=False)
            print(f" Results saved to {csv_output}")
            
            # Try to save as parquet
            try:
                result_df.to_parquet(output_file, index=False)
                print(f" Parquet version saved to {output_file}")
            except Exception as e:
                print(f" Could not save parquet: {e}")
                print(f" CSV version available at {csv_output}")
            
        except Exception as e:
            print(f" Error saving results: {e}")
            return
        
        # Print summary
        print("=" * 50)
        print(" PROCESSING COMPLETED!")
        print(f" Files processed: {len(test_files)}")
        print(f" Successful: {successful_files}")
        print(f" Failed: {failed_files}")
        print(f" Total records: {len(all_records)}")
        print(f" Output: {csv_output}")
        if os.path.exists(output_file):
            print(f" Parquet: {output_file}")
        print("=" * 50)
        
        # Show sample of data
        if len(all_records) > 0:
            print("\n Sample of processed data:")
            sample_record = all_records[0]
            for key, value in list(sample_record.items())[:5]:  # Show first 5 fields
                print(f"  {key}: {str(value)[:50]}...")
                
    else:
        print(" No records to save")

if __name__ == "__main__":
    process_local_files_robust()

