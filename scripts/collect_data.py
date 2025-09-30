#!/usr/bin/env python3
"""
Data Collection Script
Collects Reddit data from S3 with configurable parameters
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from datetime import datetime
from src.data.loader import iter_load_data

def collect_data(bucket_name, comments_prefix, posts_prefix, 
                files_per_chunk=1, rows_per_chunk=1000, max_chunks=None, output_dir="data/raw"):
    """Collect data from S3 and save to local directory"""
    print(f" Collecting data from S3...")
    print(f"   Bucket: {bucket_name}")
    print(f"   Comments: {comments_prefix}")
    print(f"   Posts: {posts_prefix}")
    print(f"   Chunk size: {files_per_chunk} files, {rows_per_chunk} rows")
    
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_count = 0
    total_rows = 0
    
    try:
        for df_raw, keys in iter_load_data(
            bucket_name,
            comments_prefix, 
            posts_prefix,
            files_per_chunk=files_per_chunk,
            rows_per_chunk=rows_per_chunk
        ):
            # Build text fields
            if "body" in df_raw.columns:
                mask = df_raw["__source_label__"] == "comments"
                df_raw.loc[mask, "text"] = df_raw.loc[mask, "body"].fillna("")
            if "title" in df_raw.columns or "selftext" in df_raw.columns:
                mask = df_raw["__source_label__"] == "posts"
                title = df_raw.loc[mask, "title"].fillna("") if "title" in df_raw.columns else ""
                selftext = df_raw.loc[mask, "selftext"].fillna("") if "selftext" in df_raw.columns else ""
                df_raw.loc[mask, "text"] = title + " " + selftext
            
            # Save chunk
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_count:03d}.parquet")
            df_raw.to_parquet(chunk_file)
            
            chunk_count += 1
            total_rows += len(df_raw)
            
            print(f"📦 Chunk {chunk_count}: {len(df_raw)} rows saved to {chunk_file}")
            
            if max_chunks and chunk_count >= max_chunks:
                break
                
    except Exception as e:
        print(f" Data collection error: {e}")
        return False
        
    print(f" Data collection complete: {chunk_count} chunks, {total_rows} total rows")
    return True

def main():
    parser = argparse.ArgumentParser(description="Collect Reddit data from S3")
    parser.add_argument("--config", default="configs/config.yaml", help="Configuration file")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--max-chunks", type=int, help="Maximum number of chunks to collect")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    s3_config = config['data']['s3']
    chunking_config = config['data']['chunking']
    
    # Collect data
    success = collect_data(
        bucket_name=s3_config['bucket'],
        comments_prefix=s3_config['comments_path'],
        posts_prefix=s3_config['posts_path'],
        files_per_chunk=chunking_config['files_per_chunk'],
        rows_per_chunk=chunking_config['rows_per_chunk'],
        max_chunks=args.max_chunks,
        output_dir=args.output
    )
    
    if success:
        print(" Data collection completed successfully!")
    else:
        print(" Data collection failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
