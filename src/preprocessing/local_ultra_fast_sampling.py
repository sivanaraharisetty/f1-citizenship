#!/usr/bin/env python3
"""
Local Ultra Fast Reddit Data Sampling
Processes local parquet files from /Users/comm-naraharisetty/Documents/f1_s3_files/
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_ultra_fast_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LocalUltraFastSampler:
    """Ultra-fast local parquet file sampler"""
    
    def __init__(self, data_dir: str, max_workers: int = 10, rows_per_file: int = 100):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.rows_per_file = rows_per_file
        self.stats = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_records': 0,
            'bytes_processed': 0,
            'start_time': time.time()
        }
        self.sample_records = []
        
        # Ensure output directory exists
        os.makedirs('sampled_data', exist_ok=True)
        
        logger.info(f" Local Ultra Fast Sampler initialized")
        logger.info(f" Data directory: {self.data_dir}")
        logger.info(f" Max workers: {self.max_workers}")
        logger.info(f" Target rows per file: {self.rows_per_file}")
    
    def discover_files(self) -> List[Path]:
        """Discover all parquet files in the data directory"""
        logger.info(" Discovering local parquet files...")
        
        parquet_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.parquet', '.csv')):
                    parquet_files.append(Path(root) / file)
        
        logger.info(f" Found {len(parquet_files)} files")
        return parquet_files
    
    def sample_file(self, file_path: Path) -> Tuple[bool, List[Dict], str]:
        """Sample a single parquet file"""
        try:
            logger.info(f" Processing: {file_path.name}")
            
            # Get file size
            file_size = file_path.stat().st_size
            self.stats['bytes_processed'] += file_size
            logger.info(f" File size: {file_size:,} bytes")
            
            # Read parquet file
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:  # CSV file
                df = pd.read_csv(file_path)
            
            logger.info(f" Loaded {len(df)} rows from {file_path.name}")
            
            if len(df) == 0:
                return False, [], "Empty file"
            
            # Sample rows
            if len(df) > self.rows_per_file:
                sample_df = df.sample(n=self.rows_per_file, random_state=42)
            else:
                sample_df = df
            
            # Convert to records
            sample_records = sample_df.to_dict('records')
            
            # Add metadata
            for record in sample_records:
                record['_source_file'] = str(file_path)
                record['_file_size'] = file_size
                record['_total_rows'] = len(df)
                record['_sampled_rows'] = len(sample_df)
            
            logger.info(f" Sampled {len(sample_records)} records from {file_path.name}")
            return True, sample_records, f"Success: {len(sample_records)} records"
            
        except Exception as e:
            logger.error(f" Failed to process {file_path.name}: {e}")
            return False, [], f"Failed: {e}"
    
    def process_files_parallel(self, file_paths: List[Path]) -> None:
        """Process files in parallel"""
        logger.info(f" Starting parallel processing with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.sample_file, file_path): file_path 
                for file_path in file_paths
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                self.stats['files_processed'] += 1
                
                try:
                    success, records, message = future.result()
                    
                    if success:
                        self.stats['files_successful'] += 1
                        self.stats['total_records'] += len(records)
                        self.sample_records.extend(records)
                        logger.info(f" {file_path.name}: {message}")
                    else:
                        self.stats['files_failed'] += 1
                        logger.warning(f" {file_path.name}: {message}")
                    
                    # Progress update
                    if self.stats['files_processed'] % 10 == 0:
                        elapsed = time.time() - self.stats['start_time']
                        rate = self.stats['files_processed'] / elapsed * 60
                        logger.info(f" Progress: {self.stats['files_processed']}/{len(file_paths)} "
                                  f"({self.stats['files_processed']/len(file_paths)*100:.1f}%) - "
                                  f"Rate: {rate:.1f} files/min - Records: {self.stats['total_records']}")
                
                except Exception as e:
                    self.stats['files_failed'] += 1
                    logger.error(f" {file_path.name}: Exception: {e}")
    
    def save_results(self) -> None:
        """Save sampling results"""
        logger.info(" Saving results...")
        
        if not self.sample_records:
            logger.warning(" No records to save")
            return
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.sample_records)
        output_file = 'sampled_data/local_ultra_fast_sample.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f" Results saved to {output_file}")
        
        # Save metadata
        metadata = {
            'total_files_processed': self.stats['files_processed'],
            'successful_files': self.stats['files_successful'],
            'failed_files': self.stats['files_failed'],
            'success_rate': self.stats['files_successful'] / max(self.stats['files_processed'], 1) * 100,
            'total_records': self.stats['total_records'],
            'bytes_processed': self.stats['bytes_processed'],
            'processing_time_seconds': time.time() - self.stats['start_time'],
            'files_per_minute': self.stats['files_processed'] / (time.time() - self.stats['start_time']) * 60,
            'output_file': output_file,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = 'sampled_data/local_ultra_fast_sample_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f" Metadata saved to {metadata_file}")
    
    def print_summary(self) -> None:
        """Print processing summary"""
        elapsed = time.time() - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info(" LOCAL ULTRA FAST SAMPLING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱  Total time: {elapsed:.2f} seconds")
        logger.info(f" Files processed: {self.stats['files_processed']}")
        logger.info(f" Success rate: {self.stats['files_successful'] / max(self.stats['files_processed'], 1) * 100:.1f}%")
        logger.info(f" Files per minute: {self.stats['files_processed'] / elapsed * 60:.1f}")
        logger.info(f" Total records: {self.stats['total_records']}")
        logger.info(f" Output file: sampled_data/local_ultra_fast_sample.parquet")
        logger.info("=" * 60)

def main():
    """Main execution function"""
    logger.info(" Starting LOCAL Ultra Fast Reddit Data Sampling")
    logger.info("=" * 60)
    
    # Configuration
    data_dir = "/Users/comm-naraharisetty/Documents/f1_s3_files"
    max_workers = 10
    rows_per_file = 100
    
    # Initialize sampler
    sampler = LocalUltraFastSampler(
        data_dir=data_dir,
        max_workers=max_workers,
        rows_per_file=rows_per_file
    )
    
    # Discover files
    file_paths = sampler.discover_files()
    
    if not file_paths:
        logger.error(" No parquet files found!")
        return
    
    # Limit files for testing (remove this for full processing)
    file_paths = file_paths[:50]  # Process first 50 files
    logger.info(f" Processing {len(file_paths)} files (limited for testing)")
    
    # Process files
    sampler.process_files_parallel(file_paths)
    
    # Save results
    sampler.save_results()
    
    # Print summary
    sampler.print_summary()

if __name__ == "__main__":
    main()
