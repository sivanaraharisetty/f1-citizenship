#!/usr/bin/env python3
"""
Parallel 2025 Analysis - Runs alongside 2024 analysis
This script analyzes ALL 2025 data while 2024 analysis is running
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import boto3
import json
import logging
import re
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_2025_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "siva-test-9"
S3_2025_PREFIX = "reddit/new_data/2025_data/"
S3_RESULTS_PREFIX = "reddit/new_data/results/"

class Parallel2025Analyzer:
    """Analyzer for 2025 data - runs in parallel with 2024 analysis"""
    
    def __init__(self, max_workers=20):
        self.max_workers = max_workers
        self.s3_client = boto3.client('s3')
        self.stats = {
            '2025': {
                'total_files': 0,
                'total_records': 0,
                'processed_files': 0,
                'failed_files': 0
            }
        }
    
    def get_all_2025_files_by_month(self) -> Dict[int, List[str]]:
        """Get ALL 2025 files organized by month - handles pagination"""
        logger.info(f" Discovering ALL 2025 files with pagination...")
        
        try:
            all_files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=S3_BUCKET,
                Prefix=S3_2025_PREFIX
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.parquet'):
                            all_files.append(obj['Key'])
            
            logger.info(f" Found {len(all_files)} total 2025 files")
            
            # Organize by month
            monthly_files = {}
            for file_key in all_files:
                if 'month=' in file_key:
                    month_str = file_key.split('month=')[1].split('/')[0]
                    month = int(month_str)
                    if month not in monthly_files:
                        monthly_files[month] = []
                    monthly_files[month].append(file_key)
            
            logger.info(f" 2025 files by month: {[(month, len(files)) for month, files in monthly_files.items()]}")
            return monthly_files
            
        except Exception as e:
            logger.error(f" Error discovering 2025 files: {e}")
            return {}
    
    def process_file_complete(self, file_key: str) -> Dict[str, Any]:
        """Process a single file completely - NO sampling"""
        try:
            logger.debug(f" Processing COMPLETE file: {file_key}")
            
            # Get file size
            try:
                response = self.s3_client.head_object(Bucket=S3_BUCKET, Key=file_key)
                file_size = response['ContentLength']
            except Exception as e:
                logger.warning(f" Could not get file size for {file_key}: {e}")
                file_size = 0
            
            # Download and parse file
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
            content = response['Body'].read()
            
            with io.BytesIO(content) as buffer:
                parquet_file = pq.ParquetFile(buffer)
                table = parquet_file.read(columns=None)
                df = table.to_pandas()
            
            logger.debug(f" Loaded ALL {len(df)} records from {file_key}")
            
            # Analyze the complete file
            analysis = self.analyze_file_data(df, file_key)
            analysis['file_size'] = file_size
            analysis['records_count'] = len(df)
            
            return analysis
            
        except Exception as e:
            logger.error(f" Error processing {file_key}: {e}")
            self.stats['2025']['failed_files'] += 1
            return None
    
    def analyze_file_data(self, df: pd.DataFrame, file_key: str) -> Dict[str, Any]:
        """Analyze complete file data"""
        analysis = {
            'file_key': file_key,
            'total_records': len(df)
        }
        
        # Text analysis
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'body', 'content', 'comment', 'post'])]
        if text_columns:
            text_col = text_columns[0]
            df[text_col] = df[text_col].astype(str)
            
            # Fear analysis
            fear_keywords = ['afraid', 'scared', 'worried', 'anxious', 'panic', 'terrified', 'fear', 'concern', 'nervous']
            fear_pattern = '|'.join([re.escape(keyword) for keyword in fear_keywords])
            df['has_fear'] = df[text_col].str.lower().str.contains(fear_pattern, na=False, regex=True)
            analysis["total_fear"] = int(df['has_fear'].sum())
            analysis["fear_rate"] = float(df['has_fear'].mean())
            
            # Q&A analysis
            qa_keywords = ['?', 'how', 'what', 'when', 'where', 'why', 'can i', 'should i', 'help']
            qa_pattern = '|'.join([re.escape(keyword) for keyword in qa_keywords])
            df['is_question'] = df[text_col].str.lower().str.contains(qa_pattern, na=False, regex=True)
            analysis["total_qa"] = int(df['is_question'].sum())
            analysis["qa_rate"] = float(df['is_question'].mean())
            
            # Stage analysis
            stages = {
                'F1': ['f1', 'student visa', 'student', 'f-1', 'f1 visa'],
                'OPT': ['opt', 'optional practical training', 'stem opt', 'cpt'],
                'H1B': ['h1b', 'h-1b', 'work visa'],
                'greencard': ['green card', 'greencard', 'permanent resident', 'gc', 'i-140', 'i-485'],
                'citizenship': ['citizenship', 'naturalization', 'citizen', 'n-400'],
                'general_immigration': ['immigration', 'visa', 'immigrant', 'uscis', 'immigration law']
            }
            
            stage_counts = {}
            for stage, keywords in stages.items():
                pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                stage_counts[stage] = int(df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum())
            
            analysis["stage_analysis"] = stage_counts
        
        return analysis
    
    def process_month_complete(self, month: int, files: List[str]) -> Dict[str, Any]:
        """Process ALL files for a month completely"""
        logger.info(f" Processing ALL {len(files)} files for 2025-{month:02d}")
        
        monthly_analysis = {
            'month': month,
            'total_files': len(files),
            'processed_files': 0,
            'total_records': 0,
            'total_fear': 0,
            'total_qa': 0,
            'stage_analysis': {},
            'file_analyses': []
        }
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_file_complete, file_key): file_key for file_key in files}
            
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                try:
                    analysis = future.result()
                    if analysis:
                        monthly_analysis['file_analyses'].append(analysis)
                        monthly_analysis['processed_files'] += 1
                        monthly_analysis['total_records'] += analysis['total_records']
                        monthly_analysis['total_fear'] += analysis.get('total_fear', 0)
                        monthly_analysis['total_qa'] += analysis.get('total_qa', 0)
                        
                        # Aggregate stage analysis
                        for stage, count in analysis.get('stage_analysis', {}).items():
                            if stage not in monthly_analysis['stage_analysis']:
                                monthly_analysis['stage_analysis'][stage] = 0
                            monthly_analysis['stage_analysis'][stage] += count
                        
                        self.stats['2025']['processed_files'] += 1
                        self.stats['2025']['total_records'] += analysis['total_records']
                        
                        if monthly_analysis['processed_files'] % 50 == 0:
                            logger.info(f" 2025-{month:02d}: Processed {monthly_analysis['processed_files']}/{len(files)} files")
                    
                except Exception as e:
                    logger.error(f" Error processing {file_key}: {e}")
                    self.stats['2025']['failed_files'] += 1
        
        logger.info(f" 2025-{month:02d}: Processed {monthly_analysis['processed_files']}/{len(files)} files")
        logger.info(f" Aggregating 2025-{month:02d} analysis from {monthly_analysis['processed_files']} files...")
        logger.info(f" 2025-{month:02d} aggregated: {monthly_analysis['total_records']:,} records, stages: {monthly_analysis['stage_analysis']}")
        
        return monthly_analysis
    
    def process_2025_complete(self) -> Dict[str, Any]:
        """Process ALL 2025 data completely"""
        logger.info(f" Starting COMPLETE analysis for 2025 - NO SAMPLING...")
        
        # Get ALL files for 2025
        monthly_files = self.get_all_2025_files_by_month()
        
        if not monthly_files:
            logger.error(f" No files found for 2025")
            return {}
        
        self.stats['2025']['total_files'] = sum(len(files) for files in monthly_files.values())
        
        # Process ALL files for each month
        monthly_analyses = {}
        for month in sorted(monthly_files.keys()):
            files = monthly_files[month]
            monthly_analysis = self.process_month_complete(month, files)
            monthly_analyses[f"2025-{month:02d}"] = monthly_analysis
        
        # Aggregate all monthly data
        total_analysis = {
            'year': 2025,
            'total_files': self.stats['2025']['total_files'],
            'processed_files': self.stats['2025']['processed_files'],
            'failed_files': self.stats['2025']['failed_files'],
            'total_records': self.stats['2025']['total_records'],
            'total_fear': 0,
            'total_qa': 0,
            'stage_analysis': {},
            'monthly_breakdown': monthly_analyses,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Aggregate across all months
        for month_data in monthly_analyses.values():
            total_analysis['total_fear'] += month_data['total_fear']
            total_analysis['total_qa'] += month_data['total_qa']
            
            for stage, count in month_data['stage_analysis'].items():
                if stage not in total_analysis['stage_analysis']:
                    total_analysis['stage_analysis'][stage] = 0
                total_analysis['stage_analysis'][stage] += count
        
        # Calculate rates
        if total_analysis['total_records'] > 0:
            total_analysis['fear_rate'] = total_analysis['total_fear'] / total_analysis['total_records']
            total_analysis['qa_rate'] = total_analysis['total_qa'] / total_analysis['total_records']
        else:
            total_analysis['fear_rate'] = 0
            total_analysis['qa_rate'] = 0
        
        return total_analysis
    
    def save_analysis(self, analysis: Dict[str, Any]):
        """Save analysis results"""
        os.makedirs('complete_2025_analysis', exist_ok=True)
        
        # Save complete analysis
        with open('complete_2025_analysis/complete_2025_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f" Saved complete 2025 analysis to complete_2025_analysis/complete_2025_analysis.json")
        
        # Upload to S3
        try:
            s3_key = f"{S3_RESULTS_PREFIX}complete_2025_analysis.json"
            self.s3_client.upload_file('complete_2025_analysis/complete_2025_analysis.json', S3_BUCKET, s3_key)
            logger.info(f" Uploaded complete 2025 analysis to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logger.error(f" Error uploading to S3: {e}")

def main():
    logger.info(" Starting Parallel 2025 Analysis - NO SAMPLING...")
    
    analyzer = Parallel2025Analyzer(max_workers=20)
    
    # Process 2025
    logger.info("=" * 60)
    logger.info("PROCESSING ALL 2025 DATA - NO SAMPLING (PARALLEL)")
    logger.info("=" * 60)
    start_time = time.time()
    
    analysis_2025 = analyzer.process_2025_complete()
    if analysis_2025:
        analyzer.save_analysis(analysis_2025)
        logger.info(f" 2025 Analysis Complete! Processed {analyzer.stats['2025']['total_records']:,} records")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f" Parallel 2025 Analysis Complete!")
    logger.info(f"⏱ Total time: {total_time/60:.1f} minutes")
    logger.info(f" Total records processed: {analyzer.stats['2025']['total_records']:,}")
    logger.info(f" Total files processed: {analyzer.stats['2025']['processed_files']:,}")
    logger.info(f" Failed files: {analyzer.stats['2025']['failed_files']:,}")

if __name__ == "__main__":
    main()
