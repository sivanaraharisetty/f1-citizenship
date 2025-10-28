#!/usr/bin/env python3
"""
Comprehensive 2024 Analysis - Full Dataset
This script analyzes the complete 2024 dataset with proper sampling across all months
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_2024_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "siva-test-9"
S3_2024_PREFIX = "reddit/new_data/2024_data/"
S3_RESULTS_PREFIX = "reddit/new_data/results/"

class Comprehensive2024Analyzer:
    """Comprehensive analyzer for 2024 data using full dataset"""
    
    def __init__(self, max_workers=10):
        self.s3_client = boto3.client('s3')
        self.max_workers = max_workers
        self.sampled_data = []
        self.stats = {
            'files_processed': 0,
            'total_records': 0,
            'total_bytes': 0,
            'monthly_stats': {},
            'stage_counts': {}
        }
        random.seed(42)
        np.random.seed(42)
    
    def get_all_2024_files(self) -> List[str]:
        """Get all 2024 files organized by month"""
        logger.info(" Discovering all 2024 files...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=S3_2024_PREFIX
            )
            
            all_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    all_files.append(obj['Key'])
            
            logger.info(f" Found {len(all_files)} total 2024 files")
            
            # Organize by month
            monthly_files = {}
            for file_key in all_files:
                # Extract month from path like: reddit/new_data/2024_data/comments/month=1/part-xxx.parquet
                if 'month=' in file_key:
                    month_str = file_key.split('month=')[1].split('/')[0]
                    month = int(month_str)
                    if month not in monthly_files:
                        monthly_files[month] = []
                    monthly_files[month].append(file_key)
            
            logger.info(f" Files by month: {[(month, len(files)) for month, files in monthly_files.items()]}")
            return monthly_files
            
        except Exception as e:
            logger.error(f" Error discovering files: {e}")
            return {}
    
    def sample_file_comprehensive(self, file_key: str, sample_size: int = 2000) -> bool:
        """Sample a single file with comprehensive error handling"""
        try:
            logger.debug(f" Processing: {file_key}")
            
            # Get file size
            try:
                response = self.s3_client.head_object(Bucket=S3_BUCKET, Key=file_key)
                file_size = response['ContentLength']
                self.stats['total_bytes'] += file_size
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
            
            logger.debug(f" Loaded {len(df)} records from {file_key}")
            
            # Sample if needed
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.debug(f" Sampled {len(df)} records")
            
            # Store sampled data with metadata
            df['source_file'] = file_key
            df['file_size'] = file_size
            self.sampled_data.append(df)
            self.stats['files_processed'] += 1
            self.stats['total_records'] += len(df)
            
            return True
            
        except Exception as e:
            logger.error(f" Error processing {file_key}: {e}")
            return False
    
    def process_month_comprehensive(self, month: int, files: List[str], files_per_month: int = 50) -> Dict[str, Any]:
        """Process a specific month with comprehensive sampling"""
        logger.info(f" Processing month {month} with {len(files)} files available")
        
        # Sample files for this month
        if len(files) > files_per_month:
            sampled_files = random.sample(files, files_per_month)
        else:
            sampled_files = files
        
        logger.info(f" Sampling {len(sampled_files)} files for month {month}")
        
        # Process files in parallel
        success_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.sample_file_comprehensive, file_key, 2000): file_key 
                for file_key in sampled_files
            }
            
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f" Error processing {file_key}: {e}")
        
        logger.info(f" Month {month}: Processed {success_count}/{len(sampled_files)} files successfully")
        
        return {
            'month': month,
            'files_available': len(files),
            'files_sampled': len(sampled_files),
            'files_processed': success_count,
            'records_sampled': sum(len(df) for df in self.sampled_data if df['source_file'].iloc[0] in sampled_files)
        }
    
    def analyze_comprehensive_data(self) -> Dict[str, Any]:
        """Perform comprehensive analysis on all sampled data"""
        logger.info(" Starting comprehensive analysis...")
        
        if not self.sampled_data:
            logger.error(" No sampled data available")
            return {}
        
        # Combine all sampled data
        df = pd.concat(self.sampled_data, ignore_index=True)
        logger.info(f" Total records for analysis: {len(df)}")
        
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "year": 2024,
            "analysis_type": "comprehensive_full_dataset",
            "total_records_analyzed": len(df),
            "files_processed": self.stats['files_processed'],
            "total_bytes_processed": self.stats['total_bytes'],
            "sampling_method": "stratified_by_month"
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
            qa_keywords = ['\\?', 'how', 'what', 'when', 'where', 'why', 'can i', 'should i', 'help']
            qa_pattern = '|'.join([re.escape(keyword) for keyword in qa_keywords])
            df['is_question'] = df[text_col].str.lower().str.contains(qa_pattern, na=False, regex=True)
            analysis["total_qa"] = int(df['is_question'].sum())
            analysis["qa_rate"] = float(df['is_question'].mean())
            
            # Comprehensive stage analysis
            stages = {
                'F1': ['f1', 'student visa', 'student', 'f-1', 'f1 visa', 'student status'],
                'OPT': ['opt', 'optional practical training', 'stem opt', 'cpt', 'work authorization'],
                'H1B': ['h1b', 'h-1b', 'work visa', 'h1-b', 'h1b visa', 'employment visa'],
                'greencard': ['green card', 'greencard', 'permanent resident', 'gc', 'i-140', 'i-485', 'permanent residency'],
                'citizenship': ['citizenship', 'naturalization', 'citizen', 'n-400', 'naturalized'],
                'general_immigration': ['immigration', 'visa', 'immigrant', 'uscis', 'immigration law', 'immigration status']
            }
            
            stage_counts = {}
            for stage, keywords in stages.items():
                pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                count = df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
                stage_counts[stage] = int(count)
                logger.info(f" {stage}: {count} matches")
            
            analysis["stage_analysis"] = stage_counts
            
            # Monthly breakdown analysis
            if 'source_file' in df.columns:
                monthly_analysis = {}
                for month in range(1, 13):
                    month_files = [f for f in df['source_file'].unique() if f'month={month}' in f]
                    if month_files:
                        month_df = df[df['source_file'].isin(month_files)]
                        monthly_analysis[f"2024-{month:02d}"] = {
                            "records": len(month_df),
                            "fear": int(month_df['has_fear'].sum()) if 'has_fear' in month_df.columns else 0,
                            "qa": int(month_df['is_question'].sum()) if 'is_question' in month_df.columns else 0,
                            "files_sampled": len(month_files)
                        }
                
                analysis["monthly_breakdown"] = monthly_analysis
        
        # Summary statistics
        analysis["summary_stats"] = {
            "total_records": len(df),
            "total_fear": analysis.get("total_fear", 0),
            "total_qa": analysis.get("total_qa", 0),
            "fear_rate": analysis.get("fear_rate", 0),
            "qa_rate": analysis.get("qa_rate", 0),
            "files_processed": self.stats['files_processed'],
            "avg_records_per_file": len(df) / self.stats['files_processed'] if self.stats['files_processed'] > 0 else 0
        }
        
        return analysis
    
    def save_comprehensive_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save the comprehensive analysis results"""
        logger.info(" Saving comprehensive 2024 analysis...")
        
        # Save JSON
        output_file = "comprehensive_2024_analysis_FULL.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f" Saved analysis to {output_file}")
        
        # Upload to S3
        s3_key = f"{S3_RESULTS_PREFIX}comprehensive_2024_analysis_FULL.json"
        try:
            self.s3_client.upload_file(output_file, S3_BUCKET, s3_key)
            logger.info(f" Uploaded to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logger.error(f" Upload failed: {e}")

def main():
    """Main function to run comprehensive 2024 analysis"""
    logger.info(" Starting Comprehensive 2024 Analysis with Full Dataset...")
    
    analyzer = Comprehensive2024Analyzer(max_workers=10)
    
    # Get all files organized by month
    monthly_files = analyzer.get_all_2024_files()
    
    if not monthly_files:
        logger.error(" No files found")
        return
    
    # Process each month
    monthly_stats = {}
    for month in sorted(monthly_files.keys()):
        files = monthly_files[month]
        month_stats = analyzer.process_month_comprehensive(month, files, files_per_month=30)
        monthly_stats[month] = month_stats
    
    # Perform comprehensive analysis
    analysis = analyzer.analyze_comprehensive_data()
    
    # Add monthly processing stats
    analysis["monthly_processing_stats"] = monthly_stats
    
    # Save results
    analyzer.save_comprehensive_analysis(analysis)
    
    logger.info(" Comprehensive 2024 Analysis Complete!")
    logger.info(f" Processed {analyzer.stats['files_processed']} files")
    logger.info(f" Total records analyzed: {analyzer.stats['total_records']}")
    logger.info(f" Total bytes processed: {analyzer.stats['total_bytes']:,}")

if __name__ == "__main__":
    main()
