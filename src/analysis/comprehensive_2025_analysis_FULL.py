#!/usr/bin/env python3
"""
Comprehensive 2025 Analysis - Full Dataset with Proper Monthly Separation
This script analyzes the complete 2025 dataset ensuring each month uses different data
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
        logging.FileHandler('comprehensive_2025_analysis_FULL.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "siva-test-9"
S3_2025_PREFIX = "reddit/new_data/2025_data/"
S3_RESULTS_PREFIX = "reddit/new_data/results/"

class Comprehensive2025Analyzer:
    """Comprehensive analyzer for 2025 data with proper monthly separation"""
    
    def __init__(self, max_workers=10):
        self.s3_client = boto3.client('s3')
        self.max_workers = max_workers
        self.monthly_data = {}
        self.stats = {
            'total_files_processed': 0,
            'total_records': 0,
            'total_bytes': 0,
            'monthly_stats': {}
        }
        random.seed(42)
        np.random.seed(42)
    
    def get_2025_files_by_month(self) -> Dict[int, List[str]]:
        """Get all 2025 files organized by month"""
        logger.info(" Discovering all 2025 files by month...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=S3_2025_PREFIX
            )
            
            all_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    all_files.append(obj['Key'])
            
            logger.info(f" Found {len(all_files)} total 2025 files")
            
            # Organize by month
            monthly_files = {}
            for file_key in all_files:
                # Extract month from path like: reddit/new_data/2025_data/comments/month=1/part-xxx.parquet
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
    
    def sample_file_for_month(self, file_key: str, sample_size: int = 3000) -> pd.DataFrame:
        """Sample a single file for a specific month"""
        try:
            logger.debug(f" Processing: {file_key}")
            
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
            
            logger.debug(f" Loaded {len(df)} records from {file_key}")
            
            # Sample if needed
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.debug(f" Sampled {len(df)} records")
            
            # Add metadata
            df['source_file'] = file_key
            df['file_size'] = file_size
            
            return df
            
        except Exception as e:
            logger.error(f" Error processing {file_key}: {e}")
            return pd.DataFrame()
    
    def process_month_independently(self, month: int, files: List[str], files_per_month: int = 40) -> Dict[str, Any]:
        """Process a specific month independently with its own data"""
        logger.info(f" Processing month {month} independently with {len(files)} files available")
        
        # Sample files for this month
        if len(files) > files_per_month:
            sampled_files = random.sample(files, files_per_month)
        else:
            sampled_files = files
        
        logger.info(f" Sampling {len(sampled_files)} files for month {month}")
        
        # Process files and collect data
        monthly_dataframes = []
        files_processed = 0
        total_bytes = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.sample_file_for_month, file_key, 3000): file_key 
                for file_key in sampled_files
            }
            
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                try:
                    df = future.result()
                    if not df.empty:
                        monthly_dataframes.append(df)
                        files_processed += 1
                        total_bytes += df['file_size'].iloc[0] if 'file_size' in df.columns else 0
                except Exception as e:
                    logger.error(f" Error processing {file_key}: {e}")
        
        if not monthly_dataframes:
            logger.warning(f" No data collected for month {month}")
            return {}
        
        # Combine monthly data
        month_df = pd.concat(monthly_dataframes, ignore_index=True)
        logger.info(f" Month {month}: {len(month_df)} records from {files_processed} files")
        
        # Store monthly data
        self.monthly_data[month] = month_df
        
        return {
            'month': month,
            'files_available': len(files),
            'files_sampled': len(sampled_files),
            'files_processed': files_processed,
            'records_analyzed': len(month_df),
            'total_bytes': total_bytes
        }
    
    def analyze_month_data(self, month: int, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data for a specific month"""
        logger.info(f" Analyzing month {month} data...")
        
        analysis = {
            "month": month,
            "total_records": len(df),
            "files_processed": len(df['source_file'].unique()) if 'source_file' in df.columns else 0
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
            
            # Stage analysis
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
            
            analysis["stage_analysis"] = stage_counts
            
            logger.info(f" Month {month} stage analysis: {stage_counts}")
        
        return analysis
    
    def create_comprehensive_analysis(self) -> Dict[str, Any]:
        """Create comprehensive analysis from all monthly data"""
        logger.info(" Creating comprehensive analysis from monthly data...")
        
        if not self.monthly_data:
            logger.error(" No monthly data available")
            return {}
        
        # Analyze each month independently
        monthly_analyses = {}
        total_records = 0
        total_fear = 0
        total_qa = 0
        combined_stage_counts = {}
        
        for month, df in self.monthly_data.items():
            month_analysis = self.analyze_month_data(month, df)
            monthly_analyses[f"2025-{month:02d}"] = month_analysis
            
            total_records += month_analysis.get("total_records", 0)
            total_fear += month_analysis.get("total_fear", 0)
            total_qa += month_analysis.get("total_qa", 0)
            
            # Combine stage counts
            stage_analysis = month_analysis.get("stage_analysis", {})
            for stage, count in stage_analysis.items():
                if stage not in combined_stage_counts:
                    combined_stage_counts[stage] = 0
                combined_stage_counts[stage] += count
        
        # Create comprehensive analysis
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "year": 2025,
            "analysis_type": "comprehensive_independent_monthly",
            "months_analyzed": list(self.monthly_data.keys()),
            "total_records": total_records,
            "total_fear": total_fear,
            "total_qa": total_qa,
            "fear_rate": total_fear / total_records if total_records > 0 else 0,
            "qa_rate": total_qa / total_records if total_records > 0 else 0,
            "stage_analysis": combined_stage_counts,
            "monthly_breakdown": monthly_analyses,
            "summary_stats": {
                "total_records": total_records,
                "total_fear": total_fear,
                "total_qa": total_qa,
                "fear_rate": total_fear / total_records if total_records > 0 else 0,
                "qa_rate": total_qa / total_records if total_records > 0 else 0,
                "months_covered": len(self.monthly_data),
                "total_files_processed": self.stats['total_files_processed']
            }
        }
        
        return analysis
    
    def save_comprehensive_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save the comprehensive analysis results"""
        logger.info(" Saving comprehensive 2025 analysis...")
        
        # Save JSON
        output_file = "comprehensive_2025_analysis_FULL.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
        logger.info(f" Saved analysis to {output_file}")
        
        # Upload to S3
        s3_key = f"{S3_RESULTS_PREFIX}comprehensive_2025_analysis_FULL.json"
        try:
            self.s3_client.upload_file(output_file, S3_BUCKET, s3_key)
            logger.info(f" Uploaded to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logger.error(f" Upload failed: {e}")

def main():
    """Main function to run comprehensive 2025 analysis"""
    logger.info(" Starting Comprehensive 2025 Analysis with Independent Monthly Data...")
    
    analyzer = Comprehensive2025Analyzer(max_workers=10)
    
    # Get all files organized by month
    monthly_files = analyzer.get_2025_files_by_month()
    
    if not monthly_files:
        logger.error(" No files found")
        return
    
    # Process each month independently
    monthly_stats = {}
    for month in sorted(monthly_files.keys()):
        files = monthly_files[month]
        month_stats = analyzer.process_month_independently(month, files, files_per_month=30)
        monthly_stats[month] = month_stats
        analyzer.stats['total_files_processed'] += month_stats.get('files_processed', 0)
        analyzer.stats['total_records'] += month_stats.get('records_analyzed', 0)
    
    # Create comprehensive analysis
    analysis = analyzer.create_comprehensive_analysis()
    
    # Add processing stats
    analysis["processing_stats"] = monthly_stats
    
    # Save results
    analyzer.save_comprehensive_analysis(analysis)
    
    logger.info(" Comprehensive 2025 Analysis Complete!")
    logger.info(f" Processed {analyzer.stats['total_files_processed']} files")
    logger.info(f" Total records analyzed: {analyzer.stats['total_records']}")
    logger.info(f" Months analyzed: {list(monthly_files.keys())}")

if __name__ == "__main__":
    main()
