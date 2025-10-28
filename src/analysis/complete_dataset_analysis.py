#!/usr/bin/env python3
"""
Complete Dataset Analysis - ALL 2024 and 2025 Data
This script processes the complete datasets for both years with proper sampling
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
        logging.FileHandler('complete_dataset_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "siva-test-9"
S3_2024_PREFIX = "reddit/new_data/2024_data/"
S3_2025_PREFIX = "reddit/new_data/2025_data/"
S3_RESULTS_PREFIX = "reddit/new_data/results/"

class CompleteDatasetAnalyzer:
    """Analyzer for complete datasets with comprehensive sampling"""
    
    def __init__(self, max_workers=20):
        self.s3_client = boto3.client('s3')
        self.max_workers = max_workers
        self.stats = {
            '2024': {'files_processed': 0, 'total_records': 0, 'total_bytes': 0},
            '2025': {'files_processed': 0, 'total_records': 0, 'total_bytes': 0}
        }
        random.seed(42)
        np.random.seed(42)
    
    def get_all_files_by_year_month(self, year: int) -> Dict[int, List[str]]:
        """Get all files for a year organized by month"""
        prefix = S3_2024_PREFIX if year == 2024 else S3_2025_PREFIX
        logger.info(f" Discovering all {year} files...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=prefix
            )
            
            all_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    all_files.append(obj['Key'])
            
            logger.info(f" Found {len(all_files)} total {year} files")
            
            # Organize by month
            monthly_files = {}
            for file_key in all_files:
                if 'month=' in file_key:
                    month_str = file_key.split('month=')[1].split('/')[0]
                    month = int(month_str)
                    if month not in monthly_files:
                        monthly_files[month] = []
                    monthly_files[month].append(file_key)
            
            logger.info(f" {year} files by month: {[(month, len(files)) for month, files in monthly_files.items()]}")
            return monthly_files
            
        except Exception as e:
            logger.error(f" Error discovering {year} files: {e}")
            return {}
    
    def sample_file_comprehensive(self, file_key: str, sample_size: int = 5000) -> pd.DataFrame:
        """Sample a single file with comprehensive error handling"""
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
    
    def process_year_comprehensive(self, year: int, files_per_month: int = 100) -> Dict[str, Any]:
        """Process a complete year with comprehensive sampling"""
        logger.info(f" Starting comprehensive analysis for {year}...")
        
        # Get all files for the year
        monthly_files = self.get_all_files_by_year_month(year)
        
        if not monthly_files:
            logger.error(f" No files found for {year}")
            return {}
        
        # Process each month
        monthly_data = {}
        monthly_analyses = {}
        
        for month in sorted(monthly_files.keys()):
            files = monthly_files[month]
            logger.info(f" Processing {year}-{month:02d} with {len(files)} files available")
            
            # Sample files for this month
            if len(files) > files_per_month:
                sampled_files = random.sample(files, files_per_month)
            else:
                sampled_files = files
            
            logger.info(f" Sampling {len(sampled_files)} files for {year}-{month:02d}")
            
            # Process files and collect data
            monthly_dataframes = []
            files_processed = 0
            total_bytes = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.sample_file_comprehensive, file_key, 5000): file_key 
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
                logger.warning(f" No data collected for {year}-{month:02d}")
                continue
            
            # Combine monthly data
            month_df = pd.concat(monthly_dataframes, ignore_index=True)
            logger.info(f" {year}-{month:02d}: {len(month_df)} records from {files_processed} files")
            
            # Analyze monthly data
            month_analysis = self.analyze_month_data(year, month, month_df)
            monthly_analyses[f"{year}-{month:02d}"] = month_analysis
            
            # Update stats
            self.stats[str(year)]['files_processed'] += files_processed
            self.stats[str(year)]['total_records'] += len(month_df)
            self.stats[str(year)]['total_bytes'] += total_bytes
            
            monthly_data[month] = month_df
        
        # Create comprehensive analysis
        analysis = self.create_comprehensive_analysis(year, monthly_analyses)
        
        return analysis
    
    def analyze_month_data(self, year: int, month: int, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data for a specific month"""
        logger.info(f" Analyzing {year}-{month:02d} data...")
        
        analysis = {
            "year": year,
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
            
            # Comprehensive stage analysis
            stages = {
                'F1': ['f1', 'student visa', 'student', 'f-1', 'f1 visa', 'student status', 'f1 status'],
                'OPT': ['opt', 'optional practical training', 'stem opt', 'cpt', 'work authorization', 'opt status'],
                'H1B': ['h1b', 'h-1b', 'work visa', 'h1-b', 'h1b visa', 'employment visa', 'h1b status'],
                'greencard': ['green card', 'greencard', 'permanent resident', 'gc', 'i-140', 'i-485', 'permanent residency'],
                'citizenship': ['citizenship', 'naturalization', 'citizen', 'n-400', 'naturalized', 'citizen status'],
                'general_immigration': ['immigration', 'visa', 'immigrant', 'uscis', 'immigration law', 'immigration status']
            }
            
            stage_counts = {}
            for stage, keywords in stages.items():
                pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                count = df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
                stage_counts[stage] = int(count)
            
            analysis["stage_analysis"] = stage_counts
            
            logger.info(f" {year}-{month:02d} stage analysis: {stage_counts}")
        
        return analysis
    
    def create_comprehensive_analysis(self, year: int, monthly_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive analysis from all monthly data"""
        logger.info(f" Creating comprehensive analysis for {year}...")
        
        if not monthly_analyses:
            logger.error(f" No monthly data available for {year}")
            return {}
        
        # Aggregate all monthly data
        total_records = sum(analysis.get("total_records", 0) for analysis in monthly_analyses.values())
        total_fear = sum(analysis.get("total_fear", 0) for analysis in monthly_analyses.values())
        total_qa = sum(analysis.get("total_qa", 0) for analysis in monthly_analyses.values())
        
        # Combine stage counts
        combined_stage_counts = {}
        for month_analysis in monthly_analyses.values():
            stage_analysis = month_analysis.get("stage_analysis", {})
            for stage, count in stage_analysis.items():
                if stage not in combined_stage_counts:
                    combined_stage_counts[stage] = 0
                combined_stage_counts[stage] += count
        
        # Create comprehensive analysis
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "year": year,
            "analysis_type": "complete_dataset_comprehensive",
            "months_analyzed": list(monthly_analyses.keys()),
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
                "months_covered": len(monthly_analyses),
                "total_files_processed": self.stats[str(year)]['files_processed'],
                "total_bytes_processed": self.stats[str(year)]['total_bytes']
            }
        }
        
        return analysis
    
    def save_analysis(self, year: int, analysis: Dict[str, Any]) -> None:
        """Save the analysis results"""
        logger.info(f" Saving comprehensive {year} analysis...")
        
        # Save JSON
        output_file = f"complete_{year}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
        logger.info(f" Saved analysis to {output_file}")
        
        # Upload to S3
        s3_key = f"{S3_RESULTS_PREFIX}complete_{year}_analysis.json"
        try:
            self.s3_client.upload_file(output_file, S3_BUCKET, s3_key)
            logger.info(f" Uploaded to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logger.error(f" Upload failed: {e}")

def main():
    """Main function to run complete dataset analysis"""
    logger.info(" Starting Complete Dataset Analysis for 2024 and 2025...")
    
    analyzer = CompleteDatasetAnalyzer(max_workers=20)
    
    # Process 2024
    logger.info("=" * 60)
    logger.info("PROCESSING 2024 DATA")
    logger.info("=" * 60)
    start_time = time.time()
    
    analysis_2024 = analyzer.process_year_comprehensive(2024, files_per_month=150)
    if analysis_2024:
        analyzer.save_analysis(2024, analysis_2024)
        logger.info(f" 2024 Analysis Complete! Processed {analyzer.stats['2024']['total_records']:,} records")
    
    # Process 2025
    logger.info("=" * 60)
    logger.info("PROCESSING 2025 DATA")
    logger.info("=" * 60)
    
    analysis_2025 = analyzer.process_year_comprehensive(2025, files_per_month=150)
    if analysis_2025:
        analyzer.save_analysis(2025, analysis_2025)
        logger.info(f" 2025 Analysis Complete! Processed {analyzer.stats['2025']['total_records']:,} records")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 60)
    logger.info(f" 2024: {analyzer.stats['2024']['total_records']:,} records from {analyzer.stats['2024']['files_processed']} files")
    logger.info(f" 2025: {analyzer.stats['2025']['total_records']:,} records from {analyzer.stats['2025']['files_processed']} files")
    logger.info(f"⏱ Total time: {total_time/60:.1f} minutes")
    logger.info(" Complete Dataset Analysis Finished!")

if __name__ == "__main__":
    main()
