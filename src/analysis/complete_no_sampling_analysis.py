#!/usr/bin/env python3
"""
Complete Dataset Analysis - NO SAMPLING
This script analyzes the ENTIRE dataset for both 2024 and 2025 without any sampling
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
        logging.FileHandler('complete_no_sampling_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "siva-test-9"
S3_2024_PREFIX = "reddit/new_data/2024_data/"
S3_2025_PREFIX = "reddit/new_data/2025_data/"
S3_RESULTS_PREFIX = "reddit/new_data/results/"

class CompleteNoSamplingAnalyzer:
    """Analyzer for complete datasets with NO sampling - processes ALL data"""
    
    def __init__(self, max_workers=30):
        self.s3_client = boto3.client('s3')
        self.max_workers = max_workers
        self.stats = {
            '2024': {'files_processed': 0, 'total_records': 0, 'total_bytes': 0},
            '2025': {'files_processed': 0, 'total_records': 0, 'total_bytes': 0}
        }
        random.seed(42)
        np.random.seed(42)
    
    def get_all_files_by_year_month(self, year: int) -> Dict[int, List[str]]:
        """Get ALL files for a year organized by month - handles pagination"""
        prefix = S3_2024_PREFIX if year == 2024 else S3_2025_PREFIX
        logger.info(f" Discovering ALL {year} files with pagination...")
        
        try:
            all_files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=S3_BUCKET,
                Prefix=prefix
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
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
            return {'error': str(e), 'file_key': file_key}
    
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
        
        return analysis
    
    def process_year_complete(self, year: int) -> Dict[str, Any]:
        """Process a complete year with NO sampling - ALL files"""
        logger.info(f" Starting COMPLETE analysis for {year} - NO SAMPLING...")
        
        # Get ALL files for the year
        monthly_files = self.get_all_files_by_year_month(year)
        
        if not monthly_files:
            logger.error(f" No files found for {year}")
            return {}
        
        # Process ALL files for each month
        monthly_analyses = {}
        
        for month in sorted(monthly_files.keys()):
            files = monthly_files[month]
            logger.info(f" Processing ALL {len(files)} files for {year}-{month:02d}")
            
            # Process ALL files in parallel
            file_analyses = []
            files_processed = 0
            total_bytes = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file_complete, file_key): file_key 
                    for file_key in files
                }
                
                for future in as_completed(future_to_file):
                    file_key = future_to_file[future]
                    try:
                        analysis = future.result()
                        if 'error' not in analysis:
                            file_analyses.append(analysis)
                            files_processed += 1
                            total_bytes += analysis.get('file_size', 0)
                    except Exception as e:
                        logger.error(f" Error processing {file_key}: {e}")
            
            logger.info(f" {year}-{month:02d}: Processed {files_processed}/{len(files)} files")
            
            # Aggregate monthly analysis
            if file_analyses:
                month_analysis = self.aggregate_month_analysis(year, month, file_analyses)
                monthly_analyses[f"{year}-{month:02d}"] = month_analysis
                
                # Update stats
                self.stats[str(year)]['files_processed'] += files_processed
                self.stats[str(year)]['total_records'] += sum(a.get('total_records', 0) for a in file_analyses)
                self.stats[str(year)]['total_bytes'] += total_bytes
        
        # Create comprehensive analysis
        analysis = self.create_comprehensive_analysis(year, monthly_analyses)
        
        return analysis
    
    def aggregate_month_analysis(self, year: int, month: int, file_analyses: List[Dict]) -> Dict[str, Any]:
        """Aggregate analysis from all files in a month"""
        logger.info(f" Aggregating {year}-{month:02d} analysis from {len(file_analyses)} files...")
        
        # Sum all metrics
        total_records = sum(a.get('total_records', 0) for a in file_analyses)
        total_fear = sum(a.get('total_fear', 0) for a in file_analyses)
        total_qa = sum(a.get('total_qa', 0) for a in file_analyses)
        
        # Aggregate stage counts
        combined_stage_counts = {}
        for analysis in file_analyses:
            stage_analysis = analysis.get("stage_analysis", {})
            for stage, count in stage_analysis.items():
                if stage not in combined_stage_counts:
                    combined_stage_counts[stage] = 0
                combined_stage_counts[stage] += count
        
        month_analysis = {
            "year": year,
            "month": month,
            "total_records": total_records,
            "total_fear": total_fear,
            "total_qa": total_qa,
            "fear_rate": total_fear / total_records if total_records > 0 else 0,
            "qa_rate": total_qa / total_records if total_records > 0 else 0,
            "stage_analysis": combined_stage_counts,
            "files_processed": len(file_analyses)
        }
        
        logger.info(f" {year}-{month:02d} aggregated: {total_records:,} records, stages: {combined_stage_counts}")
        
        return month_analysis
    
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
            "analysis_type": "complete_dataset_NO_SAMPLING",
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
        logger.info(f" Saving complete {year} analysis...")
        
        # Save JSON
        output_file = f"complete_{year}_analysis_NO_SAMPLING.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
        logger.info(f" Saved analysis to {output_file}")
        
        # Upload to S3
        s3_key = f"{S3_RESULTS_PREFIX}complete_{year}_analysis_NO_SAMPLING.json"
        try:
            self.s3_client.upload_file(output_file, S3_BUCKET, s3_key)
            logger.info(f" Uploaded to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logger.error(f" Upload failed: {e}")

def main():
    """Main function to run complete dataset analysis with NO sampling"""
    logger.info(" Starting Complete Dataset Analysis - NO SAMPLING for 2024 and 2025...")
    
    analyzer = CompleteNoSamplingAnalyzer(max_workers=30)
    
    # Process 2024
    logger.info("=" * 60)
    logger.info("PROCESSING ALL 2024 DATA - NO SAMPLING")
    logger.info("=" * 60)
    start_time = time.time()
    
    analysis_2024 = analyzer.process_year_complete(2024)
    if analysis_2024:
        analyzer.save_analysis(2024, analysis_2024)
        logger.info(f" 2024 Analysis Complete! Processed {analyzer.stats['2024']['total_records']:,} records")
    
    # Process 2025
    logger.info("=" * 60)
    logger.info("PROCESSING ALL 2025 DATA - NO SAMPLING")
    logger.info("=" * 60)
    
    analysis_2025 = analyzer.process_year_complete(2025)
    if analysis_2025:
        analyzer.save_analysis(2025, analysis_2025)
        logger.info(f" 2025 Analysis Complete! Processed {analyzer.stats['2025']['total_records']:,} records")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE ANALYSIS FINISHED!")
    logger.info("=" * 60)
    logger.info(f" 2024: {analyzer.stats['2024']['total_records']:,} records from {analyzer.stats['2024']['files_processed']} files")
    logger.info(f" 2025: {analyzer.stats['2025']['total_records']:,} records from {analyzer.stats['2025']['files_processed']} files")
    logger.info(f"⏱ Total time: {total_time/60:.1f} minutes")
    logger.info(" Complete Dataset Analysis - NO SAMPLING Finished!")

if __name__ == "__main__":
    main()
