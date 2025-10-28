#!/usr/bin/env python3
"""
Comprehensive 6-Month 2025 Reddit Visa Discourse Analysis Pipeline
Processes all 6 months (January-June 2025) and generates complete analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import boto3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
import io
import re

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('six_month_2025_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SixMonthReddit2025Analyzer:
    """Comprehensive analyzer for all 6 months of 2025 Reddit visa discourse data"""
    
    def __init__(self, s3_bucket: str = "siva-test-9", s3_prefix: str = "reddit/new_data/2025_data/"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        self.results = {}
        self.sample_data = []
        self.monthly_data = {}
        self.daily_data = {}
        
        # Create output directories
        os.makedirs('six_month_2025_analysis', exist_ok=True)
        os.makedirs('six_month_2025_analysis/plots', exist_ok=True)
        os.makedirs('six_month_2025_analysis/daily_csvs', exist_ok=True)
        os.makedirs('six_month_2025_analysis/monthly_csvs', exist_ok=True)
        
        logger.info(" Initialized Six-Month Reddit 2025 Analyzer")
    
    def discover_files_by_month(self) -> Dict[int, List[str]]:
        """Discover all parquet files organized by month"""
        logger.info(" Discovering 2025 data files by month...")
        
        monthly_files = {}
        
        for month in range(1, 7):  # January to June
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"{self.s3_prefix}comments/month={month}/"
                )
                
                files = []
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.parquet'):
                        files.append(obj['Key'])
                
                monthly_files[month] = files
                logger.info(f" Found {len(files)} files for month {month}")
                
            except Exception as e:
                logger.error(f" Error discovering files for month {month}: {e}")
                monthly_files[month] = []
        
        total_files = sum(len(files) for files in monthly_files.values())
        logger.info(f" Total files across all months: {total_files}")
        
        return monthly_files
    
    def sample_file_comprehensive(self, file_key: str, sample_size: int = 1500) -> Dict[str, Any]:
        """Sample data from a single parquet file with comprehensive metadata"""
        try:
            logger.info(f" Sampling: {file_key}")
            
            # Download file
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=file_key)
            content = response['Body'].read()
            
            # Parse parquet
            with io.BytesIO(content) as buffer:
                df = pd.read_parquet(buffer)
            
            logger.info(f" Loaded {len(df)} rows from {file_key}")
            
            if len(df) == 0:
                return {"success": False, "data": [], "file": file_key}
            
            # Sample data
            if len(df) > sample_size:
                sample_df = df.sample(n=sample_size, random_state=42)
            else:
                sample_df = df
            
            # Add comprehensive metadata
            sample_df['_source_file'] = file_key
            sample_df['_sample_size'] = len(sample_df)
            sample_df['_total_rows'] = len(df)
            sample_df['_file_size'] = len(content)
            
            # Extract month from file path
            month_match = re.search(r'month=(\d+)', file_key)
            if month_match:
                sample_df['_month'] = int(month_match.group(1))
            
            return {
                "success": True,
                "data": sample_df.to_dict('records'),
                "file": file_key,
                "total_rows": len(df),
                "sampled_rows": len(sample_df),
                "file_size": len(content),
                "month": int(month_match.group(1)) if month_match else None
            }
            
        except Exception as e:
            logger.error(f" Error sampling {file_key}: {e}")
            return {"success": False, "data": [], "file": file_key, "error": str(e)}
    
    def process_month_comprehensive(self, month: int, file_keys: List[str], max_workers: int = 20, sample_size: int = 1500) -> Dict[str, Any]:
        """Process all files for a specific month"""
        logger.info(f" Processing month {month} with {len(file_keys)} files...")
        
        successful_files = 0
        failed_files = 0
        total_records = 0
        total_bytes = 0
        month_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for this month
            future_to_file = {
                executor.submit(self.sample_file_comprehensive, file_key, sample_size): file_key 
                for file_key in file_keys
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result["success"]:
                        successful_files += 1
                        total_records += len(result["data"])
                        total_bytes += result.get("file_size", 0)
                        month_data.extend(result["data"])
                        logger.info(f" Month {month} - {file_key}: {len(result['data'])} records")
                    else:
                        failed_files += 1
                        logger.warning(f" Month {month} - {file_key}: Failed")
                    
                    # Progress update
                    processed = successful_files + failed_files
                    if processed % 50 == 0:
                        logger.info(f" Month {month} Progress: {processed}/{len(future_to_file)} "
                                  f"({processed/len(future_to_file)*100:.1f}%) - "
                                  f"Records: {total_records:,}")
                
                except Exception as e:
                    failed_files += 1
                    logger.error(f" Month {month} - {file_key}: Exception: {e}")
        
        logger.info(f" Month {month} completed: {successful_files} successful, {failed_files} failed")
        logger.info(f" Month {month} total records: {total_records:,}")
        logger.info(f" Month {month} total bytes: {total_bytes:,}")
        
        return {
            "month": month,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_records": total_records,
            "total_bytes": total_bytes,
            "data": month_data
        }
    
    def process_all_months(self, monthly_files: Dict[int, List[str]]) -> None:
        """Process all 6 months of data"""
        logger.info(" Starting comprehensive 6-month processing...")
        
        for month in range(1, 7):
            if month in monthly_files and monthly_files[month]:
                month_result = self.process_month_comprehensive(
                    month, 
                    monthly_files[month], 
                    max_workers=20, 
                    sample_size=1500
                )
                
                self.monthly_data[month] = month_result
                self.sample_data.extend(month_result["data"])
                
                logger.info(f" Month {month} processing completed")
            else:
                logger.warning(f" No files found for month {month}")
        
        total_records = sum(data["total_records"] for data in self.monthly_data.values())
        total_files = sum(data["successful_files"] for data in self.monthly_data.values())
        
        logger.info(f" All 6 months processing completed!")
        logger.info(f" Total records across all months: {total_records:,}")
        logger.info(f" Total files processed: {total_files}")
    
    def analyze_data_comprehensive(self) -> Dict[str, Any]:
        """Perform comprehensive analysis for all 6 months"""
        logger.info(" Starting comprehensive 6-month analysis...")
        
        if not self.sample_data:
            logger.error(" No data to analyze")
            return {}
        
        df = pd.DataFrame(self.sample_data)
        logger.info(f" Analyzing {len(df):,} records across 6 months")
        
        # Basic stats
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "year": 2025,
            "months_analyzed": list(range(1, 7)),
            "total_records": len(df),
            "sample_files": len(df['_source_file'].unique()) if '_source_file' in df.columns else 0
        }
        
        # Monthly breakdown
        monthly_stats = {}
        for month in range(1, 7):
            if month in self.monthly_data:
                month_data = self.monthly_data[month]
                monthly_stats[f"2025-{month:02d}"] = {
                    "total_records": month_data["total_records"],
                    "files_processed": month_data["successful_files"],
                    "total_bytes": month_data["total_bytes"],
                    "avg_records_per_file": month_data["total_records"] / max(month_data["successful_files"], 1)
                }
        
        analysis["monthly_breakdown"] = monthly_stats
        
        # Date analysis (if date columns exist)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            try:
                df['date'] = pd.to_datetime(df[date_columns[0]], errors='coerce')
                df = df.dropna(subset=['date'])
                
                analysis["date_range"] = {
                    "start": df['date'].min().isoformat(),
                    "end": df['date'].max().isoformat()
                }
                analysis["total_days"] = (df['date'].max() - df['date'].min()).days + 1
                
                # Daily analysis for CSV generation
                df['date_str'] = df['date'].dt.date
                daily_stats = df.groupby('date_str').agg({
                    'date': 'count',
                    '_source_file': 'nunique',
                    '_month': 'first'
                }).rename(columns={'date': 'total', '_source_file': 'files'})
                
                self.daily_data = daily_stats.to_dict('index')
                
                # Monthly analysis with actual dates
                df['month'] = df['date'].dt.to_period('M')
                monthly_date_stats = df.groupby('month').agg({
                    'date': 'count',
                    '_source_file': 'nunique'
                }).rename(columns={'date': 'total', '_source_file': 'files'})
                
                # Convert Period keys to strings for JSON serialization
                analysis["monthly_analysis"] = {str(k): v.to_dict() for k, v in monthly_date_stats.iterrows()}
                
            except Exception as e:
                logger.warning(f" Date analysis failed: {e}")
        
        # Text analysis (if text columns exist)
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'body', 'content', 'comment', 'post'])]
        if text_columns:
            text_col = text_columns[0]
            df[text_col] = df[text_col].astype(str)
            
            # Fear analysis
            fear_keywords = ['afraid', 'scared', 'worried', 'anxious', 'panic', 'terrified', 'fear', 'concern', 'nervous']
            fear_pattern = '|'.join([re.escape(keyword) for keyword in fear_keywords])
            df['has_fear'] = df[text_col].str.lower().str.contains(fear_pattern, na=False, regex=True)
            analysis["total_fear"] = df['has_fear'].sum()
            analysis["fear_rate"] = df['has_fear'].mean()
            
            # Q&A analysis
            qa_keywords = ['\\?', 'how', 'what', 'when', 'where', 'why', 'can i', 'should i', 'help']
            qa_pattern = '|'.join([re.escape(keyword) for keyword in qa_keywords])
            df['is_question'] = df[text_col].str.lower().str.contains(qa_pattern, na=False, regex=True)
            analysis["total_qa"] = df['is_question'].sum()
            analysis["qa_rate"] = df['is_question'].mean()
            
            # Stage analysis
            stages = {
                'F1': ['f1', 'student visa', 'student'],
                'OPT': ['opt', 'optional practical training'],
                'H1B': ['h1b', 'h-1b', 'work visa'],
                'greencard': ['green card', 'greencard', 'permanent resident'],
                'citizenship': ['citizenship', 'naturalization', 'citizen'],
                'general_immigration': ['immigration', 'visa', 'immigrant']
            }
            
            stage_counts = {}
            for stage, keywords in stages.items():
                pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                stage_counts[stage] = df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
            
            analysis["stage_analysis"] = stage_counts
            
            # Monthly stage analysis
            monthly_stage_analysis = {}
            for month in range(1, 7):
                month_df = df[df['_month'] == month] if '_month' in df.columns else df
                if len(month_df) > 0:
                    month_stages = {}
                    for stage, keywords in stages.items():
                        pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                        month_stages[stage] = month_df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
                    monthly_stage_analysis[f"2025-{month:02d}"] = month_stages
            
            analysis["monthly_stage_analysis"] = monthly_stage_analysis
        
        # Summary stats
        analysis["summary_stats"] = {
            "total_records": len(df),
            "total_fear": analysis.get("total_fear", 0),
            "total_qa": analysis.get("total_qa", 0),
            "fear_rate": analysis.get("fear_rate", 0),
            "qa_rate": analysis.get("qa_rate", 0),
            "avg_records_per_file": len(df) / max(analysis.get("sample_files", 1), 1),
            "months_covered": len(monthly_stats)
        }
        
        logger.info(" Comprehensive 6-month analysis completed")
        return analysis
    
    def create_comprehensive_visualizations(self, analysis: Dict[str, Any]) -> None:
        """Create all visualizations for 6-month analysis"""
        logger.info(" Creating comprehensive 6-month visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Comprehensive Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('2025 Reddit Visa Discourse Analysis - 6 Months (Jan-June)', fontsize=16, fontweight='bold')
        
        # Summary stats
        stats = analysis.get("summary_stats", {})
        axes[0, 0].bar(['Total Records', 'Fear Cases', 'Q&A Cases'], 
                      [stats.get('total_records', 0), stats.get('total_fear', 0), stats.get('total_qa', 0)])
        axes[0, 0].set_title('Overall Statistics (6 Months)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Fear vs Q&A rates
        axes[0, 1].pie([stats.get('fear_rate', 0), 1 - stats.get('fear_rate', 0)], 
                      labels=['Fear', 'Non-Fear'], autopct='%1.1f%%')
        axes[0, 1].set_title('Fear Rate Distribution')
        
        # Monthly trends
        monthly_data = analysis.get("monthly_breakdown", {})
        if monthly_data:
            months = list(monthly_data.keys())
            totals = [monthly_data[month]['total_records'] for month in months]
            axes[0, 2].plot(range(len(months)), totals, marker='o', linewidth=2, markersize=8)
            axes[0, 2].set_title('Monthly Activity Trends')
            axes[0, 2].set_xticks(range(len(months)))
            axes[0, 2].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[0, 2].set_ylabel('Records')
        
        # Stage analysis
        stage_data = analysis.get("stage_analysis", {})
        if stage_data:
            stages = list(stage_data.keys())
            counts = list(stage_data.values())
            axes[1, 0].bar(stages, counts)
            axes[1, 0].set_title('Visa Stage Distribution (6 Months)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Monthly stage trends
        monthly_stage_data = analysis.get("monthly_stage_analysis", {})
        if monthly_stage_data:
            months = list(monthly_stage_data.keys())
            opt_trends = [monthly_stage_data[month].get('OPT', 0) for month in months]
            f1_trends = [monthly_stage_data[month].get('F1', 0) for month in months]
            
            axes[1, 1].plot(range(len(months)), opt_trends, marker='o', label='OPT', linewidth=2)
            axes[1, 1].plot(range(len(months)), f1_trends, marker='s', label='F1', linewidth=2)
            axes[1, 1].set_title('Top Visa Stages by Month')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[1, 1].legend()
            axes[1, 1].set_ylabel('Mentions')
        
        # Files processed by month
        if monthly_data:
            months = list(monthly_data.keys())
            files = [monthly_data[month]['files_processed'] for month in months]
            axes[1, 2].bar(range(len(months)), files, color='skyblue', alpha=0.7)
            axes[1, 2].set_title('Files Processed by Month')
            axes[1, 2].set_xticks(range(len(months)))
            axes[1, 2].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[1, 2].set_ylabel('Files')
        
        plt.tight_layout()
        plt.savefig('six_month_2025_analysis/plots/comprehensive_dashboard_6months_2025.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Monthly Activity Comparison
        if monthly_data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('2025 Monthly Activity Analysis (Jan-June)', fontsize=16, fontweight='bold')
            
            months = list(monthly_data.keys())
            totals = [monthly_data[month]['total_records'] for month in months]
            files = [monthly_data[month]['files_processed'] for month in months]
            
            # Records by month
            axes[0, 0].bar(range(len(months)), totals, color='lightblue', alpha=0.7)
            axes[0, 0].set_title('Records by Month')
            axes[0, 0].set_xticks(range(len(months)))
            axes[0, 0].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[0, 0].set_ylabel('Records')
            
            # Files by month
            axes[0, 1].bar(range(len(months)), files, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Files Processed by Month')
            axes[0, 1].set_xticks(range(len(months)))
            axes[0, 1].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[0, 1].set_ylabel('Files')
            
            # Average records per file
            avg_records = [monthly_data[month]['avg_records_per_file'] for month in months]
            axes[1, 0].plot(range(len(months)), avg_records, marker='o', linewidth=2, markersize=8)
            axes[1, 0].set_title('Average Records per File')
            axes[1, 0].set_xticks(range(len(months)))
            axes[1, 0].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[1, 0].set_ylabel('Records/File')
            
            # Data volume trend
            bytes_data = [monthly_data[month]['total_bytes'] / (1024*1024) for month in months]  # Convert to MB
            axes[1, 1].plot(range(len(months)), bytes_data, marker='s', linewidth=2, markersize=8, color='red')
            axes[1, 1].set_title('Data Volume by Month (MB)')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
            axes[1, 1].set_ylabel('MB')
            
            plt.tight_layout()
            plt.savefig('six_month_2025_analysis/plots/monthly_activity_6months_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Stage Analysis by Month
        if monthly_stage_data:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Visa Stage Analysis by Month (Jan-June 2025)', fontsize=16, fontweight='bold')
            
            stages = ['OPT', 'F1', 'H1B', 'greencard', 'citizenship', 'general_immigration']
            months = list(monthly_stage_data.keys())
            
            for i, stage in enumerate(stages):
                row = i // 3
                col = i % 3
                if row < 2 and col < 3:
                    stage_trends = [monthly_stage_data[month].get(stage, 0) for month in months]
                    axes[row, col].bar(range(len(months)), stage_trends, alpha=0.7)
                    axes[row, col].set_title(f'{stage} Mentions by Month')
                    axes[row, col].set_xticks(range(len(months)))
                    axes[row, col].set_xticklabels([m.split('-')[1] for m in months], rotation=45)
                    axes[row, col].set_ylabel('Mentions')
            
            plt.tight_layout()
            plt.savefig('six_month_2025_analysis/plots/stage_analysis_6months_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Daily Patterns (if available)
        if self.daily_data:
            dates = list(self.daily_data.keys())
            totals = [self.daily_data[date]['total'] for date in dates]
            
            plt.figure(figsize=(15, 6))
            plt.plot(dates, totals, marker='o', linewidth=1, markersize=3, alpha=0.7)
            plt.title('2025 Daily Activity Patterns (Jan-June)')
            plt.xlabel('Date')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('six_month_2025_analysis/plots/daily_patterns_6months_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(" Comprehensive 6-month visualizations created")
    
    def create_monthly_csvs(self, analysis: Dict[str, Any]) -> None:
        """Create monthly CSV summaries"""
        logger.info(" Creating monthly CSV summaries...")
        
        monthly_data = analysis.get("monthly_breakdown", {})
        monthly_stage_data = analysis.get("monthly_stage_analysis", {})
        
        for month_key, month_stats in monthly_data.items():
            filename = f"2025_{month_key.split('-')[1]}_monthly_summary.csv"
            
            # Get stage data for this month
            stage_data = monthly_stage_data.get(month_key, {})
            
            # Create CSV content
            csv_data = {
                'month': [month_key],
                'total_records': [month_stats['total_records']],
                'files_processed': [month_stats['files_processed']],
                'total_bytes_mb': [month_stats['total_bytes'] / (1024*1024)],
                'avg_records_per_file': [month_stats['avg_records_per_file']],
                'opt_mentions': [stage_data.get('OPT', 0)],
                'f1_mentions': [stage_data.get('F1', 0)],
                'h1b_mentions': [stage_data.get('H1B', 0)],
                'greencard_mentions': [stage_data.get('greencard', 0)],
                'citizenship_mentions': [stage_data.get('citizenship', 0)],
                'general_immigration_mentions': [stage_data.get('general_immigration', 0)]
            }
            
            df_csv = pd.DataFrame(csv_data)
            df_csv.to_csv(f'six_month_2025_analysis/monthly_csvs/{filename}', index=False)
        
        logger.info(f" Created {len(monthly_data)} monthly CSV files")
    
    def create_daily_csvs(self) -> None:
        """Create daily CSV summaries"""
        logger.info(" Creating daily CSV summaries...")
        
        if not self.daily_data:
            logger.warning(" No daily data available for CSV generation")
            return
        
        for date_str, stats in self.daily_data.items():
            # Format date as YYYY_MM_DD
            date_obj = datetime.strptime(str(date_str), '%Y-%m-%d')
            filename = f"2025_{date_obj.strftime('%m_%d')}_daily_summary.csv"
            
            # Create CSV content
            csv_data = {
                'date': [date_str],
                'total_records': [stats['total']],
                'files_processed': [stats['files']],
                'month': [stats.get('_month', '')],
                'avg_records_per_file': [stats['total'] / max(stats['files'], 1)]
            }
            
            df_csv = pd.DataFrame(csv_data)
            df_csv.to_csv(f'six_month_2025_analysis/daily_csvs/{filename}', index=False)
        
        logger.info(f" Created {len(self.daily_data)} daily CSV files")
    
    def save_results(self, analysis: Dict[str, Any]) -> None:
        """Save comprehensive 6-month analysis results"""
        logger.info(" Saving comprehensive 6-month results...")
        
        # Save JSON analysis
        with open('six_month_2025_analysis/comprehensive_analysis_6months_2025.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save sample data
        if self.sample_data:
            df = pd.DataFrame(self.sample_data)
            df.to_parquet('six_month_2025_analysis/sample_data_6months_2025.parquet', index=False)
        
        logger.info(" Comprehensive 6-month results saved")
    
    def upload_to_s3(self) -> None:
        """Upload comprehensive 6-month results to S3"""
        logger.info(" Uploading comprehensive 6-month results to S3...")
        
        try:
            # Upload analysis JSON
            self.s3_client.upload_file(
                'six_month_2025_analysis/comprehensive_analysis_6months_2025.json',
                self.s3_bucket,
                'reddit/new_data/results/2025_analysis/comprehensive_analysis_6months_2025.json'
            )
            
            # Upload all plots
            plot_files = [
                'comprehensive_dashboard_6months_2025.png',
                'monthly_activity_6months_2025.png',
                'stage_analysis_6months_2025.png',
                'daily_patterns_6months_2025.png'
            ]
            
            for plot_file in plot_files:
                if os.path.exists(f'six_month_2025_analysis/plots/{plot_file}'):
                    self.s3_client.upload_file(
                        f'six_month_2025_analysis/plots/{plot_file}',
                        self.s3_bucket,
                        f'reddit/new_data/results/2025_analysis/plots/{plot_file}'
                    )
            
            # Upload monthly CSV files
            csv_dir = 'six_month_2025_analysis/monthly_csvs'
            if os.path.exists(csv_dir):
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        self.s3_client.upload_file(
                            f'{csv_dir}/{csv_file}',
                            self.s3_bucket,
                            f'reddit/new_data/results/monthly_csvs/{csv_file}'
                        )
            
            # Upload daily CSV files
            csv_dir = 'six_month_2025_analysis/daily_csvs'
            if os.path.exists(csv_dir):
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        self.s3_client.upload_file(
                            f'{csv_dir}/{csv_file}',
                            self.s3_bucket,
                            f'reddit/new_data/results/daily_csvs/{csv_file}'
                        )
            
            logger.info(" Comprehensive 6-month results uploaded to S3")
            
        except Exception as e:
            logger.error(f" Error uploading to S3: {e}")

def main():
    """Main execution function"""
    logger.info(" Starting Comprehensive 6-Month 2025 Reddit Visa Discourse Analysis")
    logger.info("=" * 80)
    
    # Initialize analyzer
    analyzer = SixMonthReddit2025Analyzer()
    
    # Discover files by month
    monthly_files = analyzer.discover_files_by_month()
    if not any(monthly_files.values()):
        logger.error(" No files found!")
        return
    
    # Process all months
    analyzer.process_all_months(monthly_files)
    
    if not analyzer.sample_data:
        logger.error(" No data sampled!")
        return
    
    # Analyze data comprehensively
    analysis = analyzer.analyze_data_comprehensive()
    
    # Create comprehensive visualizations
    analyzer.create_comprehensive_visualizations(analysis)
    
    # Create monthly CSV summaries
    analyzer.create_monthly_csvs(analysis)
    
    # Create daily CSV summaries
    analyzer.create_daily_csvs()
    
    # Save results
    analyzer.save_results(analysis)
    
    # Upload to S3
    analyzer.upload_to_s3()
    
    logger.info(" Comprehensive 6-Month 2025 Analysis completed successfully!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
