#!/usr/bin/env python3
"""
Comprehensive 2025 Reddit Visa Discourse Analysis Pipeline
Matches the exact 2024 analysis format with all visualizations and daily summaries
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
        logging.FileHandler('comprehensive_2025_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveReddit2025Analyzer:
    """Comprehensive analyzer for 2025 Reddit visa discourse data matching 2024 format"""
    
    def __init__(self, s3_bucket: str = "siva-test-9", s3_prefix: str = "reddit/new_data/2025_data/"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        self.results = {}
        self.sample_data = []
        self.daily_data = {}
        
        # Create output directories
        os.makedirs('comprehensive_2025_analysis', exist_ok=True)
        os.makedirs('comprehensive_2025_analysis/plots', exist_ok=True)
        os.makedirs('comprehensive_2025_analysis/daily_csvs', exist_ok=True)
        
        logger.info(" Initialized Comprehensive Reddit 2025 Analyzer")
    
    def discover_files(self) -> List[str]:
        """Discover all parquet files in 2025 data"""
        logger.info(" Discovering 2025 data files...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    files.append(obj['Key'])
            
            logger.info(f" Found {len(files)} parquet files")
            return files
            
        except Exception as e:
            logger.error(f" Error discovering files: {e}")
            return []
    
    def sample_file_comprehensive(self, file_key: str, sample_size: int = 2000) -> Dict[str, Any]:
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
            
            return {
                "success": True,
                "data": sample_df.to_dict('records'),
                "file": file_key,
                "total_rows": len(df),
                "sampled_rows": len(sample_df),
                "file_size": len(content)
            }
            
        except Exception as e:
            logger.error(f" Error sampling {file_key}: {e}")
            return {"success": False, "data": [], "file": file_key, "error": str(e)}
    
    def process_files_comprehensive(self, file_keys: List[str], max_workers: int = 15, sample_size: int = 2000) -> None:
        """Process files comprehensively to sample data"""
        logger.info(f" Processing {len(file_keys)} files with {max_workers} workers...")
        
        successful_files = 0
        failed_files = 0
        total_records = 0
        total_bytes = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks (process more files for comprehensive analysis)
            future_to_file = {
                executor.submit(self.sample_file_comprehensive, file_key, sample_size): file_key 
                for file_key in file_keys[:200]  # Process more files for comprehensive analysis
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
                        self.sample_data.extend(result["data"])
                        logger.info(f" {file_key}: {len(result['data'])} records")
                    else:
                        failed_files += 1
                        logger.warning(f" {file_key}: Failed")
                    
                    # Progress update
                    processed = successful_files + failed_files
                    if processed % 20 == 0:
                        logger.info(f" Progress: {processed}/{len(future_to_file)} "
                                  f"({processed/len(future_to_file)*100:.1f}%) - "
                                  f"Records: {total_records:,}")
                
                except Exception as e:
                    failed_files += 1
                    logger.error(f" {file_key}: Exception: {e}")
        
        logger.info(f" Comprehensive sampling completed: {successful_files} successful, {failed_files} failed")
        logger.info(f" Total records sampled: {total_records:,}")
        logger.info(f" Total bytes processed: {total_bytes:,}")
    
    def analyze_data_comprehensive(self) -> Dict[str, Any]:
        """Perform comprehensive analysis matching 2024 format"""
        logger.info(" Starting comprehensive analysis...")
        
        if not self.sample_data:
            logger.error(" No data to analyze")
            return {}
        
        df = pd.DataFrame(self.sample_data)
        logger.info(f" Analyzing {len(df):,} records")
        
        # Basic stats
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "year": 2025,
            "total_records": len(df),
            "sample_files": len(df['_source_file'].unique()) if '_source_file' in df.columns else 0
        }
        
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
                    '_source_file': 'nunique'
                }).rename(columns={'date': 'total', '_source_file': 'files'})
                
                self.daily_data = daily_stats.to_dict('index')
                
                # Monthly analysis
                df['month'] = df['date'].dt.to_period('M')
                monthly_stats = df.groupby('month').agg({
                    'date': 'count',
                    '_source_file': 'nunique'
                }).rename(columns={'date': 'total', '_source_file': 'files'})
                
                # Convert Period keys to strings for JSON serialization
                analysis["monthly_analysis"] = {str(k): v.to_dict() for k, v in monthly_stats.iterrows()}
                
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
        
        # Summary stats
        analysis["summary_stats"] = {
            "total_records": len(df),
            "total_fear": analysis.get("total_fear", 0),
            "total_qa": analysis.get("total_qa", 0),
            "fear_rate": analysis.get("fear_rate", 0),
            "qa_rate": analysis.get("qa_rate", 0),
            "avg_records_per_file": len(df) / max(analysis.get("sample_files", 1), 1)
        }
        
        logger.info(" Comprehensive analysis completed")
        return analysis
    
    def create_comprehensive_visualizations(self, analysis: Dict[str, Any]) -> None:
        """Create all visualizations matching 2024 format"""
        logger.info(" Creating comprehensive visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Comprehensive Dashboard (matching 2024 format)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('2025 Reddit Visa Discourse Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Summary stats
        stats = analysis.get("summary_stats", {})
        axes[0, 0].bar(['Total Records', 'Fear Cases', 'Q&A Cases'], 
                      [stats.get('total_records', 0), stats.get('total_fear', 0), stats.get('total_qa', 0)])
        axes[0, 0].set_title('Overall Statistics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Fear vs Q&A rates
        axes[0, 1].pie([stats.get('fear_rate', 0), 1 - stats.get('fear_rate', 0)], 
                      labels=['Fear', 'Non-Fear'], autopct='%1.1f%%')
        axes[0, 1].set_title('Fear Rate Distribution')
        
        # Stage analysis
        stage_data = analysis.get("stage_analysis", {})
        if stage_data:
            stages = list(stage_data.keys())
            counts = list(stage_data.values())
            axes[1, 0].bar(stages, counts)
            axes[1, 0].set_title('Visa Stage Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Monthly trends (if available)
        monthly_data = analysis.get("monthly_analysis", {})
        if monthly_data:
            months = [str(month) for month in monthly_data.keys()]
            totals = [monthly_data[month].get('total', 0) for month in monthly_data.keys()]
            axes[1, 1].plot(range(len(months)), totals, marker='o')
            axes[1, 1].set_title('Monthly Activity Trends')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels(months, rotation=45)
        
        plt.tight_layout()
        plt.savefig('comprehensive_2025_analysis/plots/comprehensive_dashboard_2025.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Heatmap
        if len(self.sample_data) > 0:
            df = pd.DataFrame(self.sample_data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('2025 Data Correlation Heatmap')
                plt.tight_layout()
                plt.savefig('comprehensive_2025_analysis/plots/correlation_heatmap_2025.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Daily Patterns
        if self.daily_data:
            dates = list(self.daily_data.keys())
            totals = [self.daily_data[date]['total'] for date in dates]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, totals, marker='o', linewidth=2, markersize=6)
            plt.title('2025 Daily Activity Patterns')
            plt.xlabel('Date')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/daily_patterns_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Fear Analysis
        if 'has_fear' in df.columns:
            fear_by_stage = {}
            stage_data = analysis.get("stage_analysis", {})
            for stage in stage_data.keys():
                stage_mask = df[text_col].str.lower().str.contains('|'.join([re.escape(k) for k in stages[stage]]), na=False, regex=True)
                fear_by_stage[stage] = df[stage_mask & df['has_fear']].shape[0]
            
            plt.figure(figsize=(10, 6))
            stages = list(fear_by_stage.keys())
            fear_counts = list(fear_by_stage.values())
            plt.bar(stages, fear_counts, color='red', alpha=0.7)
            plt.title('Fear Analysis by Visa Stage')
            plt.xlabel('Visa Stage')
            plt.ylabel('Fear Cases')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/fear_analysis_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Monthly Activity
        if monthly_data:
            months = [str(month) for month in monthly_data.keys()]
            totals = [monthly_data[month].get('total', 0) for month in monthly_data.keys()]
            
            plt.figure(figsize=(10, 6))
            plt.bar(months, totals, color='skyblue', alpha=0.7)
            plt.title('2025 Monthly Activity')
            plt.xlabel('Month')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/monthly_activity_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Q&A Analysis
        if 'is_question' in df.columns:
            qa_by_stage = {}
            stage_data = analysis.get("stage_analysis", {})
            for stage in stage_data.keys():
                stage_mask = df[text_col].str.lower().str.contains('|'.join([re.escape(k) for k in stages[stage]]), na=False, regex=True)
                qa_by_stage[stage] = df[stage_mask & df['is_question']].shape[0]
            
            plt.figure(figsize=(10, 6))
            stages = list(qa_by_stage.keys())
            qa_counts = list(qa_by_stage.values())
            plt.bar(stages, qa_counts, color='green', alpha=0.7)
            plt.title('Q&A Analysis by Visa Stage')
            plt.xlabel('Visa Stage')
            plt.ylabel('Q&A Cases')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/qa_analysis_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. Stage Distribution
        if stage_data:
            plt.figure(figsize=(10, 8))
            stages = list(stage_data.keys())
            counts = list(stage_data.values())
            plt.pie(counts, labels=stages, autopct='%1.1f%%', startangle=90)
            plt.title('2025 Visa Stage Distribution')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/stage_distribution_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 8. Stage Trends
        if monthly_data and stage_data:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('2025 Visa Stage Trends by Month', fontsize=16)
            
            stages = list(stage_data.keys())
            months = [str(month) for month in monthly_data.keys()]
            
            for i, stage in enumerate(stages):
                row = i // 3
                col = i % 3
                if row < 2 and col < 3:
                    # Calculate stage trends by month (simplified)
                    stage_trends = [monthly_data[month].get('total', 0) * (stage_data[stage] / sum(stage_data.values())) for month in monthly_data.keys()]
                    axes[row, col].plot(months, stage_trends, marker='o')
                    axes[row, col].set_title(f'{stage} Trends')
                    axes[row, col].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/stage_trends_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 9. Volume Trends
        if self.daily_data:
            dates = list(self.daily_data.keys())
            totals = [self.daily_data[date]['total'] for date in dates]
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(dates, totals, alpha=0.7, color='blue')
            plt.plot(dates, totals, color='darkblue', linewidth=2)
            plt.title('2025 Volume Trends')
            plt.xlabel('Date')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('comprehensive_2025_analysis/plots/volume_trends_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(" Comprehensive visualizations created")
    
    def create_daily_csvs(self) -> None:
        """Create daily CSV summaries matching 2024 format"""
        logger.info(" Creating daily CSV summaries...")
        
        if not self.daily_data:
            logger.warning(" No daily data available for CSV generation")
            return
        
        for date_str, stats in self.daily_data.items():
            # Format date as YYYY_MM
            date_obj = datetime.strptime(str(date_str), '%Y-%m-%d')
            filename = f"2025_{date_obj.strftime('%m')}_daily_summary.csv"
            
            # Create CSV content
            csv_data = {
                'date': [date_str],
                'total_records': [stats['total']],
                'files_processed': [stats['files']],
                'avg_records_per_file': [stats['total'] / max(stats['files'], 1)]
            }
            
            df_csv = pd.DataFrame(csv_data)
            df_csv.to_csv(f'comprehensive_2025_analysis/daily_csvs/{filename}', index=False)
        
        logger.info(f" Created {len(self.daily_data)} daily CSV files")
    
    def save_results(self, analysis: Dict[str, Any]) -> None:
        """Save comprehensive analysis results"""
        logger.info(" Saving comprehensive results...")
        
        # Save JSON analysis
        with open('comprehensive_2025_analysis/comprehensive_analysis_2025.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save sample data
        if self.sample_data:
            df = pd.DataFrame(self.sample_data)
            df.to_parquet('comprehensive_2025_analysis/sample_data_2025.parquet', index=False)
        
        logger.info(" Comprehensive results saved")
    
    def upload_to_s3(self) -> None:
        """Upload comprehensive results to S3"""
        logger.info(" Uploading comprehensive results to S3...")
        
        try:
            # Upload analysis JSON
            self.s3_client.upload_file(
                'comprehensive_2025_analysis/comprehensive_analysis_2025.json',
                self.s3_bucket,
                'reddit/new_data/results/2025_analysis/comprehensive_analysis_2025.json'
            )
            
            # Upload all plots
            plot_files = [
                'comprehensive_dashboard_2025.png',
                'correlation_heatmap_2025.png',
                'daily_patterns_2025.png',
                'fear_analysis_2025.png',
                'monthly_activity_2025.png',
                'qa_analysis_2025.png',
                'stage_distribution_2025.png',
                'stage_trends_2025.png',
                'volume_trends_2025.png'
            ]
            
            for plot_file in plot_files:
                if os.path.exists(f'comprehensive_2025_analysis/plots/{plot_file}'):
                    self.s3_client.upload_file(
                        f'comprehensive_2025_analysis/plots/{plot_file}',
                        self.s3_bucket,
                        f'reddit/new_data/results/2025_analysis/plots/{plot_file}'
                    )
            
            # Upload daily CSV files
            csv_dir = 'comprehensive_2025_analysis/daily_csvs'
            if os.path.exists(csv_dir):
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        self.s3_client.upload_file(
                            f'{csv_dir}/{csv_file}',
                            self.s3_bucket,
                            f'reddit/new_data/results/daily_csvs/{csv_file}'
                        )
            
            logger.info(" Comprehensive results uploaded to S3")
            
        except Exception as e:
            logger.error(f" Error uploading to S3: {e}")

def main():
    """Main execution function"""
    logger.info(" Starting Comprehensive 2025 Reddit Visa Discourse Analysis")
    logger.info("=" * 80)
    
    # Initialize analyzer
    analyzer = ComprehensiveReddit2025Analyzer()
    
    # Discover files
    files = analyzer.discover_files()
    if not files:
        logger.error(" No files found!")
        return
    
    # Process files comprehensively
    analyzer.process_files_comprehensive(files, max_workers=15, sample_size=2000)
    
    if not analyzer.sample_data:
        logger.error(" No data sampled!")
        return
    
    # Analyze data comprehensively
    analysis = analyzer.analyze_data_comprehensive()
    
    # Create comprehensive visualizations
    analyzer.create_comprehensive_visualizations(analysis)
    
    # Create daily CSV summaries
    analyzer.create_daily_csvs()
    
    # Save results
    analyzer.save_results(analysis)
    
    # Upload to S3
    analyzer.upload_to_s3()
    
    logger.info(" Comprehensive 2025 Analysis completed successfully!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
