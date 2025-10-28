#!/usr/bin/env python3
"""
2025 Reddit Visa Discourse Analysis Pipeline
Processes 2025 data and generates comprehensive analysis matching 2024 format
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
        logging.FileHandler('2025_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Reddit2025Analyzer:
    """Comprehensive analyzer for 2025 Reddit visa discourse data"""
    
    def __init__(self, s3_bucket: str = "siva-test-9", s3_prefix: str = "reddit/new_data/2025_data/"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        self.results = {}
        self.sample_data = []
        
        # Create output directories
        os.makedirs('2025_analysis', exist_ok=True)
        os.makedirs('2025_analysis/plots', exist_ok=True)
        os.makedirs('2025_analysis/daily_csvs', exist_ok=True)
        
        logger.info(" Initialized Reddit 2025 Analyzer")
    
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
    
    def sample_file(self, file_key: str, sample_size: int = 1000) -> Dict[str, Any]:
        """Sample data from a single parquet file"""
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
            
            # Add metadata
            sample_df['_source_file'] = file_key
            sample_df['_sample_size'] = len(sample_df)
            sample_df['_total_rows'] = len(df)
            
            return {
                "success": True,
                "data": sample_df.to_dict('records'),
                "file": file_key,
                "total_rows": len(df),
                "sampled_rows": len(sample_df)
            }
            
        except Exception as e:
            logger.error(f" Error sampling {file_key}: {e}")
            return {"success": False, "data": [], "file": file_key, "error": str(e)}
    
    def process_files_parallel(self, file_keys: List[str], max_workers: int = 10, sample_size: int = 1000) -> None:
        """Process files in parallel to sample data"""
        logger.info(f" Processing {len(file_keys)} files with {max_workers} workers...")
        
        successful_files = 0
        failed_files = 0
        total_records = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.sample_file, file_key, sample_size): file_key 
                for file_key in file_keys[:50]  # Limit to first 50 files for testing
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result["success"]:
                        successful_files += 1
                        total_records += len(result["data"])
                        self.sample_data.extend(result["data"])
                        logger.info(f" {file_key}: {len(result['data'])} records")
                    else:
                        failed_files += 1
                        logger.warning(f" {file_key}: Failed")
                    
                    # Progress update
                    processed = successful_files + failed_files
                    if processed % 10 == 0:
                        logger.info(f" Progress: {processed}/{len(future_to_file)} "
                                  f"({processed/len(future_to_file)*100:.1f}%) - "
                                  f"Records: {total_records}")
                
                except Exception as e:
                    failed_files += 1
                    logger.error(f" {file_key}: Exception: {e}")
        
        logger.info(f" Sampling completed: {successful_files} successful, {failed_files} failed")
        logger.info(f" Total records sampled: {total_records}")
    
    def analyze_data(self) -> Dict[str, Any]:
        """Perform comprehensive analysis on sampled data"""
        logger.info(" Starting comprehensive analysis...")
        
        if not self.sample_data:
            logger.error(" No data to analyze")
            return {}
        
        df = pd.DataFrame(self.sample_data)
        logger.info(f" Analyzing {len(df)} records")
        
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
        
        logger.info(" Analysis completed")
        return analysis
    
    def create_visualizations(self, analysis: Dict[str, Any]) -> None:
        """Create comprehensive visualizations"""
        logger.info(" Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Comprehensive Dashboard
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
        plt.savefig('2025_analysis/plots/comprehensive_dashboard_2025.png', dpi=300, bbox_inches='tight')
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
                plt.savefig('2025_analysis/plots/correlation_heatmap_2025.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(" Visualizations created")
    
    def save_results(self, analysis: Dict[str, Any]) -> None:
        """Save analysis results"""
        logger.info(" Saving results...")
        
        # Save JSON analysis
        with open('2025_analysis/comprehensive_analysis_2025.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save sample data
        if self.sample_data:
            df = pd.DataFrame(self.sample_data)
            df.to_parquet('2025_analysis/sample_data_2025.parquet', index=False)
        
        logger.info(" Results saved")
    
    def upload_to_s3(self) -> None:
        """Upload results to S3"""
        logger.info(" Uploading results to S3...")
        
        try:
            # Upload analysis JSON
            self.s3_client.upload_file(
                '2025_analysis/comprehensive_analysis_2025.json',
                self.s3_bucket,
                'reddit/new_data/results/2025_analysis/comprehensive_analysis_2025.json'
            )
            
            # Upload plots
            plot_files = [
                'comprehensive_dashboard_2025.png',
                'correlation_heatmap_2025.png'
            ]
            
            for plot_file in plot_files:
                if os.path.exists(f'2025_analysis/plots/{plot_file}'):
                    self.s3_client.upload_file(
                        f'2025_analysis/plots/{plot_file}',
                        self.s3_bucket,
                        f'reddit/new_data/results/2025_analysis/plots/{plot_file}'
                    )
            
            logger.info(" Results uploaded to S3")
            
        except Exception as e:
            logger.error(f" Error uploading to S3: {e}")

def main():
    """Main execution function"""
    logger.info(" Starting 2025 Reddit Visa Discourse Analysis")
    logger.info("=" * 60)
    
    # Initialize analyzer
    analyzer = Reddit2025Analyzer()
    
    # Discover files
    files = analyzer.discover_files()
    if not files:
        logger.error(" No files found!")
        return
    
    # Process files (sample data)
    analyzer.process_files_parallel(files, max_workers=10, sample_size=1000)
    
    if not analyzer.sample_data:
        logger.error(" No data sampled!")
        return
    
    # Analyze data
    analysis = analyzer.analyze_data()
    
    # Create visualizations
    analyzer.create_visualizations(analysis)
    
    # Save results
    analyzer.save_results(analysis)
    
    # Upload to S3
    analyzer.upload_to_s3()
    
    logger.info(" 2025 Analysis completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
