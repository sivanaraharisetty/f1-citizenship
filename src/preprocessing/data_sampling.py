"""
Data Sampling Module for Reddit Visa Discourse Analysis
Implements stratified sampling with oversampling for rare events
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import boto3
import s3fs
from pathlib import Path
import json
from tqdm import tqdm
import random
from collections import Counter
from sklearn.model_selection import train_test_split

from config import config
from aws_config import S3_PATHS, OUTPUT_PATHS

class RedditDataSampler:
    """Handles sampling of Reddit data with stratification and oversampling"""
    
    def __init__(self, s3_bucket: str = None, s3_prefix: str = None):
        self.s3_bucket = s3_bucket or config.s3_bucket
        self.s3_prefix = s3_prefix or config.s3_prefix
        self.s3_client = boto3.client('s3')
        self.fs = s3fs.S3FileSystem()
        
    def list_s3_files(self) -> List[str]:
        """List all files in S3 bucket with year-based partitioning"""
        try:
            all_files = []
            
            # List files from different year partitions
            for path_name, s3_path in S3_PATHS.items():
                print(f"Listing files from {path_name}: {s3_path}")
                
                # Extract prefix from S3 path
                prefix = s3_path.replace(f"s3://{self.s3_bucket}/", "")
                
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=prefix
                )
                
                files = [obj['Key'] for obj in response.get('Contents', [])]
                all_files.extend(files)
                print(f"Found {len(files)} files in {path_name}")
            
            print(f"Total files found: {len(all_files)}")
            return all_files
        except Exception as e:
            print(f"Error listing S3 files: {e}")
            return []
    
    def load_file_sample(self, file_path: str, sample_rate: float = None) -> pd.DataFrame:
        """Load and sample a single file from S3"""
        sample_rate = sample_rate or config.sample_rate
        
        try:
            # Read file from S3
            if file_path.endswith('.json'):
                df = pd.read_json(f"s3://{self.s3_bucket}/{file_path}", lines=True)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(f"s3://{self.s3_bucket}/{file_path}")
            else:
                print(f"Unsupported file format: {file_path}")
                return pd.DataFrame()
            
            # Apply sampling
            if len(df) > 0:
                sample_size = max(int(len(df) * sample_rate), config.min_samples_per_file)
                if sample_size < len(df):
                    df = df.sample(n=sample_size, random_state=42)
                
                # Add metadata
                df['source_file'] = file_path
                df['sample_rate'] = sample_rate
                
            return df
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()
    
    def identify_visa_stage(self, text: str, subreddit: str) -> str:
        """Identify visa stage based on text content and subreddit"""
        text_lower = text.lower()
        subreddit_lower = subreddit.lower()
        
        # Check subreddit patterns
        for stage, stage_info in config.visa_stages.items():
            for subreddit_pattern in stage_info["subreddits"]:
                if subreddit_pattern.lower() in subreddit_lower:
                    return stage
        
        # Check keyword patterns
        for stage, stage_info in config.visa_stages.items():
            for keyword in stage_info["keywords"]:
                if keyword.lower() in text_lower:
                    return stage
        
        return "unknown"
    
    def detect_rare_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and mark rare events for oversampling"""
        # Define patterns for rare events (fear-driven questions, etc.)
        fear_patterns = [
            r'\b(afraid|scared|worried|anxious|panic|fear|terrified|nervous)\b',
            r'\b(denied|rejected|refused|failed)\b',
            r'\b(emergency|urgent|help|desperate)\b'
        ]
        
        question_patterns = [
            r'\?',
            r'\b(how|what|when|where|why|can|could|should|would)\b'
        ]
        
        # Create binary indicators
        df['has_fear'] = df['text'].str.contains('|'.join(fear_patterns), case=False, na=False)
        df['has_question'] = df['text'].str.contains('|'.join(question_patterns), case=False, na=False)
        df['is_rare_event'] = df['has_fear'] & df['has_question']
        
        return df
    
    def stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform stratified sampling by subreddit, month, and visa stage"""
        if len(df) == 0:
            return df
        
        # Add temporal information if available
        if 'created_utc' in df.columns:
            df['month'] = pd.to_datetime(df['created_utc'], unit='s').dt.month
        else:
            df['month'] = 1  # Default month if timestamp not available
        
        # Add visa stage
        df['visa_stage'] = df.apply(
            lambda row: self.identify_visa_stage(
                str(row.get('text', '')), 
                str(row.get('subreddit', ''))
            ), axis=1
        )
        
        # Detect rare events
        df = self.detect_rare_events(df)
        
        # Stratified sampling
        sampled_dfs = []
        for (subreddit, month, visa_stage), group in df.groupby(['subreddit', 'month', 'visa_stage']):
            sample_size = max(int(len(group) * config.sample_rate), 1)
            sampled_group = group.sample(n=min(sample_size, len(group)), random_state=42)
            sampled_dfs.append(sampled_group)
        
        # Oversample rare events
        if config.oversample_rare_events:
            rare_events = df[df['is_rare_event']]
            if len(rare_events) > 0:
                # Oversample rare events by 2x
                oversampled_rare = rare_events.sample(
                    n=min(len(rare_events) * 2, len(rare_events) * 10), 
                    replace=True, 
                    random_state=42
                )
                sampled_dfs.append(oversampled_rare)
        
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        return result_df
    
    def sample_all_files(self, file_list: List[str] = None) -> pd.DataFrame:
        """Sample all files in the dataset"""
        if file_list is None:
            file_list = self.list_s3_files()
        
        if not file_list:
            print("No files found to sample")
            return pd.DataFrame()
        
        all_samples = []
        
        print(f"Sampling {len(file_list)} files...")
        for file_path in tqdm(file_list):
            try:
                sample_df = self.load_file_sample(file_path)
                if not sample_df.empty:
                    stratified_df = self.stratified_sample(sample_df)
                    all_samples.append(stratified_df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if all_samples:
            combined_df = pd.concat(all_samples, ignore_index=True)
            
            # Save sampled data
            output_path = config.sampled_data_dir / "sampled_reddit_data.parquet"
            config.sampled_data_dir.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(output_path, index=False)
            
            print(f"Sampling complete. Total samples: {len(combined_df)}")
            print(f"Saved to: {output_path}")
            
            # Print summary statistics
            self.print_sampling_summary(combined_df)
            
            return combined_df
        else:
            print("No data was successfully sampled")
            return pd.DataFrame()
    
    def print_sampling_summary(self, df: pd.DataFrame):
        """Print summary statistics of sampled data"""
        print("\n=== SAMPLING SUMMARY ===")
        print(f"Total samples: {len(df)}")
        print(f"Unique subreddits: {df['subreddit'].nunique()}")
        print(f"Visa stage distribution:")
        print(df['visa_stage'].value_counts())
        print(f"Rare events: {df['is_rare_event'].sum()} ({df['is_rare_event'].mean():.2%})")
        print(f"Fear indicators: {df['has_fear'].sum()} ({df['has_fear'].mean():.2%})")
        print(f"Question indicators: {df['has_question'].sum()} ({df['has_question'].mean():.2%})")

def main():
    """Main function for data sampling"""
    sampler = RedditDataSampler()
    sampled_data = sampler.sample_all_files()
    return sampled_data

if __name__ == "__main__":
    main()
