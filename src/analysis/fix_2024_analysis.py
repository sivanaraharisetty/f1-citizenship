#!/usr/bin/env python3
"""
Fix 2024 Analysis - Re-run with improved visa stage detection
This script will re-analyze 2024 data with the same keyword patterns that worked for 2025
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import boto3
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_2024_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "siva-test-9"
S3_2024_PREFIX = "reddit/new_data/2024_data/"
S3_RESULTS_PREFIX = "reddit/new_data/results/"

class Fixed2024Analyzer:
    """Fixed analyzer for 2024 data with improved visa stage detection"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.sampled_data = []
        self.stats = {
            'files_processed': 0,
            'total_records': 0,
            'total_fear': 0,
            'total_qa': 0,
            'stage_counts': {}
        }
    
    def sample_file_fixed(self, file_key: str, sample_size: int = 1000) -> bool:
        """Sample a single file with improved error handling"""
        try:
            logger.info(f" Processing: {file_key}")
            
            # Download file
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
            content = response['Body'].read()
            
            # Parse parquet
            with io.BytesIO(content) as buffer:
                parquet_file = pq.ParquetFile(buffer)
                table = parquet_file.read(columns=None)
                df = table.to_pandas()
            
            logger.info(f" Loaded {len(df)} records from {file_key}")
            
            # Sample if needed
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f" Sampled {len(df)} records")
            
            # Store sampled data
            self.sampled_data.append(df)
            self.stats['files_processed'] += 1
            self.stats['total_records'] += len(df)
            
            return True
            
        except Exception as e:
            logger.error(f" Error processing {file_key}: {e}")
            return False
    
    def analyze_2024_fixed(self) -> Dict[str, Any]:
        """Perform fixed analysis for 2024 data"""
        logger.info(" Starting FIXED 2024 analysis...")
        
        if not self.sampled_data:
            logger.error(" No sampled data available")
            return {}
        
        # Combine all sampled data
        df = pd.concat(self.sampled_data, ignore_index=True)
        logger.info(f" Total records for analysis: {len(df)}")
        
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "year": 2024,
            "total_records": len(df),
            "fix_applied": True,
            "fix_description": "Re-ran analysis with improved visa stage keyword detection"
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
            
            # FIXED Stage analysis with improved keywords
            stages = {
                'F1': ['f1', 'student visa', 'student', 'f-1', 'f1 visa'],
                'OPT': ['opt', 'optional practical training', 'stem opt', 'cpt'],
                'H1B': ['h1b', 'h-1b', 'work visa', 'h1-b', 'h1b visa'],
                'greencard': ['green card', 'greencard', 'permanent resident', 'gc', 'i-140', 'i-485'],
                'citizenship': ['citizenship', 'naturalization', 'citizen', 'n-400'],
                'general_immigration': ['immigration', 'visa', 'immigrant', 'uscis', 'immigration law']
            }
            
            stage_counts = {}
            for stage, keywords in stages.items():
                pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                count = df[text_col].str.lower().str.contains(pattern, na=False, regex=True).sum()
                stage_counts[stage] = int(count)  # Convert to int for JSON serialization
                logger.info(f" {stage}: {count} matches")
            
            analysis["stage_analysis"] = stage_counts
            
            # Debug: Show sample matches for each stage
            logger.info(" Debugging stage detection...")
            for stage, keywords in stages.items():
                pattern = '|'.join([re.escape(keyword) for keyword in keywords])
                matches = df[df[text_col].str.lower().str.contains(pattern, na=False, regex=True)]
                if len(matches) > 0:
                    sample_text = matches[text_col].iloc[0][:100] + "..."
                    logger.info(f" {stage} sample: {sample_text}")
        
        return analysis
    
    def save_fixed_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save the fixed analysis results"""
        logger.info(" Saving fixed 2024 analysis...")
        
        # Save JSON
        output_file = "fixed_2024_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f" Saved analysis to {output_file}")
        
        # Upload to S3
        s3_key = f"{S3_RESULTS_PREFIX}fixed_2024_analysis.json"
        try:
            self.s3_client.upload_file(output_file, S3_BUCKET, s3_key)
            logger.info(f" Uploaded to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logger.error(f" Upload failed: {e}")

def main():
    """Main function to fix 2024 analysis"""
    logger.info(" Starting 2024 Analysis Fix...")
    
    analyzer = Fixed2024Analyzer()
    
    # Get list of 2024 files (sample a few for testing)
    try:
        response = analyzer.s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_2024_PREFIX,
            MaxKeys=10  # Limit for testing
        )
        
        file_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]
        logger.info(f" Found {len(file_keys)} 2024 files to process")
        
        # Process files
        for i, file_key in enumerate(file_keys):
            logger.info(f" Processing {i+1}/{len(file_keys)}: {file_key}")
            analyzer.sample_file_fixed(file_key, sample_size=500)  # Smaller sample for testing
        
        # Perform fixed analysis
        analysis = analyzer.analyze_2024_fixed()
        
        # Save results
        analyzer.save_fixed_analysis(analysis)
        
        logger.info(" 2024 Analysis Fix Complete!")
        logger.info(f" Processed {analyzer.stats['files_processed']} files")
        logger.info(f" Total records: {analyzer.stats['total_records']}")
        
    except Exception as e:
        logger.error(f" Main process failed: {e}")

if __name__ == "__main__":
    main()
