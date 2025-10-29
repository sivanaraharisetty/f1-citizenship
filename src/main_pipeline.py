"""
Main Pipeline for Reddit Visa Discourse Analysis
Orchestrates the complete analysis workflow from data sampling to visualization
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
import argparse
import sys

# Import all modules
from data_sampling import RedditDataSampler
from data_cleaning import RedditDataCleaner
from descriptive_analysis import RedditDescriptiveAnalyzer
from annotation_system import RedditAnnotationSystem
# BERT classifier removed - using keyword-based analysis
from evaluation_metrics import RedditEvaluationMetrics
from temporal_analysis import RedditTemporalAnalyzer
from visualization_tools import RedditVisualizationTools

from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RedditAnalysisPipeline:
    """Main pipeline for Reddit visa discourse analysis"""
    
    def __init__(self, config_override: Dict = None):
        self.config = config
        if config_override:
            self.config.__dict__.update(config_override)
        
        self.results = {}
        self.start_time = datetime.now()
        
        logger.info("Initialized Reddit Analysis Pipeline")
        logger.info(f"Project root: {self.config.project_root}")
    
    def run_data_sampling(self, sample_size: int = None) -> pd.DataFrame:
        """Run data sampling step"""
        logger.info("Starting data sampling...")
        
        sampler = RedditDataSampler(
            s3_bucket=self.config.s3_bucket,
            s3_prefix=self.config.s3_prefix
        )
        
        # Get file list
        file_list = sampler.list_s3_files()
        if not file_list:
            logger.warning("No files found in S3 bucket")
            return pd.DataFrame()
        
        # Sample data
        sampled_data = sampler.sample_all_files(file_list)
        
        if sampled_data.empty:
            logger.error("No data was sampled")
            return pd.DataFrame()
        
        self.results['sampling'] = {
            'total_files': len(file_list),
            'sampled_records': len(sampled_data),
            'sample_rate': self.config.sample_rate
        }
        
        logger.info(f"Data sampling completed: {len(sampled_data)} records")
        return sampled_data
    
    def run_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run data cleaning step"""
        logger.info("Starting data cleaning...")
        
        cleaner = RedditDataCleaner()
        
        # Clean data
        cleaned_df = cleaner.clean_dataframe(df)
        
        if cleaned_df.empty:
            logger.error("No data remained after cleaning")
            return pd.DataFrame()
        
        # Get cleaning summary
        cleaning_summary = cleaner.get_cleaning_summary(cleaned_df)
        
        self.results['cleaning'] = cleaning_summary
        
        logger.info(f"Data cleaning completed: {len(cleaned_df)} records")
        return cleaned_df
    
    def run_descriptive_analysis(self, df: pd.DataFrame) -> Dict:
        """Run descriptive analysis step"""
        logger.info("Starting descriptive analysis...")
        
        analyzer = RedditDescriptiveAnalyzer()
        analysis_results = analyzer.run_full_analysis()
        
        self.results['descriptive_analysis'] = analysis_results
        
        logger.info("Descriptive analysis completed")
        return analysis_results
    
    def run_annotation(self, df: pd.DataFrame, method: str = 'batch') -> pd.DataFrame:
        """Run annotation step"""
        logger.info("Starting annotation...")
        
        annotator = RedditAnnotationSystem()
        
        # Create annotation sample
        annotation_sample = annotator.create_annotation_sample(df, sample_size=1000)
        
        # Run annotation
        annotated_df = annotator.run_annotation_pipeline(annotation_sample, method=method)
        
        # Export annotations
        output_file = annotator.export_annotations(annotated_df)
        
        # Get annotation summary
        annotation_summary = annotator.evaluate_annotations(annotated_df)
        
        self.results['annotation'] = annotation_summary
        
        logger.info(f"Annotation completed: {len(annotated_df)} annotated records")
        return annotated_df
    
    def run_keyword_analysis(self, df: pd.DataFrame) -> Dict:
        """Run keyword-based analysis"""
        logger.info("Starting keyword-based analysis...")
        
        # Keyword-based analysis is performed in complete_no_sampling_analysis.py
        # This method kept for compatibility but analysis runs separately
        logger.info("Keyword-based analysis runs via complete_no_sampling_analysis.py")
        return {}
    
    def run_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_scores: np.ndarray = None, texts: List[str] = None) -> Dict:
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        evaluator = RedditEvaluationMetrics()
        
        # Run evaluation
        evaluation_results = evaluator.run_comprehensive_evaluation(
            y_true, y_pred, y_scores, texts
        )
        
        self.results['evaluation'] = evaluation_results
        
        logger.info("Comprehensive evaluation completed")
        return evaluation_results
    
    def run_temporal_analysis(self) -> Dict:
        """Run temporal analysis"""
        logger.info("Starting temporal analysis...")
        
        analyzer = RedditTemporalAnalyzer()
        temporal_results = analyzer.run_comprehensive_temporal_analysis()
        
        self.results['temporal_analysis'] = temporal_results
        
        logger.info("Temporal analysis completed")
        return temporal_results
    
    def run_visualization(self, all_data: Dict) -> Path:
        """Run visualization generation"""
        logger.info("Starting visualization generation...")
        
        viz_tools = RedditVisualizationTools()
        
        # Create comprehensive report
        output_dir = self.config.visualizations_dir
        dashboard_file = viz_tools.create_comprehensive_report(all_data, output_dir)
        
        self.results['visualization'] = {
            'output_directory': str(output_dir),
            'dashboard_file': str(dashboard_file)
        }
        
        logger.info(f"Visualization completed: {dashboard_file}")
        return dashboard_file
    
    def run_full_pipeline(self, steps: List[str] = None) -> Dict:
        """Run the complete analysis pipeline"""
        if steps is None:
            steps = [
                'sampling', 'cleaning', 'descriptive_analysis', 
                'annotation', 'evaluation',
                'temporal_analysis', 'visualization'
            ]
        
        logger.info(f"Starting full pipeline with steps: {steps}")
        
        try:
            # Step 1: Data Sampling
            if 'sampling' in steps:
                sampled_data = self.run_data_sampling()
                if sampled_data.empty:
                    logger.error("Pipeline failed at data sampling step")
                    return self.results
            else:
                # Load existing sampled data
                sampled_file = self.config.sampled_data_dir / "sampled_reddit_data.parquet"
                if sampled_file.exists():
                    sampled_data = pd.read_parquet(sampled_file)
                else:
                    logger.error("No sampled data found and sampling step skipped")
                    return self.results
            
            # Step 2: Data Cleaning
            if 'cleaning' in steps:
                cleaned_data = self.run_data_cleaning(sampled_data)
                if cleaned_data.empty:
                    logger.error("Pipeline failed at data cleaning step")
                    return self.results
            else:
                # Load existing cleaned data
                cleaned_file = self.config.cleaned_data_dir / "cleaned_reddit_data.parquet"
                if cleaned_file.exists():
                    cleaned_data = pd.read_parquet(cleaned_file)
                else:
                    logger.error("No cleaned data found and cleaning step skipped")
                    return self.results
            
            # Step 3: Descriptive Analysis
            if 'descriptive_analysis' in steps:
                descriptive_results = self.run_descriptive_analysis(cleaned_data)
            else:
                # Load existing descriptive analysis
                desc_file = self.config.descriptive_analysis_dir / "descriptive_analysis_results.json"
                if desc_file.exists():
                    with open(desc_file, 'r') as f:
                        descriptive_results = json.load(f)
                else:
                    descriptive_results = {}
            
            # Step 4: Annotation
            if 'annotation' in steps:
                annotated_data = self.run_annotation(cleaned_data)
                if annotated_data.empty:
                    logger.error("Pipeline failed at annotation step")
                    return self.results
            else:
                # Load existing annotations
                annotation_file = self.config.annotation_dir / "exported_annotations.parquet"
                if annotation_file.exists():
                    annotated_data = pd.read_parquet(annotation_file)
                else:
                    logger.error("No annotated data found and annotation step skipped")
                    return self.results
            
            # Step 5: Evaluation
            if 'evaluation' in steps:
                # This would typically use model predictions
                # For now, we'll skip this step
                logger.info("Evaluation step skipped - requires model predictions")
                evaluation_results = {}
            else:
                evaluation_results = {}
            
            # Step 7: Temporal Analysis
            if 'temporal_analysis' in steps:
                temporal_results = self.run_temporal_analysis()
            else:
                # Load existing temporal analysis
                temporal_file = self.config.pre_post_analysis_dir / "temporal_analysis_results.json"
                if temporal_file.exists():
                    with open(temporal_file, 'r') as f:
                        temporal_results = json.load(f)
                else:
                    temporal_results = {}
            
            # Step 8: Visualization
            if 'visualization' in steps:
                # Combine all results for visualization
                all_data = {
                    'descriptive_analysis': descriptive_results,
                    'evaluation': evaluation_results,
                    'temporal_analysis': temporal_results
                }
                
                dashboard_file = self.run_visualization(all_data)
            else:
                dashboard_file = None
            
            # Save final results
            self.save_final_results()
            
            logger.info("Full pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
        
        return self.results
    
    def save_final_results(self):
        """Save final pipeline results"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        final_results = {
            'pipeline_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'duration_human': str(duration)
            },
            'results': self.results
        }
        
        # Save results
        results_file = self.config.project_root / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_file}")
    
    def print_pipeline_summary(self):
        """Print summary of pipeline execution"""
        print("\n" + "="*50)
        print("REDDIT VISA DISCOURSE ANALYSIS PIPELINE SUMMARY")
        print("="*50)
        
        if 'sampling' in self.results:
            sampling = self.results['sampling']
            print(f"Data Sampling:")
            print(f"  Total files processed: {sampling['total_files']}")
            print(f"  Records sampled: {sampling['sampled_records']}")
            print(f"  Sample rate: {sampling['sample_rate']}")
        
        if 'cleaning' in self.results:
            cleaning = self.results['cleaning']
            print(f"\nData Cleaning:")
            print(f"  Valid entries: {cleaning['valid_entries']}")
            print(f"  Average word count: {cleaning['avg_word_count']:.1f}")
            print(f"  Unique subreddits: {cleaning['unique_subreddits']}")
        
        if 'annotation' in self.results:
            annotation = self.results['annotation']
            print(f"\nAnnotation:")
            print(f"  Total annotated: {annotation['total_annotated']}")
            print(f"  Coverage: {annotation['coverage']:.2%}")
            print(f"  Average confidence: {annotation['avg_confidence']:.3f}")
        
        
        duration = datetime.now() - self.start_time
        print(f"\nPipeline Duration: {duration}")
        print("="*50)

def main():
    """Main function for the pipeline"""
    parser = argparse.ArgumentParser(description='Reddit Visa Discourse Analysis Pipeline')
    parser.add_argument('--steps', nargs='+', 
                       choices=['sampling', 'cleaning', 'descriptive_analysis', 
                               'annotation', 'evaluation',
                               'temporal_analysis', 'visualization'],
                       default=None,
                       help='Specific steps to run (default: all steps)')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--sample-size', type=int, help='Sample size for annotation')
    parser.add_argument('--annotation-method', choices=['batch', 'interactive'], 
                       default='batch', help='Annotation method')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RedditAnalysisPipeline()
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(steps=args.steps)
        pipeline.print_pipeline_summary()
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {pipeline.config.project_root}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
