"""
Configuration file for Reddit Visa Discourse Analysis Pipeline
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ProjectConfig:
    """Main configuration class for the project"""
    
    # Project paths
    project_root: Path = Path(__file__).parent
    raw_data_dir: Path = project_root / "raw_data"
    sampled_data_dir: Path = project_root / "sampled_data"
    cleaned_data_dir: Path = project_root / "cleaned_data"
    descriptive_analysis_dir: Path = project_root / "descriptive_analysis"
    annotation_dir: Path = project_root / "annotation"
    classifier_dir: Path = project_root / "classifier"
    pre_post_analysis_dir: Path = project_root / "pre_post_analysis"
    visualizations_dir: Path = project_root / "visualizations"
    
    # S3 Configuration
    s3_bucket: str = "coop-published-zone-298305347319"
    s3_prefix: str = "arcticshift_reddit/"
    
    # Sampling parameters
    sample_rate: float = 0.01  # 1% sampling
    min_samples_per_file: int = 100
    oversample_rare_events: bool = True
    rare_event_threshold: float = 0.05  # 5% threshold for rare events
    
    # Text processing
    max_text_length: int = 512  # BERT token limit
    min_text_length: int = 10
    remove_stopwords: bool = True
    handle_emojis: bool = True
    
    # BERT model configuration
    model_name: str = "bert-base-uncased"  # Can be changed to distilbert, roberta, etc.
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Classification labels
    labels: List[str] = None
    
    # Subreddit and keyword mappings by visa stage
    visa_stages: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = ["fear", "question", "fear_driven_question", "other"]
        
        if self.visa_stages is None:
            self.visa_stages = {
                "student_visa": {
                    "subreddits": ["F1Visa", "OPT", "stemopt", "immigration", "visa", "immigrationusa", "USCISquestions"],
                    "keywords": ["F1", "CPT", "OPT", "STEM OPT", "Visa interview", "Visa stamping", "visa appointment", 
                               "Work authorization", "I-765", "Visa renewal", "Visa status maintenance"]
                },
                "work_visa": {
                    "subreddits": ["h1b", "WorkVisas", "immigrationlaw", "immigrationattorney", "immigrationusa", "USCISquestions"],
                    "keywords": ["H1B", "H-1B", "Employer sponsor", "Job search visa issues", "Employer withdrawal", 
                               "Visa denial", "Visa delays", "Immigration backlog", "Immigration policy changes"]
                },
                "permanent_residency": {
                    "subreddits": ["greencard", "greencardprocess", "immigration", "USCIS", "immigrationlaw", "USCISquestions"],
                    "keywords": ["I-140", "PERM", "Green Card", "GC", "Adjustment of status", "Consular processing", 
                               "Priority date", "Visa bulletin", "RFE", "USCIS case status", "I-485", "Denial", "Delay"]
                },
                "citizenship": {
                    "subreddits": ["citizenship", "immigration", "USCIS", "immigrationlaw"],
                    "keywords": ["Citizenship", "Naturalization", "Immigration reform", "Travel ban", "Deportation risk", 
                               "Immigration policy changes"]
                },
                "general_immigration": {
                    "subreddits": ["immigration", "immigrationlaw", "USCIS", "immigrationusa", "immigrationquestions", 
                                 "visasupport", "immigrationattorney", "immigrationnews"],
                    "keywords": ["Visa denial", "Visa delays", "Immigration backlog", "Immigration reform", 
                               "Legal help", "Work authorization", "Immigration policy"]
                }
            }
    
    def get_model_path(self) -> Path:
        """Get path to saved model"""
        return self.classifier_dir / "models" / f"{self.model_name.replace('/', '_')}"
    
    def get_predictions_path(self) -> Path:
        """Get path to predictions"""
        return self.classifier_dir / "predictions"
    
    def get_metrics_path(self) -> Path:
        """Get path to metrics"""
        return self.classifier_dir / "metrics"

# Global configuration instance
config = ProjectConfig()
