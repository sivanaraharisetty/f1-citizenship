"""
Annotation System for Reddit Visa Discourse Analysis
Handles manual and semi-automated annotation for fear, Q&A, and fear-driven questions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import random
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

from config import config

class RedditAnnotationSystem:
    """Handles annotation of Reddit posts for fear, Q&A, and fear-driven questions"""
    
    def __init__(self):
        self.labels = config.labels
        self.annotation_file = config.annotation_dir / "annotations.json"
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing annotations if available
        self.annotations = self.load_annotations()
        
        # Define annotation guidelines
        self.guidelines = {
            'fear': {
                'description': 'Posts expressing fear, anxiety, worry, or concern about visa processes',
                'keywords': ['afraid', 'scared', 'worried', 'anxious', 'panic', 'fear', 'terrified', 'nervous', 
                           'denied', 'rejected', 'refused', 'failed', 'emergency', 'urgent', 'help', 'desperate'],
                'examples': [
                    'I\'m really scared my H1B will be denied',
                    'What if my visa gets rejected? I\'m so worried',
                    'Emergency: My OPT is expiring and I\'m panicking'
                ]
            },
            'question': {
                'description': 'Posts asking questions or seeking information',
                'keywords': ['how', 'what', 'when', 'where', 'why', 'can', 'could', 'should', 'would', '?'],
                'examples': [
                    'How long does H1B processing take?',
                    'What documents do I need for F1 visa?',
                    'Can I work on OPT while waiting for H1B?'
                ]
            },
            'fear_driven_question': {
                'description': 'Questions that are motivated by fear or anxiety about visa processes',
                'keywords': ['afraid', 'scared', 'worried', 'anxious', 'panic', 'fear', 'terrified', 'nervous', 
                           'denied', 'rejected', 'refused', 'failed', 'emergency', 'urgent', 'help', 'desperate',
                           'how', 'what', 'when', 'where', 'why', 'can', 'could', 'should', 'would', '?'],
                'examples': [
                    'I\'m scared my H1B will be denied. What should I do?',
                    'What if my visa gets rejected? How can I prepare?',
                    'Emergency: My OPT is expiring. Can I still work?'
                ]
            },
            'other': {
                'description': 'Posts that don\'t fit the above categories',
                'keywords': [],
                'examples': [
                    'Just got my H1B approved!',
                    'Sharing my visa interview experience',
                    'Tips for F1 visa application'
                ]
            }
        }
    
    def load_annotations(self) -> Dict:
        """Load existing annotations from file"""
        if self.annotation_file.exists():
            with open(self.annotation_file, 'r') as f:
                return json.load(f)
        return {
            'annotations': {},
            'metadata': {
                'total_annotated': 0,
                'label_distribution': {},
                'annotator_info': {}
            }
        }
    
    def save_annotations(self):
        """Save annotations to file"""
        with open(self.annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def create_annotation_sample(self, df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
        """Create a stratified sample for annotation"""
        print(f"Creating annotation sample of {sample_size} posts...")
        
        # Stratify by visa stage and subreddit
        annotation_sample = []
        
        for (visa_stage, subreddit), group in df.groupby(['visa_stage', 'subreddit']):
            if len(group) > 0:
                # Sample proportionally from each group
                group_sample_size = max(1, int(sample_size * len(group) / len(df)))
                group_sample = group.sample(n=min(group_sample_size, len(group)), random_state=42)
                annotation_sample.append(group_sample)
        
        # Combine and shuffle
        annotation_df = pd.concat(annotation_sample, ignore_index=True)
        annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Limit to requested sample size
        annotation_df = annotation_df.head(sample_size)
        
        # Add annotation columns
        annotation_df['annotated'] = False
        annotation_df['labels'] = ''
        annotation_df['confidence'] = 0.0
        annotation_df['annotator_notes'] = ''
        
        return annotation_df
    
    def suggest_labels(self, text: str) -> Dict[str, float]:
        """Suggest labels based on keyword matching and patterns"""
        text_lower = text.lower()
        suggestions = {}
        
        for label, info in self.guidelines.items():
            if label == 'other':
                continue
            
            score = 0.0
            keywords = info['keywords']
            
            # Check for keyword matches
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            
            # Check for question patterns
            if label in ['question', 'fear_driven_question']:
                if '?' in text:
                    score += 0.5
                if any(q_word in text_lower for q_word in ['how', 'what', 'when', 'where', 'why']):
                    score += 0.3
            
            # Check for fear patterns
            if label in ['fear', 'fear_driven_question']:
                fear_words = ['afraid', 'scared', 'worried', 'anxious', 'panic', 'fear', 'terrified', 'nervous']
                fear_count = sum(1 for word in fear_words if word in text_lower)
                score += fear_count * 0.5
            
            # Normalize score
            suggestions[label] = min(score / max(len(keywords), 1), 1.0)
        
        # If no strong suggestions, mark as 'other'
        if max(suggestions.values()) < 0.3:
            suggestions['other'] = 1.0
        
        return suggestions
    
    def interactive_annotation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interactive annotation interface using Streamlit"""
        st.set_page_config(page_title="Reddit Visa Discourse Annotation", layout="wide")
        
        st.title("Reddit Visa Discourse Annotation System")
        st.markdown("Annotate posts for fear, Q&A, and fear-driven questions")
        
        # Initialize session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'annotations' not in st.session_state:
            st.session_state.annotations = {}
        
        # Sidebar for navigation
        st.sidebar.header("Navigation")
        
        # Display current post
        current_post = df.iloc[st.session_state.current_index]
        
        st.subheader(f"Post {st.session_state.current_index + 1} of {len(df)}")
        
        # Post information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Text:**")
            st.text_area("", value=current_post['processed_text'], height=200, disabled=True)
        
        with col2:
            st.markdown("**Metadata:**")
            st.write(f"**Subreddit:** {current_post['subreddit']}")
            st.write(f"**Visa Stage:** {current_post['visa_stage']}")
            st.write(f"**Word Count:** {current_post['word_count']}")
            if 'sentiment' in current_post:
                st.write(f"**Sentiment:** {current_post['sentiment']:.3f}")
        
        # Annotation interface
        st.subheader("Annotation")
        
        # Get suggested labels
        suggestions = self.suggest_labels(current_post['processed_text'])
        
        # Display suggestions
        st.markdown("**Suggested Labels:**")
        for label, score in suggestions.items():
            if score > 0.1:
                st.write(f"- {label}: {score:.2f}")
        
        # Manual annotation
        selected_labels = st.multiselect(
            "Select labels:",
            options=self.labels,
            default=[],
            key=f"labels_{st.session_state.current_index}"
        )
        
        confidence = st.slider(
            "Confidence (0-1):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key=f"confidence_{st.session_state.current_index}"
        )
        
        notes = st.text_area(
            "Notes:",
            key=f"notes_{st.session_state.current_index}"
        )
        
        # Navigation buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Previous"):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()
        
        with col2:
            if st.button("Next"):
                if st.session_state.current_index < len(df) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        with col3:
            if st.button("Save Annotation"):
                # Save current annotation
                post_id = f"post_{st.session_state.current_index}"
                st.session_state.annotations[post_id] = {
                    'labels': selected_labels,
                    'confidence': confidence,
                    'notes': notes,
                    'text': current_post['processed_text'],
                    'metadata': {
                        'subreddit': current_post['subreddit'],
                        'visa_stage': current_post['visa_stage'],
                        'word_count': current_post['word_count']
                    }
                }
                st.success("Annotation saved!")
        
        with col4:
            if st.button("Skip"):
                if st.session_state.current_index < len(df) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        # Progress
        progress = (st.session_state.current_index + 1) / len(df)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.current_index + 1}/{len(df)} ({progress:.1%})")
        
        # Annotation summary
        if st.session_state.annotations:
            st.subheader("Annotation Summary")
            total_annotated = len(st.session_state.annotations)
            st.write(f"Total annotated: {total_annotated}")
            
            # Label distribution
            all_labels = []
            for ann in st.session_state.annotations.values():
                all_labels.extend(ann['labels'])
            
            if all_labels:
                label_counts = Counter(all_labels)
                st.write("Label distribution:")
                for label, count in label_counts.items():
                    st.write(f"- {label}: {count}")
        
        return df
    
    def batch_annotation(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """Batch annotation using semi-automated approach"""
        print(f"Starting batch annotation of {len(df)} posts...")
        
        # Create annotation sample if not already done
        if 'annotated' not in df.columns:
            df = self.create_annotation_sample(df, len(df))
        
        # Process in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: posts {batch_start}-{batch_end-1}")
            
            for idx, row in batch_df.iterrows():
                if row.get('annotated', False):
                    continue
                
                # Get suggested labels
                suggestions = self.suggest_labels(row['processed_text'])
                
                # Auto-annotate high-confidence suggestions
                if max(suggestions.values()) > 0.7:
                    best_label = max(suggestions, key=suggestions.get)
                    df.at[idx, 'labels'] = best_label
                    df.at[idx, 'confidence'] = suggestions[best_label]
                    df.at[idx, 'annotated'] = True
                else:
                    # Mark for manual review
                    df.at[idx, 'labels'] = 'needs_review'
                    df.at[idx, 'confidence'] = 0.0
                    df.at[idx, 'annotated'] = False
        
        return df
    
    def create_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training and validation datasets from annotations"""
        # Filter annotated data
        annotated_df = df[df['annotated'] == True].copy()
        
        if len(annotated_df) == 0:
            raise ValueError("No annotated data found. Please run annotation first.")
        
        # Prepare features and labels
        X = annotated_df['processed_text'].values
        y = []
        
        for labels_str in annotated_df['labels']:
            if isinstance(labels_str, str):
                labels = labels_str.split(',') if ',' in labels_str else [labels_str]
            else:
                labels = [labels_str]
            
            # Convert to binary labels
            binary_labels = [1 if label in labels else 0 for label in self.labels]
            y.append(binary_labels)
        
        y = np.array(y)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Create dataframes
        train_df = pd.DataFrame({
            'text': X_train,
            'labels': y_train.tolist()
        })
        
        val_df = pd.DataFrame({
            'text': X_val,
            'labels': y_val.tolist()
        })
        
        return train_df, val_df
    
    def evaluate_annotations(self, df: pd.DataFrame) -> Dict:
        """Evaluate annotation quality and consistency"""
        annotated_df = df[df['annotated'] == True]
        
        if len(annotated_df) == 0:
            return {}
        
        # Label distribution
        label_distribution = {}
        for label in self.labels:
            count = annotated_df['labels'].str.contains(label, na=False).sum()
            label_distribution[label] = count
        
        # Confidence distribution
        confidence_stats = {
            'mean': annotated_df['confidence'].mean(),
            'std': annotated_df['confidence'].std(),
            'min': annotated_df['confidence'].min(),
            'max': annotated_df['confidence'].max()
        }
        
        # Annotation quality metrics
        quality_metrics = {
            'total_annotated': len(annotated_df),
            'label_distribution': label_distribution,
            'confidence_stats': confidence_stats,
            'coverage': len(annotated_df) / len(df),
            'avg_confidence': annotated_df['confidence'].mean()
        }
        
        return quality_metrics
    
    def export_annotations(self, df: pd.DataFrame, output_file: str = None) -> str:
        """Export annotations to file"""
        if output_file is None:
            output_file = config.annotation_dir / "exported_annotations.parquet"
        
        # Filter annotated data
        annotated_df = df[df['annotated'] == True].copy()
        
        # Save to file
        annotated_df.to_parquet(output_file, index=False)
        
        print(f"Exported {len(annotated_df)} annotations to {output_file}")
        return str(output_file)
    
    def run_annotation_pipeline(self, df: pd.DataFrame, method: str = 'batch') -> pd.DataFrame:
        """Run the complete annotation pipeline"""
        print("Starting annotation pipeline...")
        
        if method == 'interactive':
            # Interactive annotation
            return self.interactive_annotation(df)
        elif method == 'batch':
            # Batch annotation
            return self.batch_annotation(df)
        else:
            raise ValueError("Method must be 'interactive' or 'batch'")
    
    def print_annotation_summary(self, df: pd.DataFrame):
        """Print summary of annotation results"""
        annotated_df = df[df['annotated'] == True]
        
        print("\n=== ANNOTATION SUMMARY ===")
        print(f"Total posts: {len(df)}")
        print(f"Annotated posts: {len(annotated_df)}")
        print(f"Coverage: {len(annotated_df)/len(df):.2%}")
        
        if len(annotated_df) > 0:
            print(f"Average confidence: {annotated_df['confidence'].mean():.3f}")
            
            # Label distribution
            print("\nLabel distribution:")
            for label in self.labels:
                count = annotated_df['labels'].str.contains(label, na=False).sum()
                print(f"  {label}: {count}")

def main():
    """Main function for annotation system"""
    import pandas as pd
    
    # Load cleaned data
    cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
    if not cleaned_file.exists():
        print("No cleaned data found. Please run data_cleaning.py first.")
        return
    
    df = pd.read_parquet(cleaned_file)
    print(f"Loaded {len(df)} cleaned records for annotation")
    
    # Create annotation system
    annotator = RedditAnnotationSystem()
    
    # Create annotation sample
    annotation_sample = annotator.create_annotation_sample(df, sample_size=1000)
    
    # Run annotation pipeline
    annotated_df = annotator.run_annotation_pipeline(annotation_sample, method='batch')
    
    # Export annotations
    output_file = annotator.export_annotations(annotated_df)
    
    # Print summary
    annotator.print_annotation_summary(annotated_df)
    
    return annotated_df

if __name__ == "__main__":
    main()
