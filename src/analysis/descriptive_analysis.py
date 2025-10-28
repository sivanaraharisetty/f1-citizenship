"""
Descriptive Analysis Module for Reddit Visa Discourse Analysis
Analyzes keywords, topics, and distributions in the dataset
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from textblob import TextBlob
import re
from tqdm import tqdm
import json
from pathlib import Path

from config import config

class RedditDescriptiveAnalyzer:
    """Handles descriptive analysis of Reddit visa discourse data"""
    
    def __init__(self):
        self.visa_stages = config.visa_stages
        self.labels = config.labels
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned data for analysis"""
        cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
        if not cleaned_file.exists():
            raise FileNotFoundError("Cleaned data not found. Please run data_cleaning.py first.")
        
        return pd.read_parquet(cleaned_file)
    
    def analyze_keywords(self, df: pd.DataFrame) -> Dict:
        """Analyze top keywords and their frequencies"""
        print("Analyzing keywords...")
        
        # Combine all processed text
        all_text = ' '.join(df['processed_text'].dropna().astype(str))
        
        # Get word frequencies
        words = all_text.split()
        word_freq = Counter(words)
        
        # Filter out very common words and short words
        filtered_words = {
            word: count for word, count in word_freq.items() 
            if len(word) > 2 and count > 5
        }
        
        # Top keywords overall
        top_keywords = dict(Counter(filtered_words).most_common(50))
        
        # Keywords by visa stage
        stage_keywords = {}
        for stage in df['visa_stage'].unique():
            if stage == 'unknown':
                continue
            stage_text = ' '.join(df[df['visa_stage'] == stage]['processed_text'].dropna().astype(str))
            stage_words = stage_text.split()
            stage_freq = Counter(stage_words)
            stage_filtered = {
                word: count for word, count in stage_freq.items() 
                if len(word) > 2 and count > 2
            }
            stage_keywords[stage] = dict(Counter(stage_filtered).most_common(20))
        
        # Keywords by subreddit
        subreddit_keywords = {}
        for subreddit in df['subreddit'].value_counts().head(10).index:
            subreddit_text = ' '.join(df[df['subreddit'] == subreddit]['processed_text'].dropna().astype(str))
            subreddit_words = subreddit_text.split()
            subreddit_freq = Counter(subreddit_words)
            subreddit_filtered = {
                word: count for word, count in subreddit_freq.items() 
                if len(word) > 2 and count > 1
            }
            subreddit_keywords[subreddit] = dict(Counter(subreddit_filtered).most_common(15))
        
        return {
            'overall_keywords': top_keywords,
            'stage_keywords': stage_keywords,
            'subreddit_keywords': subreddit_keywords,
            'total_unique_words': len(word_freq),
            'total_words': sum(word_freq.values())
        }
    
    def analyze_topic_clusters(self, df: pd.DataFrame, n_topics: int = 10) -> Dict:
        """Perform topic modeling using LDA"""
        print("Analyzing topic clusters...")
        
        # Prepare text data
        texts = df['processed_text'].dropna().astype(str).tolist()
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(tfidf_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'word_weights': topic[top_words_idx].tolist()
            })
        
        # Assign topics to documents
        doc_topic_probs = lda.transform(tfidf_matrix)
        df['dominant_topic'] = doc_topic_probs.argmax(axis=1)
        df['topic_confidence'] = doc_topic_probs.max(axis=1)
        
        return {
            'topics': topics,
            'topic_distribution': df['dominant_topic'].value_counts().to_dict(),
            'avg_topic_confidence': df['topic_confidence'].mean(),
            'feature_names': feature_names.tolist()
        }
    
    def analyze_temporal_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in the data"""
        print("Analyzing temporal distribution...")
        
        # Convert timestamp to datetime
        if 'created_utc' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['hour'] = df['datetime'].dt.hour
        else:
            print("No timestamp data available for temporal analysis")
            return {}
        
        # Temporal distributions
        temporal_stats = {
            'yearly_distribution': df['year'].value_counts().sort_index().to_dict(),
            'monthly_distribution': df['month'].value_counts().sort_index().to_dict(),
            'daily_distribution': df['day_of_week'].value_counts().to_dict(),
            'hourly_distribution': df['hour'].value_counts().sort_index().to_dict()
        }
        
        # Temporal patterns by visa stage
        stage_temporal = {}
        for stage in df['visa_stage'].unique():
            if stage == 'unknown':
                continue
            stage_data = df[df['visa_stage'] == stage]
            stage_temporal[stage] = {
                'yearly': stage_data['year'].value_counts().sort_index().to_dict(),
                'monthly': stage_data['month'].value_counts().sort_index().to_dict()
            }
        
        return {
            'temporal_stats': temporal_stats,
            'stage_temporal': stage_temporal,
            'date_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max()
            }
        }
    
    def analyze_subreddit_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze subreddit distribution and characteristics"""
        print("Analyzing subreddit distribution...")
        
        subreddit_stats = df['subreddit'].value_counts()
        
        # Subreddit characteristics
        subreddit_analysis = {}
        for subreddit in subreddit_stats.head(20).index:
            subreddit_data = df[df['subreddit'] == subreddit]
            subreddit_analysis[subreddit] = {
                'count': len(subreddit_data),
                'avg_word_count': subreddit_data['word_count'].mean(),
                'visa_stage_distribution': subreddit_data['visa_stage'].value_counts().to_dict(),
                'avg_text_length': subreddit_data['processed_text'].str.len().mean()
            }
        
        return {
            'subreddit_counts': subreddit_stats.to_dict(),
            'subreddit_analysis': subreddit_analysis,
            'total_unique_subreddits': df['subreddit'].nunique()
        }
    
    def analyze_sentiment_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze sentiment patterns in the data"""
        print("Analyzing sentiment patterns...")
        
        sentiments = []
        for text in tqdm(df['processed_text'].dropna(), desc="Computing sentiment"):
            blob = TextBlob(str(text))
            sentiments.append(blob.sentiment.polarity)
        
        df['sentiment'] = sentiments
        
        # Sentiment analysis by visa stage
        stage_sentiment = {}
        for stage in df['visa_stage'].unique():
            if stage == 'unknown':
                continue
            stage_sentiments = df[df['visa_stage'] == stage]['sentiment']
            stage_sentiment[stage] = {
                'mean': stage_sentiments.mean(),
                'std': stage_sentiments.std(),
                'positive_ratio': (stage_sentiments > 0.1).mean(),
                'negative_ratio': (stage_sentiments < -0.1).mean()
            }
        
        # Sentiment by subreddit
        subreddit_sentiment = {}
        for subreddit in df['subreddit'].value_counts().head(10).index:
            subreddit_sentiments = df[df['subreddit'] == subreddit]['sentiment']
            subreddit_sentiment[subreddit] = {
                'mean': subreddit_sentiments.mean(),
                'std': subreddit_sentiments.std()
            }
        
        return {
            'overall_sentiment': {
                'mean': df['sentiment'].mean(),
                'std': df['sentiment'].std(),
                'positive_ratio': (df['sentiment'] > 0.1).mean(),
                'negative_ratio': (df['sentiment'] < -0.1).mean()
            },
            'stage_sentiment': stage_sentiment,
            'subreddit_sentiment': subreddit_sentiment
        }
    
    def create_visualizations(self, analysis_results: Dict, df: pd.DataFrame):
        """Create visualizations for the analysis results"""
        print("Creating visualizations...")
        
        # Set up the output directory
        viz_dir = config.visualizations_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Top keywords word cloud
        if 'keyword_analysis' in analysis_results:
            keywords = analysis_results['keyword_analysis']['overall_keywords']
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color='white',
                max_words=100
            ).generate_from_frequencies(keywords)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Top Keywords in Reddit Visa Discourse')
            plt.tight_layout()
            plt.savefig(viz_dir / 'wordcloud_keywords.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Visa stage distribution
        stage_counts = df['visa_stage'].value_counts()
        plt.figure(figsize=(10, 6))
        stage_counts.plot(kind='bar')
        plt.title('Distribution of Posts by Visa Stage')
        plt.xlabel('Visa Stage')
        plt.ylabel('Number of Posts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'visa_stage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Subreddit distribution
        top_subreddits = df['subreddit'].value_counts().head(15)
        plt.figure(figsize=(12, 8))
        top_subreddits.plot(kind='barh')
        plt.title('Top 15 Subreddits by Post Count')
        plt.xlabel('Number of Posts')
        plt.tight_layout()
        plt.savefig(viz_dir / 'subreddit_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Temporal distribution (if available)
        if 'temporal_analysis' in analysis_results:
            temporal = analysis_results['temporal_analysis']
            if 'temporal_stats' in temporal:
                # Yearly distribution
                if 'yearly_distribution' in temporal['temporal_stats']:
                    years = list(temporal['temporal_stats']['yearly_distribution'].keys())
                    counts = list(temporal['temporal_stats']['yearly_distribution'].values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(years, counts, marker='o')
                    plt.title('Posts Over Time')
                    plt.xlabel('Year')
                    plt.ylabel('Number of Posts')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(viz_dir / 'temporal_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 5. Sentiment analysis (if available)
        if 'sentiment_analysis' in analysis_results:
            sentiment = analysis_results['sentiment_analysis']
            if 'stage_sentiment' in sentiment:
                stages = list(sentiment['stage_sentiment'].keys())
                means = [sentiment['stage_sentiment'][stage]['mean'] for stage in stages]
                
                plt.figure(figsize=(10, 6))
                plt.bar(stages, means)
                plt.title('Average Sentiment by Visa Stage')
                plt.xlabel('Visa Stage')
                plt.ylabel('Average Sentiment Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(viz_dir / 'sentiment_by_stage.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def run_full_analysis(self) -> Dict:
        """Run complete descriptive analysis"""
        print("Starting comprehensive descriptive analysis...")
        
        # Load data
        df = self.load_cleaned_data()
        print(f"Loaded {len(df)} cleaned records")
        
        # Run all analyses
        analysis_results = {}
        
        # Keyword analysis
        analysis_results['keyword_analysis'] = self.analyze_keywords(df)
        
        # Topic modeling
        analysis_results['topic_analysis'] = self.analyze_topic_clusters(df)
        
        # Temporal analysis
        analysis_results['temporal_analysis'] = self.analyze_temporal_distribution(df)
        
        # Subreddit analysis
        analysis_results['subreddit_analysis'] = self.analyze_subreddit_distribution(df)
        
        # Sentiment analysis
        analysis_results['sentiment_analysis'] = self.analyze_sentiment_patterns(df)
        
        # Create visualizations
        self.create_visualizations(analysis_results, df)
        
        # Save results
        output_file = config.descriptive_analysis_dir / "descriptive_analysis_results.json"
        config.descriptive_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = recursive_convert(analysis_results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Analysis results saved to: {output_file}")
        
        # Print summary
        self.print_analysis_summary(analysis_results)
        
        return analysis_results
    
    def print_analysis_summary(self, results: Dict):
        """Print summary of analysis results"""
        print("\n=== DESCRIPTIVE ANALYSIS SUMMARY ===")
        
        if 'keyword_analysis' in results:
            keywords = results['keyword_analysis']
            print(f"Total unique words: {keywords['total_unique_words']}")
            print(f"Total words: {keywords['total_words']}")
            print("Top 10 keywords:", list(keywords['overall_keywords'].keys())[:10])
        
        if 'topic_analysis' in results:
            topics = results['topic_analysis']
            print(f"Number of topics identified: {len(topics['topics'])}")
            print(f"Average topic confidence: {topics['avg_topic_confidence']:.3f}")
        
        if 'subreddit_analysis' in results:
            subreddit = results['subreddit_analysis']
            print(f"Total unique subreddits: {subreddit['total_unique_subreddits']}")
        
        if 'sentiment_analysis' in results:
            sentiment = results['sentiment_analysis']
            overall = sentiment['overall_sentiment']
            print(f"Overall sentiment: {overall['mean']:.3f} (std: {overall['std']:.3f})")
            print(f"Positive ratio: {overall['positive_ratio']:.3f}")
            print(f"Negative ratio: {overall['negative_ratio']:.3f}")

def main():
    """Main function for descriptive analysis"""
    analyzer = RedditDescriptiveAnalyzer()
    results = analyzer.run_full_analysis()
    return results

if __name__ == "__main__":
    main()
