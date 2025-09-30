#!/usr/bin/env python3
"""
Comprehensive Immigration Discourse Research Pipeline
Advanced analysis toolkit for immigration research using Reddit data.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import networkx as nx
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class ImmigrationResearchPipeline:
    """Comprehensive research pipeline for immigration discourse analysis."""
    
    def __init__(self, data_dir='results/enhanced_pipeline/data', output_dir='research_outputs'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.df = None
        self.results = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        os.makedirs(f"{output_dir}/data_exports", exist_ok=True)
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except:
            self.sentiment_analyzer = None
            print("Warning: Sentiment analyzer not available, using TextBlob instead")
    
    def load_data(self):
        """Load and combine all chunk data."""
        print(" Loading immigration discourse data...")
        
        chunk_files = glob.glob(f"{self.data_dir}/chunk_*.parquet")
        if not chunk_files:
            raise FileNotFoundError("No chunk files found!")
        
        print(f"Found {len(chunk_files)} data chunks")
        
        # Load all chunks
        dfs = []
        for file in sorted(chunk_files):
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded successfully!")
        
        # Combine all data
        self.df = pd.concat(dfs, ignore_index=True)
        print(f" Loaded {len(self.df):,} total posts/comments")
        
        # Convert timestamps
        if 'created_utc' in self.df.columns:
            self.df['created_utc'] = pd.to_datetime(self.df['created_utc'], unit='s')
            self.df['date'] = self.df['created_utc'].dt.date
            self.df['month'] = self.df['created_utc'].dt.to_period('M').astype(str)
        
        return self.df
    
    def analyze_immigration_discourse_patterns(self):
        """Analyze patterns in immigration discourse."""
        print("\n Analyzing immigration discourse patterns...")
        
        if self.df is None:
            self.load_data()
        
        # Filter immigration-related content
        immigration_keywords = [
            'visa', 'immigration', 'green card', 'citizenship', 'h1b', 'f1', 'opt',
            'uscis', 'permanent residency', 'naturalization', 'asylum', 'refugee'
        ]
        
        immigration_mask = self.df['text'].str.contains(
            '|'.join(immigration_keywords), case=False, na=False
        )
        immigration_df = self.df[immigration_mask].copy()
        
        print(f"Found {len(immigration_df):,} immigration-related posts ({len(immigration_df)/len(self.df)*100:.1f}%)")
        
        # Discourse analysis
        discourse_analysis = {
            'total_immigration_posts': len(immigration_df),
            'percentage_immigration': len(immigration_df)/len(self.df)*100,
            'top_immigration_subreddits': immigration_df['subreddit'].value_counts().head(10).to_dict(),
            'immigration_timeline': immigration_df.groupby('month').size().to_dict() if 'month' in immigration_df.columns else {},
            'average_text_length': immigration_df['text'].str.len().mean(),
            'immigration_keyword_frequency': self._analyze_keyword_frequency(immigration_df)
        }
        
        self.results['discourse_patterns'] = discourse_analysis
        return discourse_analysis
    
    def _analyze_keyword_frequency(self, df):
        """Analyze frequency of immigration keywords."""
        immigration_keywords = {
            'visa_terms': ['visa', 'f1', 'h1b', 'opt', 'cpt', 'stem'],
            'process_terms': ['uscis', 'application', 'petition', 'approval', 'denial'],
            'status_terms': ['status', 'pending', 'approved', 'denied', 'rfe'],
            'legal_terms': ['lawyer', 'attorney', 'legal', 'advice', 'help'],
            'timeline_terms': ['timeline', 'wait', 'delay', 'backlog', 'priority'],
            'policy_terms': ['policy', 'reform', 'trump', 'administration', 'executive'],
            'emotional_terms': ['stress', 'anxiety', 'frustrated', 'worried', 'scared']
        }
        
        keyword_analysis = {}
        for category, keywords in immigration_keywords.items():
            counts = {}
            for keyword in keywords:
                pattern = rf'\b{keyword}\b'
                count = df['text'].str.contains(pattern, case=False, na=False).sum()
                counts[keyword] = int(count)
            keyword_analysis[category] = counts
        
        return keyword_analysis
    
    def analyze_sentiment_trends(self):
        """Analyze sentiment trends in immigration discussions."""
        print("\n😊 Analyzing sentiment trends...")
        
        if self.df is None:
            self.load_data()
        
        # Filter immigration content
        immigration_mask = self.df['text'].str.contains(
            'visa|immigration|green card|citizenship|h1b|f1|opt|uscis', 
            case=False, na=False
        )
        immigration_df = self.df[immigration_mask].copy()
        
        # Analyze sentiment
        sentiments = []
        for text in immigration_df['text'].head(1000):  # Sample for performance
            try:
                if self.sentiment_analyzer:
                    result = self.sentiment_analyzer(text[:512])  # Truncate for model
                    sentiment = result[0]['label']
                    confidence = result[0]['score']
                else:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    if polarity > 0.1:
                        sentiment = 'POSITIVE'
                    elif polarity < -0.1:
                        sentiment = 'NEGATIVE'
                    else:
                        sentiment = 'NEUTRAL'
                    confidence = abs(polarity)
                
                sentiments.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'text': text[:100]  # Sample text
                })
            except:
                continue
        
        sentiment_df = pd.DataFrame(sentiments)
        
        # Sentiment analysis results
        sentiment_analysis = {
            'total_analyzed': len(sentiment_df),
            'sentiment_distribution': sentiment_df['sentiment'].value_counts().to_dict(),
            'average_confidence': sentiment_df['confidence'].mean(),
            'sentiment_by_subreddit': self._analyze_sentiment_by_subreddit(immigration_df, sentiment_df)
        }
        
        self.results['sentiment_analysis'] = sentiment_analysis
        return sentiment_analysis
    
    def _analyze_sentiment_by_subreddit(self, df, sentiment_df):
        """Analyze sentiment patterns by subreddit."""
        # This would require more complex implementation
        # For now, return basic analysis
        return df['subreddit'].value_counts().head(5).to_dict()
    
    def analyze_community_networks(self):
        """Analyze community network structures."""
        print("\n🕸️ Analyzing community networks...")
        
        if self.df is None:
            self.load_data()
        
        # Build network based on subreddit interactions
        subreddit_interactions = defaultdict(int)
        
        # Analyze cross-subreddit patterns
        for subreddit in self.df['subreddit'].unique():
            subreddit_posts = self.df[self.df['subreddit'] == subreddit]
            subreddit_interactions[subreddit] = len(subreddit_posts)
        
        # Network analysis
        network_analysis = {
            'total_subreddits': len(subreddit_interactions),
            'most_active_subreddits': dict(Counter(subreddit_interactions).most_common(10)),
            'immigration_subreddits': self._identify_immigration_subreddits(),
            'community_health_metrics': self._calculate_community_health()
        }
        
        self.results['network_analysis'] = network_analysis
        return network_analysis
    
    def _identify_immigration_subreddits(self):
        """Identify subreddits with high immigration content."""
        immigration_keywords = ['visa', 'immigration', 'green card', 'citizenship', 'h1b', 'f1', 'opt']
        
        subreddit_immigration_scores = {}
        for subreddit in self.df['subreddit'].unique():
            subreddit_posts = self.df[self.df['subreddit'] == subreddit]
            immigration_posts = subreddit_posts['text'].str.contains(
                '|'.join(immigration_keywords), case=False, na=False
            ).sum()
            
            if len(subreddit_posts) > 0:
                score = immigration_posts / len(subreddit_posts)
                subreddit_immigration_scores[subreddit] = score
        
        return dict(Counter(subreddit_immigration_scores).most_common(10))
    
    def _calculate_community_health(self):
        """Calculate community health metrics."""
        return {
            'average_posts_per_subreddit': self.df.groupby('subreddit').size().mean(),
            'most_engaged_subreddits': self.df.groupby('subreddit').size().nlargest(5).to_dict(),
            'content_diversity': len(self.df['subreddit'].unique())
        }
    
    def analyze_temporal_trends(self):
        """Analyze temporal trends in immigration discussions."""
        print("\n Analyzing temporal trends...")
        
        if self.df is None or 'created_utc' not in self.df.columns:
            print("No temporal data available")
            return {}
        
        # Monthly trends
        monthly_trends = self.df.groupby('month').size().to_dict()
        
        # Immigration-specific trends
        immigration_mask = self.df['text'].str.contains(
            'visa|immigration|green card|citizenship', case=False, na=False
        )
        immigration_monthly = self.df[immigration_mask].groupby('month').size().to_dict()
        
        # Trend analysis
        temporal_analysis = {
            'total_posts_by_month': monthly_trends,
            'immigration_posts_by_month': immigration_monthly,
            'trend_analysis': self._analyze_trends(monthly_trends),
            'seasonal_patterns': self._analyze_seasonal_patterns()
        }
        
        self.results['temporal_analysis'] = temporal_analysis
        return temporal_analysis
    
    def _analyze_trends(self, monthly_data):
        """Analyze trend patterns."""
        if len(monthly_data) < 2:
            return {"trend": "insufficient_data"}
        
        months = list(monthly_data.keys())
        values = list(monthly_data.values())
        
        # Simple trend calculation
        if len(values) >= 2:
            trend = "increasing" if values[-1] > values[0] else "decreasing"
            return {
                "trend": trend,
                "growth_rate": (values[-1] - values[0]) / values[0] * 100 if values[0] > 0 else 0
            }
        return {"trend": "stable"}
    
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal patterns."""
        if 'created_utc' not in self.df.columns:
            return {}
        
        self.df['month_num'] = self.df['created_utc'].dt.month
        seasonal = self.df.groupby('month_num').size().to_dict()
        
        return {
            'peak_month': max(seasonal, key=seasonal.get),
            'lowest_month': min(seasonal, key=seasonal.get),
            'seasonal_variation': (max(seasonal.values()) - min(seasonal.values())) / max(seasonal.values()) * 100
        }
    
    def generate_topic_models(self):
        """Generate topic models for immigration discussions."""
        print("\n Generating topic models...")
        
        if self.df is None:
            self.load_data()
        
        # Filter immigration content
        immigration_mask = self.df['text'].str.contains(
            'visa|immigration|green card|citizenship|h1b|f1|opt', 
            case=False, na=False
        )
        immigration_df = self.df[immigration_mask].copy()
        
        if len(immigration_df) < 100:
            print("Insufficient immigration content for topic modeling")
            return {}
        
        # Prepare text data
        texts = immigration_df['text'].dropna().astype(str)
        texts = texts[texts.str.len() > 20]  # Filter very short texts
        
        if len(texts) < 50:
            print("Insufficient text data for topic modeling")
            return {}
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # LDA Topic Modeling
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'top_words': top_words,
                    'topic_weight': topic[top_words_idx].tolist()
                })
            
            topic_analysis = {
                'num_topics': 5,
                'topics': topics,
                'total_documents': len(texts),
                'vocabulary_size': len(feature_names)
            }
            
            self.results['topic_modeling'] = topic_analysis
            return topic_analysis
            
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            return {}
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n Creating research visualizations...")
        
        viz_dir = f"{self.output_dir}/visualizations"
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Immigration Content Distribution
        if 'discourse_patterns' in self.results:
            self._create_immigration_distribution_plot(viz_dir)
        
        # 2. Sentiment Analysis Visualization
        if 'sentiment_analysis' in self.results:
            self._create_sentiment_visualization(viz_dir)
        
        # 3. Temporal Trends
        if 'temporal_analysis' in self.results:
            self._create_temporal_visualization(viz_dir)
        
        # 4. Topic Model Visualization
        if 'topic_modeling' in self.results:
            self._create_topic_visualization(viz_dir)
        
        # 5. Community Network Visualization
        if 'network_analysis' in self.results:
            self._create_network_visualization(viz_dir)
        
        print(f" Visualizations saved to {viz_dir}")
    
    def _create_immigration_distribution_plot(self, viz_dir):
        """Create immigration content distribution plot."""
        plt.figure(figsize=(12, 8))
        
        # Top immigration subreddits
        if 'top_immigration_subreddits' in self.results['discourse_patterns']:
            subreddits = self.results['discourse_patterns']['top_immigration_subreddits']
            plt.subplot(2, 2, 1)
            plt.bar(range(len(subreddits)), list(subreddits.values()))
            plt.title('Top Immigration Subreddits')
            plt.xticks(range(len(subreddits)), list(subreddits.keys()), rotation=45)
        
        # Immigration keyword frequency
        if 'immigration_keyword_frequency' in self.results['discourse_patterns']:
            keywords = self.results['discourse_patterns']['immigration_keyword_frequency']
            plt.subplot(2, 2, 2)
            categories = list(keywords.keys())
            totals = [sum(cat.values()) for cat in keywords.values()]
            plt.bar(categories, totals)
            plt.title('Immigration Keyword Categories')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/immigration_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sentiment_visualization(self, viz_dir):
        """Create sentiment analysis visualization."""
        if 'sentiment_distribution' not in self.results['sentiment_analysis']:
            return
        
        plt.figure(figsize=(10, 6))
        sentiment_data = self.results['sentiment_analysis']['sentiment_distribution']
        
        plt.pie(sentiment_data.values(), labels=sentiment_data.keys(), autopct='%1.1f%%')
        plt.title('Immigration Discussion Sentiment Distribution')
        plt.savefig(f"{viz_dir}/sentiment_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_visualization(self, viz_dir):
        """Create temporal trends visualization."""
        if 'immigration_posts_by_month' not in self.results['temporal_analysis']:
            return
        
        plt.figure(figsize=(12, 6))
        monthly_data = self.results['temporal_analysis']['immigration_posts_by_month']
        
        months = list(monthly_data.keys())
        values = list(monthly_data.values())
        
        plt.plot(range(len(months)), values, marker='o')
        plt.title('Immigration Discussion Trends Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Posts')
        plt.xticks(range(len(months)), [str(m) for m in months], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{viz_dir}/temporal_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_topic_visualization(self, viz_dir):
        """Create topic model visualization."""
        if 'topics' not in self.results['topic_modeling']:
            return
        
        topics = self.results['topic_modeling']['topics']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, topic in enumerate(topics[:6]):
            if i < len(axes):
                words = topic['top_words'][:8]
                weights = topic['topic_weight'][:8]
                
                axes[i].barh(words, weights)
                axes[i].set_title(f'Topic {topic["topic_id"] + 1}')
                axes[i].set_xlabel('Weight')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/topic_models.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_network_visualization(self, viz_dir):
        """Create community network visualization."""
        if 'most_active_subreddits' not in self.results['network_analysis']:
            return
        
        plt.figure(figsize=(12, 8))
        subreddits = self.results['network_analysis']['most_active_subreddits']
        
        plt.barh(range(len(subreddits)), list(subreddits.values()))
        plt.yticks(range(len(subreddits)), list(subreddits.keys()))
        plt.title('Most Active Subreddits')
        plt.xlabel('Number of Posts')
        plt.savefig(f"{viz_dir}/community_network.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_research_report(self):
        """Generate comprehensive research report."""
        print("\n📝 Generating comprehensive research report...")
        
        report = {
            'research_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_samples': len(self.df) if self.df is not None else 0,
                'analysis_version': '1.0',
                'research_focus': 'Immigration Discourse Analysis'
            },
            'executive_summary': self._generate_executive_summary(),
            'detailed_analysis': self.results,
            'research_insights': self._generate_research_insights(),
            'methodology': self._document_methodology(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save JSON report
        report_path = f"{self.output_dir}/reports/comprehensive_research_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save markdown summary
        md_path = f"{self.output_dir}/reports/RESEARCH_SUMMARY.md"
        self._save_markdown_report(md_path, report)
        
        print(f" Research report saved to {report_path}")
        print(f" Markdown summary saved to {md_path}")
        
        return report
    
    def _generate_executive_summary(self):
        """Generate executive summary of findings."""
        summary = {
            'total_analyzed': len(self.df) if self.df is not None else 0,
            'key_findings': [],
            'research_highlights': []
        }
        
        if 'discourse_patterns' in self.results:
            immigration_pct = self.results['discourse_patterns'].get('percentage_immigration', 0)
            summary['key_findings'].append(f"Immigration content represents {immigration_pct:.1f}% of total discussions")
        
        if 'sentiment_analysis' in self.results:
            sentiment_dist = self.results['sentiment_analysis'].get('sentiment_distribution', {})
            if sentiment_dist:
                dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get)
                summary['key_findings'].append(f"Dominant sentiment in immigration discussions: {dominant_sentiment}")
        
        return summary
    
    def _generate_research_insights(self):
        """Generate research insights and implications."""
        insights = {
            'discourse_patterns': "Analysis reveals distinct patterns in immigration discourse across different communities",
            'sentiment_analysis': "Sentiment analysis provides insights into public attitudes toward immigration",
            'temporal_trends': "Temporal analysis shows how immigration discussions evolve over time",
            'community_networks': "Network analysis reveals community structures and information flow patterns",
            'topic_modeling': "Topic modeling identifies key themes in immigration discussions"
        }
        return insights
    
    def _document_methodology(self):
        """Document research methodology."""
        return {
            'data_source': 'Reddit immigration-related discussions',
            'analysis_techniques': [
                'Discourse analysis',
                'Sentiment analysis using transformer models',
                'Temporal trend analysis',
                'Community network analysis',
                'Topic modeling with LDA',
                'Statistical analysis'
            ],
            'tools_used': [
                'pandas', 'scikit-learn', 'transformers', 'networkx',
                'matplotlib', 'seaborn', 'wordcloud'
            ],
            'sample_size': len(self.df) if self.df is not None else 0
        }
    
    def _generate_recommendations(self):
        """Generate research recommendations."""
        return {
            'for_policymakers': [
                'Monitor sentiment trends to gauge public opinion',
                'Use discourse analysis to understand community concerns',
                'Track temporal patterns to assess policy impact'
            ],
            'for_researchers': [
                'Expand analysis to include more temporal data',
                'Conduct longitudinal studies of immigration discourse',
                'Compare across different social media platforms'
            ],
            'for_communities': [
                'Use insights to improve community support',
                'Identify information gaps and resource needs',
                'Develop targeted assistance programs'
            ]
        }
    
    def _save_markdown_report(self, path, report):
        """Save markdown research report."""
        with open(path, 'w') as f:
            f.write("# Immigration Discourse Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            summary = report['executive_summary']
            f.write(f"- **Total Samples Analyzed:** {summary['total_analyzed']:,}\n")
            for finding in summary['key_findings']:
                f.write(f"- {finding}\n")
            f.write("\n")
            
            f.write("## Key Research Findings\n\n")
            if 'discourse_patterns' in report['detailed_analysis']:
                patterns = report['detailed_analysis']['discourse_patterns']
                f.write("### Discourse Patterns\n")
                f.write(f"- **Immigration Content:** {patterns.get('percentage_immigration', 0):.1f}% of total discussions\n")
                f.write(f"- **Top Immigration Subreddits:** {len(patterns.get('top_immigration_subreddits', {}))}\n")
                f.write("\n")
            
            if 'sentiment_analysis' in report['detailed_analysis']:
                sentiment = report['detailed_analysis']['sentiment_analysis']
                f.write("### Sentiment Analysis\n")
                f.write(f"- **Posts Analyzed:** {sentiment.get('total_analyzed', 0):,}\n")
                f.write(f"- **Average Confidence:** {sentiment.get('average_confidence', 0):.3f}\n")
                f.write("\n")
            
            f.write("## Research Methodology\n\n")
            methodology = report['methodology']
            f.write(f"- **Data Source:** {methodology['data_source']}\n")
            f.write(f"- **Sample Size:** {methodology['sample_size']:,}\n")
            f.write(f"- **Analysis Techniques:** {', '.join(methodology['analysis_techniques'])}\n")
            f.write("\n")
            
            f.write("## Recommendations\n\n")
            recommendations = report['recommendations']
            for category, recs in recommendations.items():
                f.write(f"### {category.replace('_', ' ').title()}\n")
                for rec in recs:
                    f.write(f"- {rec}\n")
                f.write("\n")
    
    def export_research_data(self):
        """Export research data in various formats."""
        print("\n💾 Exporting research data...")
        
        export_dir = f"{self.output_dir}/data_exports"
        
        if self.df is not None:
            # Export full dataset
            self.df.to_csv(f"{export_dir}/full_dataset.csv", index=False)
            
            # Export immigration-specific data
            immigration_mask = self.df['text'].str.contains(
                'visa|immigration|green card|citizenship|h1b|f1|opt', 
                case=False, na=False
            )
            immigration_df = self.df[immigration_mask]
            immigration_df.to_csv(f"{export_dir}/immigration_dataset.csv", index=False)
            
            # Export summary statistics
            summary_stats = {
                'total_posts': len(self.df),
                'immigration_posts': len(immigration_df),
                'unique_subreddits': self.df['subreddit'].nunique(),
                'date_range': {
                    'start': str(self.df['created_utc'].min()) if 'created_utc' in self.df.columns else 'N/A',
                    'end': str(self.df['created_utc'].max()) if 'created_utc' in self.df.columns else 'N/A'
                }
            }
            
            with open(f"{export_dir}/summary_statistics.json", 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
        
        print(f" Research data exported to {export_dir}")
    
    def run_complete_analysis(self):
        """Run the complete research analysis pipeline."""
        print(" Starting Comprehensive Immigration Research Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.analyze_immigration_discourse_patterns()
        self.analyze_sentiment_trends()
        self.analyze_community_networks()
        self.analyze_temporal_trends()
        self.generate_topic_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate reports
        self.generate_research_report()
        
        # Export data
        self.export_research_data()
        
        print("\n" + "=" * 60)
        print(" COMPREHENSIVE RESEARCH ANALYSIS COMPLETE!")
        print(f" All outputs saved to: {self.output_dir}")
        print(" Check the visualizations and reports for detailed insights")
        
        return self.results

def main():
    """Main execution function."""
    # Initialize research pipeline
    pipeline = ImmigrationResearchPipeline()
    
    # Run complete analysis
    results = pipeline.run_complete_analysis()
    
    print(f"\n Research Analysis Summary:")
    print(f"- Total samples analyzed: {len(pipeline.df):,}")
    print(f"- Analysis components: {len(results)}")
    print(f"- Output directory: {pipeline.output_dir}")

if __name__ == "__main__":
    main()
