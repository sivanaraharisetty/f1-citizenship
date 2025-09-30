#!/usr/bin/env python3
"""
Comprehensive Immigration Discourse Analysis
Analyzes all processed chunk data for immigration-related content
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

class ImmigrationAnalysis:
    def __init__(self, data_dir="results/enhanced_pipeline/data"):
        self.data_dir = data_dir
        self.all_data = None
        self.immigration_data = None
        self.analysis_results = {}
        
    def load_all_data(self):
        """Load all processed chunk data"""
        print("Loading all processed data...")
        all_chunks = []
        
        chunk_files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
        chunk_files.sort()
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(self.data_dir, chunk_file)
            try:
                df = pd.read_parquet(chunk_path)
                all_chunks.append(df)
                print(f"Loaded {chunk_file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
        
        if all_chunks:
            self.all_data = pd.concat(all_chunks, ignore_index=True)
            print(f"Total data loaded: {len(self.all_data):,} rows")
        else:
            print("No data found!")
            return False
            
        return True
    
    def identify_immigration_content(self):
        """Identify immigration-related content using enhanced patterns"""
        print("Identifying immigration-related content...")
        
        # Enhanced immigration keywords
        immigration_keywords = [
            # Visa types
            'visa', 'green card', 'citizenship', 'naturalization', 'permanent resident',
            'h1b', 'f1', 'j1', 'b1', 'b2', 'eb1', 'eb2', 'eb3', 'eb4', 'eb5',
            'student visa', 'work visa', 'tourist visa', 'business visa',
            
            # Immigration processes
            'immigration', 'emigration', 'migration', 'asylum', 'refugee',
            'deportation', 'removal', 'detention', 'border', 'customs',
            'passport', 'travel document', 'i94', 'i130', 'i485', 'i765',
            'uscis', 'dhs', 'ice', 'cbp', 'nvc', 'consulate', 'embassy',
            
            # Legal terms
            'adjustment of status', 'consular processing', 'waiver',
            'inadmissibility', 'removability', 'cancellation of removal',
            'tps', 'daca', 'parole', 'advance parole',
            
            # Countries and regions
            'mexico', 'canada', 'china', 'india', 'philippines', 'vietnam',
            'el salvador', 'guatemala', 'honduras', 'haiti', 'cuba',
            'afghanistan', 'syria', 'ukraine', 'venezuela',
            
            # Immigration stages
            'arrival', 'departure', 'entry', 'exit', 'overstay',
            'undocumented', 'illegal', 'unauthorized', 'status',
            'residency', 'domicile', 'address', 'change of address'
        ]
        
        # Subreddit patterns for immigration
        immigration_subreddits = [
            'immigration', 'uscis', 'greencard', 'citizenship', 'visa',
            'i130', 'i485', 'h1b', 'f1', 'daca', 'tps', 'asylum',
            'legaladvice', 'asklegal', 'immigrationhelp'
        ]
        
        # Create pattern for text matching
        pattern = '|'.join([re.escape(keyword) for keyword in immigration_keywords])
        
        # Check if text contains immigration keywords
        has_immigration_keywords = self.all_data['text'].str.contains(
            pattern, na=False, case=False, regex=True
        )
        
        # Check if subreddit is immigration-related
        is_immigration_subreddit = self.all_data['subreddit'].str.lower().isin(
            [sub.lower() for sub in immigration_subreddits]
        )
        
        # Combine criteria
        immigration_mask = has_immigration_keywords | is_immigration_subreddit
        
        self.immigration_data = self.all_data[immigration_mask].copy()
        
        print(f"Immigration-related content identified: {len(self.immigration_data):,} posts")
        print(f"Percentage of total content: {len(self.immigration_data)/len(self.all_data)*100:.2f}%")
        
        return len(self.immigration_data) > 0
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in immigration discourse"""
        print("Analyzing temporal patterns...")
        
        if self.immigration_data is None or len(self.immigration_data) == 0:
            return {}
        
        # Convert timestamps
        self.immigration_data['created_at'] = pd.to_datetime(
            self.immigration_data['created_utc'], unit='s', errors='coerce'
        )
        
        # Extract time components
        self.immigration_data['year'] = self.immigration_data['created_at'].dt.year
        self.immigration_data['month'] = self.immigration_data['created_at'].dt.month
        self.immigration_data['day_of_week'] = self.immigration_data['created_at'].dt.day_name()
        self.immigration_data['hour'] = self.immigration_data['created_at'].dt.hour
        
        temporal_analysis = {
            'yearly_distribution': self.immigration_data['year'].value_counts().to_dict(),
            'monthly_distribution': self.immigration_data['month'].value_counts().to_dict(),
            'daily_distribution': self.immigration_data['day_of_week'].value_counts().to_dict(),
            'hourly_distribution': self.immigration_data['hour'].value_counts().to_dict(),
            'date_range': {
                'earliest': str(self.immigration_data['created_at'].min()),
                'latest': str(self.immigration_data['created_at'].max())
            }
        }
        
        self.analysis_results['temporal'] = temporal_analysis
        return temporal_analysis
    
    def analyze_community_patterns(self):
        """Analyze community and subreddit patterns"""
        print("Analyzing community patterns...")
        
        if self.immigration_data is None or len(self.immigration_data) == 0:
            return {}
        
        community_analysis = {
            'top_subreddits': self.immigration_data['subreddit'].value_counts().head(20).to_dict(),
            'subreddit_diversity': len(self.immigration_data['subreddit'].unique()),
            'posts_per_subreddit': self.immigration_data.groupby('subreddit').size().describe().to_dict(),
            'score_distribution': {
                'mean': float(self.immigration_data['score'].mean()),
                'median': float(self.immigration_data['score'].median()),
                'std': float(self.immigration_data['score'].std())
            }
        }
        
        self.analysis_results['community'] = community_analysis
        return community_analysis
    
    def analyze_content_patterns(self):
        """Analyze content and text patterns"""
        print("Analyzing content patterns...")
        
        if self.immigration_data is None or len(self.immigration_data) == 0:
            return {}
        
        # Text length analysis
        text_lengths = self.immigration_data['text'].str.len()
        
        content_analysis = {
            'text_length_stats': {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'std': float(text_lengths.std()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max())
            },
            'total_characters': int(text_lengths.sum()),
            'average_words_per_post': float(
                self.immigration_data['text'].str.split().str.len().mean()
            )
        }
        
        # Most common words
        all_text = ' '.join(self.immigration_data['text'].astype(str))
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        word_counts = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'oil', 'sit', 'try', 'use', 'she', 'put', 'end', 'why', 'let', 'say', 'ask', 'came', 'each', 'which', 'their', 'time', 'will', 'about', 'if', 'up', 'out', 'many', 'then', 'them', 'can', 'only', 'other', 'new', 'some', 'what', 'time', 'very', 'when', 'much', 'then', 'them', 'can', 'only', 'other', 'new', 'some', 'what', 'time', 'very', 'when', 'much', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'}
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word not in stop_words and len(word) > 3}
        
        content_analysis['top_words'] = dict(Counter(filtered_words).most_common(50))
        
        self.analysis_results['content'] = content_analysis
        return content_analysis
    
    def perform_topic_modeling(self):
        """Perform topic modeling on immigration content"""
        print("Performing topic modeling...")
        
        if self.immigration_data is None or len(self.immigration_data) < 10:
            return {}
        
        # Prepare text data
        texts = self.immigration_data['text'].dropna().astype(str)
        
        if len(texts) < 10:
            return {}
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA Topic Modeling
            n_topics = min(5, len(texts) // 10)  # Adaptive number of topics
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
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
            
            topic_analysis = {
                'n_topics': n_topics,
                'topics': topics,
                'document_topic_distribution': lda.transform(tfidf_matrix).tolist()
            }
            
            self.analysis_results['topics'] = topic_analysis
            return topic_analysis
            
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            return {}
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")
        
        if self.immigration_data is None or len(self.immigration_data) == 0:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        viz_dir = "results/enhanced_pipeline/analysis"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Temporal Distribution
        if 'temporal' in self.analysis_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Immigration Discourse Temporal Patterns', fontsize=16)
            
            # Yearly distribution
            yearly_data = self.analysis_results['temporal']['yearly_distribution']
            if yearly_data:
                axes[0,0].bar(yearly_data.keys(), yearly_data.values())
                axes[0,0].set_title('Posts by Year')
                axes[0,0].set_xlabel('Year')
                axes[0,0].set_ylabel('Number of Posts')
            
            # Monthly distribution
            monthly_data = self.analysis_results['temporal']['monthly_distribution']
            if monthly_data:
                axes[0,1].bar(monthly_data.keys(), monthly_data.values())
                axes[0,1].set_title('Posts by Month')
                axes[0,1].set_xlabel('Month')
                axes[0,1].set_ylabel('Number of Posts')
            
            # Daily distribution
            daily_data = self.analysis_results['temporal']['daily_distribution']
            if daily_data:
                axes[1,0].bar(daily_data.keys(), daily_data.values())
                axes[1,0].set_title('Posts by Day of Week')
                axes[1,0].set_xlabel('Day of Week')
                axes[1,0].set_ylabel('Number of Posts')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # Hourly distribution
            hourly_data = self.analysis_results['temporal']['hourly_distribution']
            if hourly_data:
                axes[1,1].bar(hourly_data.keys(), hourly_data.values())
                axes[1,1].set_title('Posts by Hour of Day')
                axes[1,1].set_xlabel('Hour')
                axes[1,1].set_ylabel('Number of Posts')
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/temporal_patterns.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Community Analysis
        if 'community' in self.analysis_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Immigration Discourse Community Patterns', fontsize=16)
            
            # Top subreddits
            subreddit_data = self.analysis_results['community']['top_subreddits']
            if subreddit_data:
                top_10 = dict(list(subreddit_data.items())[:10])
                axes[0].barh(list(top_10.keys()), list(top_10.values()))
                axes[0].set_title('Top 10 Subreddits for Immigration Content')
                axes[0].set_xlabel('Number of Posts')
            
            # Score distribution
            scores = self.immigration_data['score'].dropna()
            if len(scores) > 0:
                axes[1].hist(scores, bins=30, alpha=0.7)
                axes[1].set_title('Score Distribution')
                axes[1].set_xlabel('Score')
                axes[1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/community_patterns.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Content Analysis
        if 'content' in self.analysis_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Immigration Content Analysis', fontsize=16)
            
            # Text length distribution
            text_lengths = self.immigration_data['text'].str.len()
            axes[0].hist(text_lengths, bins=30, alpha=0.7)
            axes[0].set_title('Text Length Distribution')
            axes[0].set_xlabel('Character Count')
            axes[0].set_ylabel('Frequency')
            
            # Word cloud
            if 'top_words' in self.analysis_results['content']:
                word_freq = self.analysis_results['content']['top_words']
                if word_freq:
                    wordcloud = WordCloud(
                        width=800, height=400, 
                        background_color='white',
                        max_words=100
                    ).generate_from_frequencies(word_freq)
                    
                    axes[1].imshow(wordcloud, interpolation='bilinear')
                    axes[1].set_title('Most Frequent Words')
                    axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/content_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Topic Modeling Visualization
        if 'topics' in self.analysis_results and self.analysis_results['topics']:
            topics = self.analysis_results['topics']['topics']
            if topics:
                fig, axes = plt.subplots(1, len(topics), figsize=(5*len(topics), 6))
                if len(topics) == 1:
                    axes = [axes]
                
                fig.suptitle('Immigration Discourse Topics', fontsize=16)
                
                for i, topic in enumerate(topics):
                    words = topic['top_words'][:10]
                    weights = topic['word_weights'][:10]
                    
                    axes[i].barh(words, weights)
                    axes[i].set_title(f'Topic {i+1}')
                    axes[i].set_xlabel('Weight')
                
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/topic_modeling.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Visualizations saved to {viz_dir}/")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("Generating comprehensive report...")
        
        report = {
            'analysis_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_samples_analyzed': len(self.all_data) if self.all_data is not None else 0,
                'immigration_samples': len(self.immigration_data) if self.immigration_data is not None else 0,
                'immigration_percentage': len(self.immigration_data)/len(self.all_data)*100 if self.all_data is not None and len(self.all_data) > 0 else 0
            },
            'dataset_overview': {
                'total_posts': len(self.all_data) if self.all_data is not None else 0,
                'immigration_posts': len(self.immigration_data) if self.immigration_data is not None else 0,
                'data_quality': {
                    'complete_records': len(self.immigration_data.dropna()) if self.immigration_data is not None else 0,
                    'missing_data_percentage': self.immigration_data.isnull().sum().sum() / (len(self.immigration_data) * len(self.immigration_data.columns)) * 100 if self.immigration_data is not None else 0
                }
            },
            'analysis_results': self.analysis_results,
            'key_insights': self._generate_insights(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save comprehensive report
        report_path = "results/enhanced_pipeline/analysis/comprehensive_immigration_analysis.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown summary
        self._generate_markdown_summary(report)
        
        print(f"Comprehensive report saved to {report_path}")
        return report
    
    def _generate_insights(self):
        """Generate key insights from analysis"""
        insights = []
        
        if self.immigration_data is not None and len(self.immigration_data) > 0:
            # Volume insights
            total_posts = len(self.all_data) if self.all_data is not None else 0
            immigration_posts = len(self.immigration_data)
            percentage = (immigration_posts / total_posts * 100) if total_posts > 0 else 0
            
            insights.append(f"Immigration content represents {percentage:.2f}% of total Reddit discussions")
            insights.append(f"Identified {immigration_posts:,} immigration-related posts from {total_posts:,} total posts")
            
            # Community insights
            if 'community' in self.analysis_results:
                top_subreddit = list(self.analysis_results['community']['top_subreddits'].keys())[0]
                top_count = list(self.analysis_results['community']['top_subreddits'].values())[0]
                insights.append(f"Most active subreddit for immigration discussions: r/{top_subreddit} ({top_count} posts)")
            
            # Temporal insights
            if 'temporal' in self.analysis_results and self.analysis_results['temporal']['yearly_distribution']:
                years = list(self.analysis_results['temporal']['yearly_distribution'].keys())
                if years:
                    insights.append(f"Data spans from {min(years)} to {max(years)}")
            
            # Content insights
            if 'content' in self.analysis_results:
                avg_length = self.analysis_results['content']['text_length_stats']['mean']
                insights.append(f"Average immigration post length: {avg_length:.0f} characters")
        
        return insights
    
    def _generate_recommendations(self):
        """Generate research recommendations"""
        recommendations = {
            'for_researchers': [
                "Conduct longitudinal analysis to track immigration discourse trends over time",
                "Compare immigration discussions across different subreddits and communities",
                "Analyze sentiment patterns in relation to policy changes",
                "Study the role of specific keywords in immigration discourse",
                "Investigate the impact of major events on immigration discussions"
            ],
            'for_policymakers': [
                "Monitor public sentiment on immigration policies through social media",
                "Use discourse analysis to understand community concerns and needs",
                "Track temporal patterns to assess policy impact",
                "Identify information gaps and areas needing better communication"
            ],
            'for_communities': [
                "Use insights to improve community support for immigrants",
                "Identify common concerns and develop targeted resources",
                "Monitor discussion trends to anticipate community needs",
                "Develop evidence-based community programs"
            ]
        }
        
        return recommendations
    
    def _generate_markdown_summary(self, report):
        """Generate markdown summary report"""
        summary_path = "results/enhanced_pipeline/analysis/IMMIGRATION_ANALYSIS_SUMMARY.md"
        
        with open(summary_path, 'w') as f:
            f.write("# Comprehensive Immigration Discourse Analysis\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            metadata = report['analysis_metadata']
            f.write(f"- **Total Samples Analyzed:** {metadata['total_samples_analyzed']:,}\n")
            f.write(f"- **Immigration Content:** {metadata['immigration_samples']:,} posts ({metadata['immigration_percentage']:.2f}%)\n")
            f.write(f"- **Analysis Date:** {metadata['generated_at']}\n\n")
            
            # Key Insights
            f.write("## Key Research Findings\n\n")
            for insight in report['key_insights']:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for category, recs in report['recommendations'].items():
                f.write(f"### {category.replace('_', ' ').title()}\n")
                for rec in recs:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            # Analysis Details
            f.write("## Analysis Details\n\n")
            if 'temporal' in report['analysis_results']:
                f.write("### Temporal Patterns\n")
                temporal = report['analysis_results']['temporal']
                if 'date_range' in temporal:
                    f.write(f"- **Date Range:** {temporal['date_range']['earliest']} to {temporal['date_range']['latest']}\n")
                f.write("\n")
            
            if 'community' in report['analysis_results']:
                f.write("### Community Analysis\n")
                community = report['analysis_results']['community']
                f.write(f"- **Subreddit Diversity:** {community['subreddit_diversity']} unique subreddits\n")
                f.write(f"- **Average Score:** {community['score_distribution']['mean']:.2f}\n")
                f.write("\n")
        
        print(f"Markdown summary saved to {summary_path}")
    
    def run_complete_analysis(self):
        """Run complete immigration discourse analysis"""
        print("="*60)
        print("COMPREHENSIVE IMMIGRATION DISCOURSE ANALYSIS")
        print("="*60)
        
        # Load all data
        if not self.load_all_data():
            print("Failed to load data. Exiting.")
            return None
        
        # Identify immigration content
        if not self.identify_immigration_content():
            print("No immigration content found. Exiting.")
            return None
        
        # Run all analyses
        print("\n" + "="*40)
        print("RUNNING ANALYSES")
        print("="*40)
        
        self.analyze_temporal_patterns()
        self.analyze_community_patterns()
        self.analyze_content_patterns()
        self.perform_topic_modeling()
        
        # Generate visualizations
        print("\n" + "="*40)
        print("GENERATING VISUALIZATIONS")
        print("="*40)
        
        self.generate_visualizations()
        
        # Generate comprehensive report
        print("\n" + "="*40)
        print("GENERATING REPORTS")
        print("="*40)
        
        report = self.generate_comprehensive_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total posts analyzed: {len(self.all_data):,}")
        print(f"Immigration posts found: {len(self.immigration_data):,}")
        print(f"Immigration percentage: {len(self.immigration_data)/len(self.all_data)*100:.2f}%")
        print("\nReports and visualizations saved to results/enhanced_pipeline/analysis/")
        
        return report

if __name__ == "__main__":
    analyzer = ImmigrationAnalysis()
    results = analyzer.run_complete_analysis()
