#!/usr/bin/env python3
"""
Descriptive Data Analysis for Immigration Content
"""
import pandas as pd
import numpy as np
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

class ImmigrationDataAnalyzer:
    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        ])
    
    def extract_keywords(self, texts, top_n=50):
        """Extract top keywords from text data"""
        # Combine all texts
        all_text = ' '.join(texts.astype(str))
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Filter out stop words and immigration-specific terms
        immigration_terms = {
            'visa', 'immigration', 'green', 'card', 'citizenship', 'naturalization',
            'uscis', 'h1b', 'f1', 'opt', 'cpt', 'perm', 'i140', 'i485', 'rfe',
            'denial', 'delay', 'backlog', 'reform', 'policy', 'law', 'attorney',
            'lawyer', 'help', 'advice', 'case', 'status', 'application', 'form',
            'interview', 'appointment', 'stamp', 'renewal', 'extension', 'work',
            'authorization', 'employment', 'sponsor', 'employer', 'job', 'search'
        }
        
        # Filter words
        filtered_words = [word for word in words 
                         if word not in self.stop_words 
                         and word not in immigration_terms
                         and len(word) > 3]
        
        # Count and return top keywords
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)
    
    def analyze_immigration_keywords(self, texts):
        """Analyze immigration-specific keywords"""
        immigration_keywords = {
            'visa_terms': ['visa', 'f1', 'h1b', 'opt', 'cpt', 'stem', 'work', 'student'],
            'green_card_terms': ['green', 'card', 'gc', 'i140', 'i485', 'perm', 'adjustment'],
            'citizenship_terms': ['citizenship', 'naturalization', 'n400', 'oath'],
            'process_terms': ['application', 'form', 'interview', 'appointment', 'stamp'],
            'status_terms': ['denial', 'delay', 'backlog', 'rfe', 'approval', 'pending'],
            'legal_terms': ['law', 'attorney', 'lawyer', 'legal', 'help', 'advice'],
            'policy_terms': ['reform', 'policy', 'trump', 'administration', 'executive']
        }
        
        results = {}
        for category, terms in immigration_keywords.items():
            counts = {}
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                count = sum(len(re.findall(pattern, text.lower())) for text in texts)
                counts[term] = count
            results[category] = counts
        
        return results
    
    def cluster_topics(self, texts, n_clusters=5):
        """Cluster texts into topics using TF-IDF and K-means"""
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_texts = texts[cluster_mask]
            
            # Get top terms for this cluster
            cluster_tfidf = tfidf_matrix[cluster_mask]
            mean_scores = np.mean(cluster_tfidf.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_texts),
                'percentage': len(cluster_texts) / len(texts) * 100,
                'top_terms': top_terms,
                'sample_texts': cluster_texts.head(3).tolist()
            }
        
        return cluster_analysis, cluster_labels
    
    def lda_topic_modeling(self, texts, n_topics=5):
        """Perform Latent Dirichlet Allocation topic modeling"""
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Perform LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        lda.fit(tfidf_matrix)
        
        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'word_weights': topic[topic.argsort()[-10:][::-1]]
            })
        
        return topics
    
    def generate_word_cloud(self, texts, title="Immigration Content Word Cloud"):
        """Generate word cloud from texts"""
        # Combine all texts
        all_text = ' '.join(texts.astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'wordcloud_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return wordcloud
    
    def analyze_by_label(self, df):
        """Analyze data by label categories"""
        analysis = {}
        
        for label in df['label'].unique():
            label_data = df[df['label'] == label]
            
            # Basic stats
            stats = {
                'count': len(label_data),
                'percentage': len(label_data) / len(df) * 100,
                'avg_text_length': label_data['text'].str.len().mean(),
                'top_keywords': self.extract_keywords(label_data['text'], top_n=20)
            }
            
            # Immigration keyword analysis
            immigration_analysis = self.analyze_immigration_keywords(label_data['text'])
            stats['immigration_keywords'] = immigration_analysis
            
            analysis[label] = stats
        
        return analysis
    
    def comprehensive_analysis(self, df):
        """Perform comprehensive descriptive analysis"""
        print(" Performing Comprehensive Descriptive Analysis...")
        
        results = {}
        
        # 1. Basic Statistics
        print(" Basic Statistics...")
        results['basic_stats'] = {
            'total_samples': len(df),
            'unique_labels': df['label'].nunique(),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'text_length_std': df['text'].str.len().std()
        }
        
        # 2. Top Keywords Analysis
        print("🔤 Top Keywords Analysis...")
        results['top_keywords'] = self.extract_keywords(df['text'], top_n=50)
        
        # 3. Immigration Keywords Analysis
        print("🏛️ Immigration Keywords Analysis...")
        results['immigration_keywords'] = self.analyze_immigration_keywords(df['text'])
        
        # 4. Topic Clustering
        print(" Topic Clustering...")
        cluster_analysis, cluster_labels = self.cluster_topics(df['text'], n_clusters=5)
        results['topic_clusters'] = cluster_analysis
        df['cluster'] = cluster_labels
        
        # 5. LDA Topic Modeling
        print(" LDA Topic Modeling...")
        results['lda_topics'] = self.lda_topic_modeling(df['text'], n_topics=5)
        
        # 6. Analysis by Label
        print("🏷️ Analysis by Label...")
        results['label_analysis'] = self.analyze_by_label(df)
        
        # 7. Generate Word Cloud
        print("☁️ Generating Word Cloud...")
        try:
            wordcloud = self.generate_word_cloud(df['text'])
            results['wordcloud'] = "Generated successfully"
        except Exception as e:
            results['wordcloud'] = f"Error: {str(e)}"
        
        return results, df
    
    def save_analysis_report(self, results, output_file="descriptive_analysis_report.txt"):
        """Save analysis results to a text file"""
        with open(output_file, 'w') as f:
            f.write(" IMMIGRATION DATA DESCRIPTIVE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic Statistics
            f.write(" BASIC STATISTICS\n")
            f.write("-" * 30 + "\n")
            for key, value in results['basic_stats'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Top Keywords
            f.write("🔤 TOP KEYWORDS\n")
            f.write("-" * 30 + "\n")
            for word, count in results['top_keywords'][:20]:
                f.write(f"{word}: {count}\n")
            f.write("\n")
            
            # Immigration Keywords
            f.write("🏛️ IMMIGRATION KEYWORDS ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for category, terms in results['immigration_keywords'].items():
                f.write(f"\n{category.upper()}:\n")
                for term, count in terms.items():
                    if count > 0:
                        f.write(f"  {term}: {count}\n")
            f.write("\n")
            
            # Topic Clusters
            f.write(" TOPIC CLUSTERS\n")
            f.write("-" * 30 + "\n")
            for cluster, info in results['topic_clusters'].items():
                f.write(f"\n{cluster.upper()}:\n")
                f.write(f"  Size: {info['size']} ({info['percentage']:.1f}%)\n")
                f.write(f"  Top Terms: {', '.join(info['top_terms'][:5])}\n")
            f.write("\n")
            
            # LDA Topics
            f.write(" LDA TOPICS\n")
            f.write("-" * 30 + "\n")
            for topic in results['lda_topics']:
                f.write(f"\nTopic {topic['topic_id']}:\n")
                f.write(f"  Top Words: {', '.join(topic['top_words'][:5])}\n")
            f.write("\n")
            
            # Label Analysis
            f.write("🏷️ LABEL ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for label, stats in results['label_analysis'].items():
                f.write(f"\n{label.upper()}:\n")
                f.write(f"  Count: {stats['count']} ({stats['percentage']:.1f}%)\n")
                f.write(f"  Avg Length: {stats['avg_text_length']:.1f}\n")
                f.write(f"  Top Keywords: {', '.join([word for word, _ in stats['top_keywords'][:5]])}\n")
        
        print(f" Analysis report saved to: {output_file}")

def run_descriptive_analysis(df):
    """Run descriptive analysis on the dataset"""
    analyzer = ImmigrationDataAnalyzer()
    results, df_with_clusters = analyzer.comprehensive_analysis(df)
    analyzer.save_analysis_report(results)
    return results, df_with_clusters

if __name__ == "__main__":
    # Example usage
    print(" Immigration Data Descriptive Analysis")
    print("This module provides comprehensive analysis of immigration content.")
