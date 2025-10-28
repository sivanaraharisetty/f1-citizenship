"""
Temporal Analysis Module for Reddit Visa Discourse Analysis
Analyzes pre/post policy changes and temporal trends in visa discourse
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm

from config import config

class RedditTemporalAnalyzer:
    """Analyzes temporal patterns and policy impacts in Reddit visa discourse"""
    
    def __init__(self):
        self.labels = config.labels
        self.visa_stages = config.visa_stages
        
        # Define major policy events for analysis
        self.policy_events = {
            'trump_travel_ban': datetime(2017, 1, 27),  # First travel ban
            'h1b_suspension': datetime(2020, 6, 22),   # H1B suspension
            'biden_immigration_reform': datetime(2021, 1, 20),  # Biden inauguration
            'daca_renewal': datetime(2021, 1, 20),      # DACA renewal
            'h1b_cap_increase': datetime(2021, 12, 6),   # H1B cap increase
        }
        
        # Analysis windows (days before/after policy events)
        self.analysis_windows = {
            'short_term': 30,   # 30 days
            'medium_term': 90, # 90 days
            'long_term': 180   # 180 days
        }
    
    def load_data_with_predictions(self) -> pd.DataFrame:
        """Load data with classifier predictions"""
        # Load cleaned data
        cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
        if not cleaned_file.exists():
            raise FileNotFoundError("Cleaned data not found. Please run data_cleaning.py first.")
        
        df = pd.read_parquet(cleaned_file)
        
        # Load predictions if available
        predictions_file = config.get_predictions_path() / "predictions.parquet"
        if predictions_file.exists():
            predictions_df = pd.read_parquet(predictions_file)
            # Merge predictions with main data
            df = df.merge(predictions_df, on='id', how='left', suffixes=('', '_pred'))
        else:
            print("No predictions found. Using manual annotations if available.")
        
        return df
    
    def prepare_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for temporal analysis"""
        # Convert timestamp to datetime
        if 'created_utc' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        else:
            print("No timestamp data available for temporal analysis")
            return df
        
        # Extract temporal features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.day_name()
        df['hour'] = df['datetime'].dt.hour
        df['week'] = df['datetime'].dt.isocalendar().week
        df['quarter'] = df['datetime'].dt.quarter
        
        # Create time-based bins
        df['time_bin'] = pd.cut(df['datetime'], bins=30, labels=False)
        
        return df
    
    def analyze_temporal_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal trends in discourse patterns"""
        print("Analyzing temporal trends...")
        
        temporal_trends = {}
        
        # Overall temporal distribution
        temporal_trends['overall_distribution'] = {
            'yearly': df['year'].value_counts().sort_index().to_dict(),
            'monthly': df['month'].value_counts().sort_index().to_dict(),
            'daily': df['weekday'].value_counts().to_dict(),
            'hourly': df['hour'].value_counts().sort_index().to_dict()
        }
        
        # Trends by visa stage
        temporal_trends['stage_trends'] = {}
        for stage in df['visa_stage'].unique():
            if stage == 'unknown':
                continue
            
            stage_data = df[df['visa_stage'] == stage]
            temporal_trends['stage_trends'][stage] = {
                'yearly': stage_data['year'].value_counts().sort_index().to_dict(),
                'monthly': stage_data['month'].value_counts().sort_index().to_dict()
            }
        
        # Trends by subreddit
        temporal_trends['subreddit_trends'] = {}
        top_subreddits = df['subreddit'].value_counts().head(10).index
        for subreddit in top_subreddits:
            subreddit_data = df[df['subreddit'] == subreddit]
            temporal_trends['subreddit_trends'][subreddit] = {
                'yearly': subreddit_data['year'].value_counts().sort_index().to_dict(),
                'monthly': subreddit_data['month'].value_counts().sort_index().to_dict()
            }
        
        return temporal_trends
    
    def analyze_policy_impacts(self, df: pd.DataFrame) -> Dict:
        """Analyze impacts of major policy events"""
        print("Analyzing policy impacts...")
        
        policy_impacts = {}
        
        for event_name, event_date in self.policy_events.items():
            print(f"Analyzing impact of {event_name}...")
            
            event_impacts = {}
            
            for window_name, window_days in self.analysis_windows.items():
                # Define pre and post periods
                pre_start = event_date - timedelta(days=window_days)
                pre_end = event_date
                post_start = event_date
                post_end = event_date + timedelta(days=window_days)
                
                # Filter data for pre and post periods
                pre_data = df[(df['datetime'] >= pre_start) & (df['datetime'] < pre_end)]
                post_data = df[(df['datetime'] >= post_start) & (df['datetime'] < post_end)]
                
                if len(pre_data) == 0 or len(post_data) == 0:
                    continue
                
                # Calculate metrics for each period
                pre_metrics = self.calculate_period_metrics(pre_data)
                post_metrics = self.calculate_period_metrics(post_data)
                
                # Calculate changes
                changes = self.calculate_period_changes(pre_metrics, post_metrics)
                
                event_impacts[window_name] = {
                    'pre_period': {
                        'start': pre_start.isoformat(),
                        'end': pre_end.isoformat(),
                        'metrics': pre_metrics
                    },
                    'post_period': {
                        'start': post_start.isoformat(),
                        'end': post_end.isoformat(),
                        'metrics': post_metrics
                    },
                    'changes': changes
                }
            
            policy_impacts[event_name] = {
                'event_date': event_date.isoformat(),
                'impacts': event_impacts
            }
        
        return policy_impacts
    
    def calculate_period_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific time period"""
        metrics = {}
        
        # Basic counts
        metrics['total_posts'] = len(data)
        metrics['unique_subreddits'] = data['subreddit'].nunique()
        metrics['unique_authors'] = data['author'].nunique() if 'author' in data.columns else 0
        
        # Visa stage distribution
        metrics['visa_stage_distribution'] = data['visa_stage'].value_counts().to_dict()
        
        # Sentiment analysis (if available)
        if 'sentiment' in data.columns:
            metrics['avg_sentiment'] = data['sentiment'].mean()
            metrics['sentiment_std'] = data['sentiment'].std()
            metrics['positive_ratio'] = (data['sentiment'] > 0.1).mean()
            metrics['negative_ratio'] = (data['sentiment'] < -0.1).mean()
        
        # Label predictions (if available)
        for label in self.labels:
            if f'{label}_pred' in data.columns:
                metrics[f'{label}_ratio'] = data[f'{label}_pred'].mean()
            elif label in data.columns:
                metrics[f'{label}_ratio'] = data[label].mean()
        
        # Text characteristics
        metrics['avg_word_count'] = data['word_count'].mean()
        metrics['avg_text_length'] = data['processed_text'].str.len().mean()
        
        return metrics
    
    def calculate_period_changes(self, pre_metrics: Dict, post_metrics: Dict) -> Dict:
        """Calculate changes between pre and post periods"""
        changes = {}
        
        for metric in pre_metrics:
            if metric in post_metrics:
                pre_value = pre_metrics[metric]
                post_value = post_metrics[metric]
                
                if isinstance(pre_value, (int, float)) and isinstance(post_value, (int, float)):
                    if pre_value != 0:
                        change_pct = ((post_value - pre_value) / pre_value) * 100
                    else:
                        change_pct = 0 if post_value == 0 else float('inf')
                    
                    changes[metric] = {
                        'pre_value': pre_value,
                        'post_value': post_value,
                        'absolute_change': post_value - pre_value,
                        'percentage_change': change_pct
                    }
        
        return changes
    
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in discourse"""
        print("Analyzing seasonal patterns...")
        
        seasonal_patterns = {}
        
        # Monthly patterns
        monthly_stats = df.groupby('month').agg({
            'processed_text': 'count',
            'word_count': 'mean',
            'sentiment': 'mean' if 'sentiment' in df.columns else lambda x: 0
        }).to_dict()
        
        seasonal_patterns['monthly'] = monthly_stats
        
        # Quarterly patterns
        quarterly_stats = df.groupby('quarter').agg({
            'processed_text': 'count',
            'word_count': 'mean',
            'sentiment': 'mean' if 'sentiment' in df.columns else lambda x: 0
        }).to_dict()
        
        seasonal_patterns['quarterly'] = quarterly_stats
        
        # Day of week patterns
        weekday_stats = df.groupby('weekday').agg({
            'processed_text': 'count',
            'word_count': 'mean',
            'sentiment': 'mean' if 'sentiment' in df.columns else lambda x: 0
        }).to_dict()
        
        seasonal_patterns['weekday'] = weekday_stats
        
        # Hourly patterns
        hourly_stats = df.groupby('hour').agg({
            'processed_text': 'count',
            'word_count': 'mean',
            'sentiment': 'mean' if 'sentiment' in df.columns else lambda x: 0
        }).to_dict()
        
        seasonal_patterns['hourly'] = hourly_stats
        
        return seasonal_patterns
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect temporal anomalies in discourse patterns"""
        print("Detecting temporal anomalies...")
        
        anomalies = {}
        
        # Daily anomaly detection
        daily_counts = df.groupby(df['datetime'].dt.date).size()
        
        # Calculate rolling statistics
        window_size = 7  # 7-day window
        rolling_mean = daily_counts.rolling(window=window_size, center=True).mean()
        rolling_std = daily_counts.rolling(window=window_size, center=True).std()
        
        # Identify anomalies (values beyond 2 standard deviations)
        threshold = 2
        anomalies_mask = np.abs(daily_counts - rolling_mean) > (threshold * rolling_std)
        daily_anomalies = daily_counts[anomalies_mask]
        
        anomalies['daily'] = {
            'anomaly_dates': daily_anomalies.index.tolist(),
            'anomaly_counts': daily_anomalies.values.tolist(),
            'total_anomalies': len(daily_anomalies)
        }
        
        # Weekly anomaly detection
        weekly_counts = df.groupby(df['datetime'].dt.isocalendar().week).size()
        weekly_rolling_mean = weekly_counts.rolling(window=4, center=True).mean()
        weekly_rolling_std = weekly_counts.rolling(window=4, center=True).std()
        
        weekly_anomalies_mask = np.abs(weekly_counts - weekly_rolling_mean) > (threshold * weekly_rolling_std)
        weekly_anomalies = weekly_counts[weekly_anomalies_mask]
        
        anomalies['weekly'] = {
            'anomaly_weeks': weekly_anomalies.index.tolist(),
            'anomaly_counts': weekly_anomalies.values.tolist(),
            'total_anomalies': len(weekly_anomalies)
        }
        
        return anomalies
    
    def create_temporal_visualizations(self, temporal_data: Dict, output_dir: Path):
        """Create temporal analysis visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall temporal distribution
        if 'overall_distribution' in temporal_data:
            overall = temporal_data['overall_distribution']
            
            # Yearly distribution
            if 'yearly' in overall:
                years = list(overall['yearly'].keys())
                counts = list(overall['yearly'].values())
                
                fig = go.Figure(data=go.Scatter(x=years, y=counts, mode='lines+markers'))
                fig.update_layout(
                    title='Posts Over Time',
                    xaxis_title='Year',
                    yaxis_title='Number of Posts'
                )
                fig.write_html(output_dir / 'yearly_distribution.html')
            
            # Monthly distribution
            if 'monthly' in overall:
                months = list(overall['monthly'].keys())
                counts = list(overall['monthly'].values())
                
                fig = go.Figure(data=go.Bar(x=months, y=counts))
                fig.update_layout(
                    title='Posts by Month',
                    xaxis_title='Month',
                    yaxis_title='Number of Posts'
                )
                fig.write_html(output_dir / 'monthly_distribution.html')
        
        # 2. Policy impact visualizations
        if 'policy_impacts' in temporal_data:
            policy_impacts = temporal_data['policy_impacts']
            
            for event_name, event_data in policy_impacts.items():
                # Create impact summary
                impact_summary = []
                for window_name, window_data in event_data['impacts'].items():
                    if 'changes' in window_data:
                        for metric, change_data in window_data['changes'].items():
                            if isinstance(change_data, dict) and 'percentage_change' in change_data:
                                impact_summary.append({
                                    'window': window_name,
                                    'metric': metric,
                                    'change_pct': change_data['percentage_change']
                                })
                
                if impact_summary:
                    impact_df = pd.DataFrame(impact_summary)
                    
                    # Create heatmap of changes
                    pivot_df = impact_df.pivot(index='metric', columns='window', values='change_pct')
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(
                        title=f'Policy Impact: {event_name}',
                        xaxis_title='Analysis Window',
                        yaxis_title='Metrics'
                    )
                    fig.write_html(output_dir / f'policy_impact_{event_name}.html')
        
        # 3. Seasonal patterns
        if 'seasonal_patterns' in temporal_data:
            seasonal = temporal_data['seasonal_patterns']
            
            # Monthly patterns
            if 'monthly' in seasonal:
                months = list(range(1, 13))
                counts = [seasonal['monthly']['processed_text'].get(month, 0) for month in months]
                
                fig = go.Figure(data=go.Scatter(x=months, y=counts, mode='lines+markers'))
                fig.update_layout(
                    title='Seasonal Patterns - Monthly',
                    xaxis_title='Month',
                    yaxis_title='Number of Posts'
                )
                fig.write_html(output_dir / 'seasonal_monthly.html')
    
    def run_comprehensive_temporal_analysis(self) -> Dict:
        """Run comprehensive temporal analysis"""
        print("Starting comprehensive temporal analysis...")
        
        # Load data
        df = self.load_data_with_predictions()
        print(f"Loaded {len(df)} records for temporal analysis")
        
        # Prepare temporal data
        df = self.prepare_temporal_data(df)
        
        # Run all analyses
        temporal_results = {}
        
        # Temporal trends
        temporal_results['temporal_trends'] = self.analyze_temporal_trends(df)
        
        # Policy impacts
        temporal_results['policy_impacts'] = self.analyze_policy_impacts(df)
        
        # Seasonal patterns
        temporal_results['seasonal_patterns'] = self.analyze_seasonal_patterns(df)
        
        # Anomaly detection
        temporal_results['anomalies'] = self.detect_anomalies(df)
        
        # Create visualizations
        viz_dir = config.pre_post_analysis_dir / "temporal_visualizations"
        self.create_temporal_visualizations(temporal_results, viz_dir)
        
        # Save results
        results_file = config.pre_post_analysis_dir / "temporal_analysis_results.json"
        config.pre_post_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(temporal_results, f, indent=2, default=str)
        
        print(f"Temporal analysis results saved to {results_file}")
        print(f"Visualizations saved to {viz_dir}")
        
        # Print summary
        self.print_temporal_summary(temporal_results)
        
        return temporal_results
    
    def print_temporal_summary(self, results: Dict):
        """Print summary of temporal analysis results"""
        print("\n=== TEMPORAL ANALYSIS SUMMARY ===")
        
        # Temporal trends
        if 'temporal_trends' in results:
            trends = results['temporal_trends']
            if 'overall_distribution' in trends:
                overall = trends['overall_distribution']
                if 'yearly' in overall:
                    yearly = overall['yearly']
                    print(f"Yearly distribution: {len(yearly)} years covered")
                    print(f"Peak year: {max(yearly, key=yearly.get)} with {max(yearly.values())} posts")
        
        # Policy impacts
        if 'policy_impacts' in results:
            impacts = results['policy_impacts']
            print(f"Analyzed {len(impacts)} policy events")
            
            for event_name, event_data in impacts.items():
                print(f"\n{event_name}:")
                for window_name, window_data in event_data['impacts'].items():
                    if 'changes' in window_data:
                        significant_changes = [
                            metric for metric, change_data in window_data['changes'].items()
                            if isinstance(change_data, dict) and 
                            abs(change_data.get('percentage_change', 0)) > 10
                        ]
                        if significant_changes:
                            print(f"  {window_name}: {len(significant_changes)} significant changes")
        
        # Anomalies
        if 'anomalies' in results:
            anomalies = results['anomalies']
            if 'daily' in anomalies:
                daily_anomalies = anomalies['daily']
                print(f"Daily anomalies detected: {daily_anomalies['total_anomalies']}")
            if 'weekly' in anomalies:
                weekly_anomalies = anomalies['weekly']
                print(f"Weekly anomalies detected: {weekly_anomalies['total_anomalies']}")

def main():
    """Main function for temporal analysis"""
    analyzer = RedditTemporalAnalyzer()
    results = analyzer.run_comprehensive_temporal_analysis()
    return results

if __name__ == "__main__":
    main()
