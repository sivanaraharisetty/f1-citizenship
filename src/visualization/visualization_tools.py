"""
Comprehensive Visualization Tools for Reddit Visa Discourse Analysis
Creates interactive and static visualizations for all analysis results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import plotly.figure_factory as ff
from wordcloud import WordCloud
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from collections import Counter, defaultdict
import warnings
from datetime import datetime
import streamlit as st

from config import config

class RedditVisualizationTools:
    """Comprehensive visualization tools for Reddit visa discourse analysis"""
    
    def __init__(self):
        self.labels = config.labels
        self.visa_stages = config.visa_stages
        self.colors = {
            'fear': '#FF6B6B',
            'question': '#4ECDC4',
            'fear_driven_question': '#FFE66D',
            'other': '#95A5A6',
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C'
        }
    
    def create_dashboard(self, data: Dict, output_dir: Path):
        """Create comprehensive dashboard with all visualizations"""
        print("Creating comprehensive dashboard...")
        
        # Set up the dashboard
        dashboard_html = self._create_dashboard_html(data)
        
        # Save dashboard
        dashboard_file = output_dir / "comprehensive_dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Dashboard saved to {dashboard_file}")
        return dashboard_file
    
    def _create_dashboard_html(self, data: Dict) -> str:
        """Create HTML dashboard"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reddit Visa Discourse Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reddit Visa Discourse Analysis Dashboard</h1>
                <p>Comprehensive analysis of visa-related discourse patterns</p>
            </div>
            
            <div class="section">
                <h2>Overview Statistics</h2>
                <div id="overview-stats"></div>
            </div>
            
            <div class="section">
                <h2>Label Distribution</h2>
                <div id="label-distribution"></div>
            </div>
            
            <div class="section">
                <h2>Temporal Trends</h2>
                <div id="temporal-trends"></div>
            </div>
            
            <div class="section">
                <h2>Visa Stage Analysis</h2>
                <div id="visa-stage-analysis"></div>
            </div>
            
            <div class="section">
                <h2>Sentiment Analysis</h2>
                <div id="sentiment-analysis"></div>
            </div>
            
            <div class="section">
                <h2>Policy Impact Analysis</h2>
                <div id="policy-impact"></div>
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
                <div id="model-performance"></div>
            </div>
        </body>
        </html>
        """
        return html
    
    def create_label_distribution_plots(self, data: Dict, output_dir: Path):
        """Create label distribution visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotation data if available
        annotation_file = config.annotation_dir / "exported_annotations.parquet"
        if annotation_file.exists():
            df = pd.read_parquet(annotation_file)
            
            # Label distribution pie chart
            label_counts = df['labels'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=label_counts.index,
                values=label_counts.values,
                hole=0.3
            )])
            fig.update_layout(
                title="Label Distribution",
                font_size=12
            )
            fig.write_html(output_dir / "label_distribution_pie.html")
            
            # Label distribution by visa stage
            if 'visa_stage' in df.columns:
                stage_label_cross = pd.crosstab(df['visa_stage'], df['labels'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=stage_label_cross.values,
                    x=stage_label_cross.columns,
                    y=stage_label_cross.index,
                    colorscale='Blues'
                ))
                fig.update_layout(
                    title="Label Distribution by Visa Stage",
                    xaxis_title="Labels",
                    yaxis_title="Visa Stage"
                )
                fig.write_html(output_dir / "label_stage_heatmap.html")
    
    def create_temporal_visualizations(self, temporal_data: Dict, output_dir: Path):
        """Create temporal trend visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Yearly trends
        if 'temporal_trends' in temporal_data and 'overall_distribution' in temporal_data['temporal_trends']:
            overall = temporal_data['temporal_trends']['overall_distribution']
            
            if 'yearly' in overall:
                years = list(overall['yearly'].keys())
                counts = list(overall['yearly'].values())
                
                fig = go.Figure(data=go.Scatter(
                    x=years, y=counts,
                    mode='lines+markers',
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title="Posts Over Time",
                    xaxis_title="Year",
                    yaxis_title="Number of Posts",
                    template="plotly_white"
                )
                fig.write_html(output_dir / "yearly_trends.html")
            
            # Monthly patterns
            if 'monthly' in overall:
                months = list(overall['monthly'].keys())
                counts = list(overall['monthly'].values())
                
                fig = go.Figure(data=go.Bar(
                    x=months, y=counts,
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="Posts by Month",
                    xaxis_title="Month",
                    yaxis_title="Number of Posts",
                    template="plotly_white"
                )
                fig.write_html(output_dir / "monthly_patterns.html")
    
    def create_visa_stage_analysis(self, data: Dict, output_dir: Path):
        """Create visa stage analysis visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cleaned data
        cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
        if cleaned_file.exists():
            df = pd.read_parquet(cleaned_file)
            
            # Visa stage distribution
            stage_counts = df['visa_stage'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=stage_counts.index,
                values=stage_counts.values,
                hole=0.4
            )])
            fig.update_layout(
                title="Distribution by Visa Stage",
                font_size=12
            )
            fig.write_html(output_dir / "visa_stage_distribution.html")
            
            # Visa stage by subreddit
            if 'subreddit' in df.columns:
                stage_subreddit_cross = pd.crosstab(df['visa_stage'], df['subreddit'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=stage_subreddit_cross.values,
                    x=stage_subreddit_cross.columns,
                    y=stage_subreddit_cross.index,
                    colorscale='Viridis'
                ))
                fig.update_layout(
                    title="Visa Stage by Subreddit",
                    xaxis_title="Subreddit",
                    yaxis_title="Visa Stage"
                )
                fig.write_html(output_dir / "visa_stage_subreddit_heatmap.html")
    
    def create_sentiment_visualizations(self, data: Dict, output_dir: Path):
        """Create sentiment analysis visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data with sentiment
        cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
        if cleaned_file.exists():
            df = pd.read_parquet(cleaned_file)
            
            if 'sentiment' in df.columns:
                # Sentiment distribution
                fig = go.Figure(data=[go.Histogram(
                    x=df['sentiment'],
                    nbinsx=50,
                    marker_color='lightgreen'
                )])
                fig.update_layout(
                    title="Sentiment Distribution",
                    xaxis_title="Sentiment Score",
                    yaxis_title="Frequency",
                    template="plotly_white"
                )
                fig.write_html(output_dir / "sentiment_distribution.html")
                
                # Sentiment by visa stage
                if 'visa_stage' in df.columns:
                    sentiment_by_stage = df.groupby('visa_stage')['sentiment'].agg(['mean', 'std']).reset_index()
                    
                    fig = go.Figure(data=go.Bar(
                        x=sentiment_by_stage['visa_stage'],
                        y=sentiment_by_stage['mean'],
                        error_y=dict(type='data', array=sentiment_by_stage['std'])
                    ))
                    fig.update_layout(
                        title="Average Sentiment by Visa Stage",
                        xaxis_title="Visa Stage",
                        yaxis_title="Average Sentiment",
                        template="plotly_white"
                    )
                    fig.write_html(output_dir / "sentiment_by_stage.html")
    
    def create_model_performance_plots(self, metrics: Dict, output_dir: Path):
        """Create model performance visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'basic_metrics' in metrics:
            basic_metrics = metrics['basic_metrics']
            
            # Precision, Recall, F1 comparison
            if 'per_label' in basic_metrics:
                labels = list(basic_metrics['per_label'].keys())
                precision = [basic_metrics['per_label'][label]['precision'] for label in labels]
                recall = [basic_metrics['per_label'][label]['recall'] for label in labels]
                f1 = [basic_metrics['per_label'][label]['f1_score'] for label in labels]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Precision', x=labels, y=precision))
                fig.add_trace(go.Bar(name='Recall', x=labels, y=recall))
                fig.add_trace(go.Bar(name='F1 Score', x=labels, y=f1))
                
                fig.update_layout(
                    title="Model Performance by Label",
                    xaxis_title="Labels",
                    yaxis_title="Score",
                    barmode='group',
                    template="plotly_white"
                )
                fig.write_html(output_dir / "model_performance.html")
            
            # Confusion matrices
            if 'confusion_matrices' in metrics:
                confusion_matrices = metrics['confusion_matrices']
                
                for label, cm_data in confusion_matrices.items():
                    cm = np.array(cm_data['matrix'])
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted 0', 'Predicted 1'],
                        y=['Actual 0', 'Actual 1'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 20}
                    ))
                    fig.update_layout(
                        title=f'Confusion Matrix - {label}',
                        xaxis_title='Predicted Label',
                        yaxis_title='True Label'
                    )
                    fig.write_html(output_dir / f'confusion_matrix_{label}.html')
    
    def create_policy_impact_visualizations(self, temporal_data: Dict, output_dir: Path):
        """Create policy impact visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'policy_impacts' in temporal_data:
            policy_impacts = temporal_data['policy_impacts']
            
            # Create impact summary for all events
            all_impacts = []
            for event_name, event_data in policy_impacts.items():
                for window_name, window_data in event_data['impacts'].items():
                    if 'changes' in window_data:
                        for metric, change_data in window_data['changes'].items():
                            if isinstance(change_data, dict) and 'percentage_change' in change_data:
                                all_impacts.append({
                                    'event': event_name,
                                    'window': window_name,
                                    'metric': metric,
                                    'change_pct': change_data['percentage_change']
                                })
            
            if all_impacts:
                impact_df = pd.DataFrame(all_impacts)
                
                # Impact heatmap
                pivot_df = impact_df.pivot_table(
                    index='metric', 
                    columns=['event', 'window'], 
                    values='change_pct', 
                    aggfunc='mean'
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_df.values,
                    x=[f"{col[0]}_{col[1]}" for col in pivot_df.columns],
                    y=pivot_df.index,
                    colorscale='RdBu',
                    zmid=0
                ))
                fig.update_layout(
                    title="Policy Impact Analysis",
                    xaxis_title="Event & Window",
                    yaxis_title="Metrics"
                )
                fig.write_html(output_dir / "policy_impact_heatmap.html")
    
    def create_word_clouds(self, data: Dict, output_dir: Path):
        """Create word cloud visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cleaned data
        cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
        if cleaned_file.exists():
            df = pd.read_parquet(cleaned_file)
            
            # Overall word cloud
            all_text = ' '.join(df['processed_text'].dropna().astype(str))
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100
            ).generate(all_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Overall Word Cloud')
            plt.tight_layout()
            plt.savefig(output_dir / 'overall_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Word clouds by visa stage
            for stage in df['visa_stage'].unique():
                if stage == 'unknown':
                    continue
                
                stage_text = ' '.join(df[df['visa_stage'] == stage]['processed_text'].dropna().astype(str))
                if stage_text:
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        max_words=50
                    ).generate(stage_text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Word Cloud - {stage}')
                    plt.tight_layout()
                    plt.savefig(output_dir / f'wordcloud_{stage}.png', dpi=300, bbox_inches='tight')
                    plt.close()
    
    def create_network_visualizations(self, data: Dict, output_dir: Path):
        """Create network visualizations for subreddit relationships"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        cleaned_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
        if cleaned_file.exists():
            df = pd.read_parquet(cleaned_file)
            
            # Create subreddit co-occurrence network
            if 'subreddit' in df.columns and 'visa_stage' in df.columns:
                # Calculate co-occurrence between subreddits and visa stages
                co_occurrence = pd.crosstab(df['subreddit'], df['visa_stage'])
                
                # Create network graph
                G = nx.Graph()
                
                # Add nodes for subreddits
                for subreddit in co_occurrence.index:
                    G.add_node(subreddit, node_type='subreddit')
                
                # Add nodes for visa stages
                for stage in co_occurrence.columns:
                    G.add_node(stage, node_type='visa_stage')
                
                # Add edges based on co-occurrence
                for subreddit in co_occurrence.index:
                    for stage in co_occurrence.columns:
                        weight = co_occurrence.loc[subreddit, stage]
                        if weight > 0:
                            G.add_edge(subreddit, stage, weight=weight)
                
                # Create network visualization
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Separate nodes by type
                subreddit_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'subreddit']
                stage_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'visa_stage']
                
                # Create plotly network
                edge_x = []
                edge_y = []
                edge_info = []
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_info.append(f"Weight: {G[edge[0]][edge[1]]['weight']}")
                
                # Create edge trace
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create node traces
                subreddit_trace = go.Scatter(
                    x=[pos[node][0] for node in subreddit_nodes],
                    y=[pos[node][1] for node in subreddit_nodes],
                    mode='markers',
                    hoverinfo='text',
                    text=subreddit_nodes,
                    marker=dict(size=10, color='lightblue')
                )
                
                stage_trace = go.Scatter(
                    x=[pos[node][0] for node in stage_nodes],
                    y=[pos[node][1] for node in stage_nodes],
                    mode='markers',
                    hoverinfo='text',
                    text=stage_nodes,
                    marker=dict(size=15, color='lightcoral')
                )
                
                # Create figure
                fig = go.Figure(data=[edge_trace, subreddit_trace, stage_trace])
                fig.update_layout(
                    title="Subreddit-Visa Stage Network",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Network visualization of subreddit-visa stage relationships",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="black", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                fig.write_html(output_dir / "network_visualization.html")
    
    def create_comprehensive_report(self, all_data: Dict, output_dir: Path):
        """Create comprehensive analysis report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create all visualizations
        self.create_label_distribution_plots(all_data, output_dir / "label_analysis")
        self.create_temporal_visualizations(all_data, output_dir / "temporal_analysis")
        self.create_visa_stage_analysis(all_data, output_dir / "visa_stage_analysis")
        self.create_sentiment_visualizations(all_data, output_dir / "sentiment_analysis")
        self.create_model_performance_plots(all_data, output_dir / "model_performance")
        self.create_policy_impact_visualizations(all_data, output_dir / "policy_analysis")
        self.create_word_clouds(all_data, output_dir / "word_clouds")
        self.create_network_visualizations(all_data, output_dir / "network_analysis")
        
        # Create main dashboard
        dashboard_file = self.create_dashboard(all_data, output_dir)
        
        print(f"Comprehensive report created in {output_dir}")
        print(f"Main dashboard: {dashboard_file}")
        
        return dashboard_file
    
    def create_streamlit_app(self, data: Dict, output_dir: Path):
        """Create Streamlit interactive dashboard"""
        app_code = f"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Load data
@st.cache_data
def load_data():
    # Load your data here
    return data

def main():
    st.set_page_config(
        page_title="Reddit Visa Discourse Analysis",
        page_icon="",
        layout="wide"
    )
    
    st.title("Reddit Visa Discourse Analysis Dashboard")
    st.markdown("Comprehensive analysis of visa-related discourse patterns")
    
    # Load data
    data = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Label Analysis", "Temporal Analysis", "Visa Stage Analysis", 
         "Sentiment Analysis", "Model Performance", "Policy Impact"]
    )
    
    if page == "Overview":
        st.header("Overview")
        # Add overview content
        
    elif page == "Label Analysis":
        st.header("Label Analysis")
        # Add label analysis content
        
    elif page == "Temporal Analysis":
        st.header("Temporal Analysis")
        # Add temporal analysis content
        
    elif page == "Visa Stage Analysis":
        st.header("Visa Stage Analysis")
        # Add visa stage analysis content
        
    elif page == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        # Add sentiment analysis content
        
    elif page == "Model Performance":
        st.header("Model Performance")
        # Add model performance content
        
    elif page == "Policy Impact":
        st.header("Policy Impact Analysis")
        # Add policy impact content

if __name__ == "__main__":
    main()
"""
        
        app_file = output_dir / "streamlit_app.py"
        with open(app_file, 'w') as f:
            f.write(app_code)
        
        print(f"Streamlit app created: {app_file}")
        return app_file

def main():
    """Main function for visualization tools"""
    # Example usage
    viz_tools = RedditVisualizationTools()
    
    # Create sample data structure
    sample_data = {
        'basic_metrics': {
            'overall': {'precision': 0.85, 'recall': 0.82, 'f1_score': 0.83},
            'per_label': {
                'fear': {'precision': 0.88, 'recall': 0.85, 'f1_score': 0.86},
                'question': {'precision': 0.82, 'recall': 0.80, 'f1_score': 0.81}
            }
        },
        'temporal_trends': {
            'overall_distribution': {
                'yearly': {2020: 1000, 2021: 1200, 2022: 1100},
                'monthly': {1: 100, 2: 120, 3: 110}
            }
        }
    }
    
    # Create visualizations
    output_dir = config.visualizations_dir
    viz_tools.create_comprehensive_report(sample_data, output_dir)
    
    return output_dir

if __name__ == "__main__":
    main()
