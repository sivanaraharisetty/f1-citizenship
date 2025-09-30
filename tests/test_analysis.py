#!/usr/bin/env python3
"""
Test analysis functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from src.analysis.descriptive_analysis import ImmigrationDataAnalyzer

def test_descriptive_analysis():
    """Test descriptive analysis functionality"""
    # Create test data
    test_data = pd.DataFrame({
        'text': [
            "F1 visa application process",
            "H1B work visa requirements", 
            "Green card application timeline",
            "Citizenship naturalization process",
            "Immigration lawyer consultation",
            "Weather forecast for today",
            "Best restaurants in town"
        ],
        'label': ['general_immigration', 'general_immigration', 'green_card', 
                 'general_immigration', 'general_immigration', 'irrelevant', 'irrelevant']
    })
    
    # Test analyzer
    analyzer = ImmigrationDataAnalyzer()
    
    # Test keyword extraction
    keywords = analyzer.extract_keywords(test_data['text'], top_n=10)
    assert len(keywords) > 0
    assert isinstance(keywords, list)
    
    # Test immigration keyword analysis
    immigration_analysis = analyzer.analyze_immigration_keywords(test_data['text'])
    assert len(immigration_analysis) > 0
    assert isinstance(immigration_analysis, dict)
    
    # Test topic clustering
    cluster_analysis, cluster_labels = analyzer.cluster_topics(test_data['text'], n_clusters=3)
    assert len(cluster_analysis) > 0
    assert len(cluster_labels) == len(test_data)
    
    print(f" Descriptive analysis test passed")

def test_analysis_comprehensive():
    """Test comprehensive analysis"""
    # Create test data
    test_data = pd.DataFrame({
        'text': [
            "F1 visa application process",
            "H1B work visa requirements", 
            "Green card application timeline",
            "Citizenship naturalization process",
            "Immigration lawyer consultation",
            "Weather forecast for today"
        ],
        'label': ['general_immigration', 'general_immigration', 'green_card', 
                 'general_immigration', 'general_immigration', 'irrelevant']
    })
    
    # Test comprehensive analysis
    analyzer = ImmigrationDataAnalyzer()
    analysis_results, df_with_clusters = analyzer.comprehensive_analysis(test_data)
    
    # Check basic results
    assert 'basic_stats' in analysis_results
    assert 'top_keywords' in analysis_results
    assert 'immigration_keywords' in analysis_results
    assert 'topic_clusters' in analysis_results
    
    # Check that clusters were added
    assert 'cluster' in df_with_clusters.columns
    
    print(f" Comprehensive analysis test passed")

if __name__ == "__main__":
    test_descriptive_analysis()
    test_analysis_comprehensive()
    print(" All analysis tests passed!")
