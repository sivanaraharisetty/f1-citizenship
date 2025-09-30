#!/usr/bin/env python3
"""
Test preprocessing functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from src.data.preprocess import preprocess_data

def test_preprocess_data():
    """Test basic preprocessing functionality"""
    # Create test data
    test_data = pd.DataFrame({
        'text': [
            "F1 visa application process",
            "H1B work visa requirements", 
            "Green card application timeline",
            "Weather forecast for today",
            "Best restaurants in town"
        ],
        'subreddit': ['f1visa', 'h1b', 'greencard', 'weather', 'food']
    })
    
    # Preprocess
    processed = preprocess_data(test_data)
    
    # Check basic properties
    assert len(processed) > 0
    assert 'text' in processed.columns
    assert 'label' in processed.columns
    
    # Check label distribution
    label_counts = processed['label'].value_counts()
    assert len(label_counts) > 0
    
    print(f" Preprocessing test passed")
    print(f"   Processed {len(processed)} samples")
    print(f"   Label distribution: {label_counts.to_dict()}")

def test_immigration_patterns():
    """Test immigration-specific pattern matching"""
    # Test data with specific immigration terms
    test_data = pd.DataFrame({
        'text': [
            "F1 visa application process",
            "H1B work visa requirements", 
            "Green card application timeline",
            "Citizenship naturalization process",
            "Weather forecast for today"
        ],
        'subreddit': ['f1visa', 'h1b', 'greencard', 'citizenship', 'weather']
    })
    
    # Preprocess
    processed = preprocess_data(test_data)
    
    # Check that immigration terms are properly labeled
    immigration_labels = processed[processed['text'].str.contains('visa|green card|citizenship', case=False)]['label']
    assert len(immigration_labels) > 0
    
    # Check that non-immigration content is labeled as irrelevant
    irrelevant_labels = processed[processed['text'].str.contains('weather', case=False)]['label']
    assert all(label == 'irrelevant' for label in irrelevant_labels)
    
    print(f" Immigration patterns test passed")

if __name__ == "__main__":
    test_preprocess_data()
    test_immigration_patterns()
    print(" All preprocessing tests passed!")
