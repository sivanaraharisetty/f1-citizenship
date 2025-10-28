"""
Data Cleaning Module for Reddit Visa Discourse Analysis
Handles text preprocessing, normalization, and cleaning
"""
import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob
import emoji
from tqdm import tqdm
import unicodedata

from config import config

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class RedditDataCleaner:
    """Handles comprehensive text cleaning and preprocessing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load spaCy model, fallback to basic processing if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Using basic text processing.")
            self.nlp = None
        
        # Reddit-specific patterns
        self.reddit_patterns = {
            'user_mentions': r'u/[A-Za-z0-9_-]+',
            'subreddit_mentions': r'r/[A-Za-z0-9_-]+',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'reddit_links': r'\[([^\]]+)\]\(https?://[^\)]+\)',
            'code_blocks': r'```[\s\S]*?```',
            'inline_code': r'`[^`]+`',
            'bold_text': r'\*\*([^*]+)\*\*',
            'italic_text': r'\*([^*]+)\*',
            'strikethrough': r'~~([^~]+)~~',
            'headers': r'^#{1,6}\s+',
            'list_items': r'^\s*[-*+]\s+',
            'quotes': r'^>\s*',
        }
        
        # Emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
    
    def clean_text(self, text: str) -> str:
        """Main text cleaning function"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Basic cleaning
        text = self.remove_reddit_formatting(text)
        text = self.remove_urls_and_mentions(text)
        text = self.normalize_whitespace(text)
        text = self.handle_emojis(text)
        text = self.remove_special_characters(text)
        text = self.normalize_unicode(text)
        
        return text.strip()
    
    def remove_reddit_formatting(self, text: str) -> str:
        """Remove Reddit-specific formatting"""
        for pattern_name, pattern in self.reddit_patterns.items():
            if pattern_name in ['bold_text', 'italic_text', 'strikethrough']:
                # Keep the content, remove formatting
                text = re.sub(pattern, r'\1', text)
            else:
                # Remove the entire match
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text
    
    def remove_urls_and_mentions(self, text: str) -> str:
        """Remove URLs and user/subreddit mentions"""
        # Remove URLs
        text = re.sub(self.reddit_patterns['urls'], '', text)
        # Remove user mentions
        text = re.sub(self.reddit_patterns['user_mentions'], '', text)
        # Remove subreddit mentions
        text = re.sub(self.reddit_patterns['subreddit_mentions'], '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def handle_emojis(self, text: str) -> str:
        """Handle emojis based on configuration"""
        if not config.handle_emojis:
            # Remove emojis
            text = self.emoji_pattern.sub('', text)
        else:
            # Convert emojis to text descriptions
            text = emoji.demojize(text, delimiters=(" ", " "))
        
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """Remove or replace special characters"""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-ASCII characters if needed
        text = ''.join(char for char in text if ord(char) < 128 or char.isspace())
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        if self.nlp:
            # Use spaCy for better tokenization
            doc = self.nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_space]
        else:
            # Fallback to NLTK
            tokens = word_tokenize(text.lower())
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokenized text"""
        if not config.remove_stopwords:
            return tokens
        
        # Add custom stopwords for immigration context
        custom_stopwords = {
            'reddit', 'post', 'comment', 'thread', 'subreddit', 'user',
            'visa', 'immigration', 'uscis', 'h1b', 'f1', 'green', 'card'
        }
        
        all_stopwords = self.stop_words.union(custom_stopwords)
        
        return [token for token in tokens if token not in all_stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens by length and content"""
        filtered = []
        for token in tokens:
            # Keep tokens that are:
            # - At least 2 characters long
            # - Not just numbers
            # - Not just punctuation
            if (len(token) >= 2 and 
                not token.isdigit() and 
                not all(c in string.punctuation for c in token)):
                filtered.append(token)
        
        return filtered
    
    def preprocess_text(self, text: str) -> Dict[str, any]:
        """Complete text preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Check if text meets minimum length requirement
        if len(cleaned_text) < config.min_text_length:
            return {
                'cleaned_text': '',
                'tokens': [],
                'processed_text': '',
                'word_count': 0,
                'is_valid': False
            }
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Reconstruct processed text
        processed_text = ' '.join(tokens)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'processed_text': processed_text,
            'word_count': len(tokens),
            'is_valid': len(tokens) > 0
        }
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean entire dataframe"""
        print("Starting data cleaning...")
        
        # Initialize result columns
        df['cleaned_text'] = ''
        df['processed_text'] = ''
        df['word_count'] = 0
        df['is_valid'] = False
        df['tokens'] = ''
        
        # Process each text
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning texts"):
            text = row.get('text', '')
            if pd.isna(text):
                continue
            
            # Preprocess text
            result = self.preprocess_text(str(text))
            
            # Update dataframe
            df.at[idx, 'cleaned_text'] = result['cleaned_text']
            df.at[idx, 'processed_text'] = result['processed_text']
            df.at[idx, 'word_count'] = result['word_count']
            df.at[idx, 'is_valid'] = result['is_valid']
            df.at[idx, 'tokens'] = ' '.join(result['tokens'])
        
        # Remove invalid entries
        original_len = len(df)
        df = df[df['is_valid']]
        removed_count = original_len - len(df)
        
        print(f"Cleaning complete. Removed {removed_count} invalid entries.")
        print(f"Valid entries: {len(df)}")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, 
                        subset: List[str] = None) -> pd.DataFrame:
        """Remove duplicate entries"""
        if subset is None:
            subset = ['text', 'subreddit']
        
        original_len = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        removed_count = original_len - len(df)
        
        print(f"Removed {removed_count} duplicate entries.")
        return df
    
    def handle_missing_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing fields in the dataset"""
        # Fill missing subreddit with 'unknown'
        df['subreddit'] = df['subreddit'].fillna('unknown')
        
        # Fill missing author with 'unknown'
        if 'author' in df.columns:
            df['author'] = df['author'].fillna('unknown')
        
        # Fill missing created_utc with current timestamp
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
            df['created_utc'] = df['created_utc'].fillna(pd.Timestamp.now().timestamp())
        
        return df
    
    def get_cleaning_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of cleaning process"""
        return {
            'total_entries': len(df),
            'valid_entries': df['is_valid'].sum(),
            'avg_word_count': df['word_count'].mean(),
            'unique_subreddits': df['subreddit'].nunique(),
            'text_length_distribution': {
                'min': df['word_count'].min(),
                'max': df['word_count'].max(),
                'median': df['word_count'].median(),
                'std': df['word_count'].std()
            }
        }

def main():
    """Main function for data cleaning"""
    import pandas as pd
    from pathlib import Path
    
    # Load sampled data
    sampled_file = config.sampled_data_dir / "sampled_reddit_data.parquet"
    if not sampled_file.exists():
        print("No sampled data found. Please run data_sampling.py first.")
        return
    
    df = pd.read_parquet(sampled_file)
    print(f"Loaded {len(df)} samples for cleaning")
    
    # Initialize cleaner
    cleaner = RedditDataCleaner()
    
    # Clean data
    cleaned_df = cleaner.clean_dataframe(df)
    
    # Remove duplicates
    cleaned_df = cleaner.remove_duplicates(cleaned_df)
    
    # Handle missing fields
    cleaned_df = cleaner.handle_missing_fields(cleaned_df)
    
    # Save cleaned data
    output_file = config.cleaned_data_dir / "cleaned_reddit_data.parquet"
    config.cleaned_data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(output_file, index=False)
    
    print(f"Cleaned data saved to: {output_file}")
    
    # Print summary
    summary = cleaner.get_cleaning_summary(cleaned_df)
    print("\n=== CLEANING SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
