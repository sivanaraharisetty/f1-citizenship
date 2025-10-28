# Credentials Template
# Placeholder for API keys (real keys should be in environment variables)

import os
from pathlib import Path

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'your_access_key_here')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'your_secret_key_here')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'siva-test-9')
S3_PREFIX_2024 = os.getenv('S3_PREFIX_2024', 'reddit/new_data/2024_data/')
S3_PREFIX_2025 = os.getenv('S3_PREFIX_2025', 'reddit/new_data/2025_data/')
S3_RESULTS_PREFIX = os.getenv('S3_RESULTS_PREFIX', 'reddit/new_data/results/')

# Reddit API Configuration (if needed)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'your_reddit_client_id')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'your_reddit_client_secret')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'BERT_Classifier_Bot/1.0')

# Database Configuration (if needed)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data.db')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/pipeline.log')

# Model Configuration
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', 'models/cache')
TRANSFORMERS_CACHE = os.getenv('TRANSFORMERS_CACHE', 'models/transformers_cache')

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['AWS_DEFAULT_REGION'] = AWS_DEFAULT_REGION
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
