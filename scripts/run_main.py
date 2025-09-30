#!/usr/bin/env python3
"""
Entry point for running the main immigration classification pipeline
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the main function
from src.main import main

if __name__ == "__main__":
    main()
