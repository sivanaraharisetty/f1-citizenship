#!/usr/bin/env python3
"""
Setup script for Immigration Classification Pipeline
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="immigration-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive machine learning pipeline for classifying immigration-related content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/immigration-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "immigration-classifier=scripts.run_enhanced_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/*.json"],
    },
)
