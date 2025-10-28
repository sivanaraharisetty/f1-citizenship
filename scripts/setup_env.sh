#!/bin/bash
# Setup Environment Script
# Sets up the development environment for the BERT Classifier project

set -e  # Exit on any error

echo "Setting up BERT Classifier Environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install development requirements
if [ -f "requirements_dev.txt" ]; then
    echo "Installing development requirements..."
    pip install -r requirements_dev.txt
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p data/{raw,interim,processed,external}
mkdir -p results/{metrics,visualizations,reports,exports}
mkdir -p models/{checkpoints,tokenizer}

# Set up environment variables
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp config/credentials_template.py .env
    echo "Please edit .env file with your actual credentials"
fi

# Download spaCy model (optional)
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm || echo "spaCy model download failed (optional)"

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

echo "Environment setup complete!"
echo "Next steps:"
echo "   1. Edit .env file with your credentials"
echo "   2. Run: source venv/bin/activate"
echo "   3. Run: ./scripts/run_pipeline.sh"
