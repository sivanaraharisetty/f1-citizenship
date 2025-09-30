#!/usr/bin/env python3
"""
Model Training Script
Trains BERT classifier on processed data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import glob
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.model.train import BertClassifierTrainer

def train_model(input_dir="data/processed", output_dir="models", epochs=3):
    """Train BERT classifier on processed data"""
    print(f"🤖 Training model on data from {input_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all processed files
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not input_files:
        print(f" No processed files found in {input_dir}")
        return False
    
    # Load and combine all data
    all_data = []
    for input_file in input_files:
        print(f"📂 Loading {input_file}...")
        df = pd.read_parquet(input_file)
        all_data.append(df)
    
    if not all_data:
        print(" No data to train on")
        return False
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f" Total samples: {len(combined_df)}")
    print(f" Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    # Create label mapping
    unique_labels = combined_df['label'].unique()
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    
    # Encode labels
    combined_df['label_encoded'] = combined_df['label'].map(label_mapping).astype(int)
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df['text'], 
        combined_df['label_encoded'], 
        test_size=0.2, 
        random_state=42,
        stratify=combined_df['label_encoded']
    )
    
    # Create training DataFrame
    train_df = pd.DataFrame({
        'text': X_train,
        'label': y_train
    })
    
    print(f" Training BERT classifier on {len(train_df)} samples...")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Number of labels: {len(label_mapping)}")
    
    # Train classifier
    trainer = BertClassifierTrainer(num_labels=len(label_mapping))
    
    try:
        metrics = trainer.train_on_dataframe(
            train_df, 
            epochs=epochs, 
            save_dir=output_dir
        )
        
        print(f" Training complete!")
        print(f" Training metrics: {metrics}")
        
        # Save model results
        model_results = {
            'training_metrics': metrics,
            'label_mapping': label_mapping,
            'train_samples': len(train_df),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "model_training_results.json"), 'w') as f:
            json.dump(model_results, f, indent=2)
        
        print(f"💾 Model saved to: {output_dir}")
        print(f"💾 Results saved to: {os.path.join(output_dir, 'model_training_results.json')}")
        
        return True
        
    except Exception as e:
        print(f" Training error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train BERT classifier")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Train model
    success = train_model(args.input, args.output, args.epochs)
    
    if success:
        print(" Model training completed successfully!")
    else:
        print(" Model training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
