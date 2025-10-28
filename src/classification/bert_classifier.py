"""
BERT-based Multi-label Classifier for Reddit Visa Discourse Analysis
Implements BERT/DistilBERT/RoBERTa models for fear, Q&A, and fear-driven question detection
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from tqdm import tqdm
import logging
from datetime import datetime

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataset(Dataset):
    """Custom dataset for Reddit text classification"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class BERTMultiLabelClassifier(nn.Module):
    """BERT-based multi-label classifier"""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super(BERTMultiLabelClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class RedditBERTClassifier:
    """Main BERT classifier for Reddit visa discourse analysis"""
    
    def __init__(self, model_name: str = None, num_labels: int = None):
        self.model_name = model_name or config.model_name
        self.num_labels = num_labels or len(config.labels)
        self.labels = config.labels
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Initialize model
        self.model = BERTMultiLabelClassifier(self.model_name, self.num_labels)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Initialized BERT classifier with {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Filter annotated data
        annotated_df = df[df['annotated'] == True].copy()
        
        if len(annotated_df) == 0:
            raise ValueError("No annotated data found. Please run annotation first.")
        
        # Prepare texts and labels
        texts = annotated_df['processed_text'].tolist()
        labels = []
        
        for labels_str in annotated_df['labels']:
            if isinstance(labels_str, str):
                label_list = labels_str.split(',') if ',' in labels_str else [labels_str]
            else:
                label_list = [labels_str]
            
            # Convert to binary labels
            binary_labels = [1 if label in label_list else 0 for label in self.labels]
            labels.append(binary_labels)
        
        labels = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels.argmax(axis=1)
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train.argmax(axis=1)
        )
        
        # Create datasets
        train_dataset = RedditDataset(X_train, y_train, self.tokenizer, config.max_length)
        val_dataset = RedditDataset(X_val, y_val, self.tokenizer, config.max_length)
        test_dataset = RedditDataset(X_test, y_test, self.tokenizer, config.max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )
        
        logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train the BERT model"""
        logger.info("Starting model training...")
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            
            # Training phase
            train_loss = 0.0
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_f1 = self.evaluate_model(val_loader)
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_f1'].append(val_f1)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= 3:  # Early stopping patience
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed")
        return training_history
    
    def evaluate_model(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate F1 score
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Convert to binary predictions
        binary_predictions = (all_predictions > 0.5).astype(int)
        
        # Calculate F1 score for each label
        f1_scores = []
        for i in range(self.num_labels):
            if all_labels[:, i].sum() > 0:  # Only if label exists in data
                f1 = precision_recall_fscore_support(
                    all_labels[:, i], binary_predictions[:, i], average='binary'
                )[2]
                f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return avg_loss, avg_f1
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []
        
        # Process in batches
        batch_size = config.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                batch_predictions = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(batch_predictions)
        
        # Convert to binary predictions
        binary_predictions = (np.array(predictions) > threshold).astype(int)
        
        return binary_predictions
    
    def get_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate detailed evaluation metrics"""
        metrics = {}
        
        # Overall metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics['overall'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        # Per-label metrics
        per_label_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_label'] = {}
        for i, label in enumerate(self.labels):
            metrics['per_label'][label] = {
                'precision': per_label_metrics[0][i],
                'recall': per_label_metrics[1][i],
                'f1_score': per_label_metrics[2][i],
                'support': per_label_metrics[3][i]
            }
        
        # Confusion matrices for each label
        metrics['confusion_matrices'] = {}
        for i, label in enumerate(self.labels):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            metrics['confusion_matrices'][label] = cm.tolist()
        
        return metrics
    
    def save_model(self, model_path: str = None):
        """Save the trained model"""
        if model_path is None:
            model_path = config.get_model_path()
        
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), model_path / 'model.pt')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save config
        config_dict = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'labels': self.labels,
            'max_length': config.max_length
        }
        
        with open(model_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = None):
        """Load a trained model"""
        if model_path is None:
            model_path = config.get_model_path()
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load config
        with open(model_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        # Update model parameters
        self.model_name = config_dict['model_name']
        self.num_labels = config_dict['num_labels']
        self.labels = config_dict['labels']
        
        # Reinitialize model
        self.model = BERTMultiLabelClassifier(self.model_name, self.num_labels)
        
        # Load model state
        self.model.load_state_dict(torch.load(model_path / 'model.pt', map_location=self.device))
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def run_full_training(self, df: pd.DataFrame) -> Dict:
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(df)
        
        # Train model
        training_history = self.train_model(train_loader, val_loader)
        
        # Evaluate on test set
        test_loss, test_f1 = self.evaluate_model(test_loader)
        
        # Get detailed metrics
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs).cpu().numpy()
                binary_predictions = (predictions > 0.5).astype(int)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(binary_predictions)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        detailed_metrics = self.get_detailed_metrics(y_true, y_pred)
        
        # Save results
        results = {
            'training_history': training_history,
            'test_metrics': {
                'test_loss': test_loss,
                'test_f1': test_f1
            },
            'detailed_metrics': detailed_metrics,
            'model_config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'labels': self.labels
            }
        }
        
        # Save metrics
        metrics_file = config.get_metrics_path() / "training_results.json"
        config.get_metrics_path().mkdir(parents=True, exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {metrics_file}")
        
        return results

def main():
    """Main function for BERT classifier training"""
    import pandas as pd
    
    # Load annotated data
    annotation_file = config.annotation_dir / "exported_annotations.parquet"
    if not annotation_file.exists():
        print("No annotated data found. Please run annotation_system.py first.")
        return
    
    df = pd.read_parquet(annotation_file)
    print(f"Loaded {len(df)} annotated records")
    
    # Initialize classifier
    classifier = RedditBERTClassifier()
    
    # Run training
    results = classifier.run_full_training(df)
    
    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Model: {classifier.model_name}")
    print(f"Labels: {classifier.labels}")
    print(f"Test F1 Score: {results['test_metrics']['test_f1']:.4f}")
    
    # Print per-label metrics
    print("\nPer-label metrics:")
    for label, metrics in results['detailed_metrics']['per_label'].items():
        print(f"  {label}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1_score']:.4f}")
    
    return results

if __name__ == "__main__":
    main()
