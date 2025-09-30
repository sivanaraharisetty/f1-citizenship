import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class BertClassifierTrainer:
    def __init__(self, num_labels=2, resume_checkpoint: Optional[str] = None, learning_rate: float = 2e-5):
        # Tokenizer & Model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        base = resume_checkpoint or "bert-base-uncased"
        self.model = BertForSequenceClassification.from_pretrained(
            base, num_labels=num_labels
        )

        # Device detection
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon
        else:
            self.device = "cpu"

        self.model.to(self.device)
        print(f" Training will run on: {self.device.upper()}")
        self.learning_rate = learning_rate

    def preprocess(self, texts, labels):
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def compute_metrics(self, preds, labels):
        # Ensure preds and labels are numpy arrays
        if hasattr(preds, 'argmax'):
            preds = preds.argmax(-1)
        else:
            preds = np.array(preds)
        
        labels = np.array(labels)
        
        # Handle single value case
        if preds.ndim == 0:
            preds = np.array([preds])
        if labels.ndim == 0:
            labels = np.array([labels])
            
        # Ensure both arrays have the same length
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]
        
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def train_on_dataframe(self, df: pd.DataFrame, epochs: int = 1, save_dir: str = "results", resume_from: Optional[str] = None):
        # Ensure results dir
        os.makedirs(save_dir, exist_ok=True)

        # Expect labels already integer-coded consistently across chunks
        if df["label"].dtype.kind not in {"i", "u"}:
            raise ValueError("Expected integer-coded labels in 'label' column for chunked training.")

        # Train/validation split (no stratification for small datasets)
        X_train, X_val, y_train, y_val = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )
        X_train, X_val = X_train.reset_index(drop=True), X_val.reset_index(drop=True)
        y_train, y_val = y_train.reset_index(drop=True), y_val.reset_index(drop=True)

        # Prepare datasets
        train_dataset = self.preprocess(X_train, y_train)
        val_dataset = self.preprocess(X_val, y_val)

        # Batch sizes - maximum speed
        train_bs, eval_bs = (32, 32) if self.device != "cpu" else (16, 16)

        # Training arguments - with monitoring and checkpointing
        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            learning_rate=self.learning_rate,  # Use full learning rate for better learning
            logging_dir=os.path.join(save_dir, "logs"),
            logging_steps=5,
            disable_tqdm=True,  # Disable progress bar for speed
            no_cuda=(self.device == "cpu"),
            save_strategy="steps",  # Enable checkpoint saving
            save_steps=1,  # Save after each step
            eval_strategy="steps",  # Enable evaluation
            eval_steps=1,  # Evaluate after each step
            load_best_model_at_end=True,  # Load best model
            fp16=(self.device == "cuda"),
            bf16=(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()),
            max_steps=50,  # More steps for better learning
            save_total_limit=3,  # Keep last 3 checkpoints
            dataloader_num_workers=0,  # No multiprocessing
            remove_unused_columns=True,
            gradient_accumulation_steps=1,
            warmup_steps=0,  # No warmup
            weight_decay=0.0,  # No weight decay
        )

        # Trainer
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(-1)
            return self.compute_metrics(preds, labels)

        # Only add EarlyStoppingCallback if evaluation is enabled
        callbacks = []
        if training_args.eval_strategy != "no":
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
        )

        # Train (optionally resume)
        trainer.train(resume_from_checkpoint=resume_from)

        # Evaluate and return metrics
        eval_output = trainer.predict(val_dataset)
        metrics = self.compute_metrics(eval_output.predictions, eval_output.label_ids)
        self.trainer = trainer
        return metrics
