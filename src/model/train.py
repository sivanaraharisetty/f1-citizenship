from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json

from typing import Optional


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
        print(f"✅ Training will run on: {self.device.upper()}")
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
        preds = preds.argmax(-1)
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

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
        X_train, X_val = X_train.reset_index(drop=True), X_val.reset_index(drop=True)
        y_train, y_val = y_train.reset_index(drop=True), y_val.reset_index(drop=True)

        # Prepare datasets
        train_dataset = self.preprocess(X_train, y_train)
        val_dataset = self.preprocess(X_val, y_val)

        # Batch sizes
        train_bs, eval_bs = (16, 16) if self.device != "cpu" else (8, 8)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            learning_rate=self.learning_rate,
            logging_dir=os.path.join(save_dir, "logs"),
            logging_steps=50,
            disable_tqdm=False,
            no_cuda=(self.device == "cpu"),
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=(self.device == "cuda"),
            bf16=(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()),
        )

        # Trainer
        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(-1)
            return self.compute_metrics(preds, labels)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train (optionally resume)
        trainer.train(resume_from_checkpoint=resume_from)

        # Evaluate and return metrics
        eval_output = trainer.predict(val_dataset)
        metrics = self.compute_metrics(eval_output.predictions, eval_output.label_ids)
        self.trainer = trainer
        return metrics
