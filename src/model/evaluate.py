import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ModelEvaluator:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        # Auto-load on init
        self.load_model()

    def load_model(self):
        """Load Hugging Face model + tokenizer from path"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, validation_df):
        """Run evaluation on validation dataframe with columns ['text', 'label']"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not loaded. Please load the model before evaluation.")

        texts = validation_df["text"].tolist()
        labels = validation_df["label"].tolist()

        # Tokenize
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # Metrics
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

        return {
            "eval_accuracy": acc,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
        }
