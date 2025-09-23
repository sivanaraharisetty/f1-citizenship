import json
import joblib
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Global variables for model and tokenizer
model = None
tokenizer = None
label_mapping = None

def model_fn(model_dir):
    """
    Load the model and tokenizer from SageMaker model directory
    """
    global model, tokenizer, label_mapping
    
    # Load label mapping
    with open(f'{model_dir}/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(label_mapping)
    )
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def input_fn(request_body, request_content_type):
    """
    Parse input data from request body
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data, model):
    """
    Make predictions on input data
    """
    global tokenizer, label_mapping
    
    predictions = []
    
    for instance in input_data.get('instances', []):
        text = instance.get('text', '')
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Get category name
        category = list(label_mapping.keys())[predicted_class_id]
        
        predictions.append({
            'category': category,
            'confidence': float(confidence),
            'class_id': predicted_class_id
        })
    
    return {'predictions': predictions}

def output_fn(prediction, content_type):
    """
    Format output for response
    """
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f'Unsupported content type: {content_type}')
