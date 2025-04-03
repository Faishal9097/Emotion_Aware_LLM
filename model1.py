import transformers
import datasets
import bertviz
import umap
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import pandas as pd  # For handling and preprocessing data
import numpy as np   # For numerical operations
from sympy.utilities.matchpy_connector import ReplacementInfo

print("All libraries are installed successfully!")

from datasets import load_dataset, DownloadConfig

# Configure DownloadConfig with appropriate parameters
# For details, refer to the datasets documentation
# https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.DownloadConfig

download_config = DownloadConfig(
    max_retries=3
)
dataset = load_dataset("go_emotions", download_config=download_config)

def encode_labels(examples):
    """Convert list of label indices to multi-hot vectors"""
    num_classes = 28
    multi_hot = np.zeros((len(examples['labels']), num_classes), dtype=np.float32)
    for idx, labels in enumerate(examples['labels']):
        for label in labels:
            if label < num_classes:
                multi_hot[idx, label] = 1.0
    return {'labels': multi_hot}

dataset = dataset.map(encode_labels, batched=True)

# Bert Model for transformers: https://huggingface.co/google-bert/bert-base-uncased
#Tokenisation using Method 1 using Provided code from BERT Git
from transformers import BertTokenizer, BertModel
#Loading Pretrained Tokeniser and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


#As we converted the dataset to pandas earlier, It will convert it back
dataset.reset_format()

# map() method would be used

def tokenize(batch):
  return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128,return_tensors="pt") #Setting max_length for Padding and Truncation

Dataset_encoded = dataset.map(tokenize, batched=True) # This line is crucial

print("phase 2 done")
# ===== DATASET PREPROCESSING =====
def encode_labels(example):
    """Convert multi-label list to multi-hot vector"""
    multi_hot = np.zeros(28, dtype=np.float32)
    for label in example['labels']:
        if label < 28:  # Handle possible out-of-bounds labels
            multi_hot[label] = 1.0
    return {'labels': multi_hot}

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.map(encode_labels)

import torch
from torch.utils.data import DataLoader, Dataset

class GoEmotionDataset(Dataset):
    def __init__(self, encoded_data):
        self.input_ids = torch.stack([
            torch.tensor(x) for x in encoded_data['input_ids']
        ])
        self.attention_mask = torch.stack([
            torch.tensor(x) for x in encoded_data['attention_mask']
        ])
        self.labels = torch.tensor(
            encoded_data['labels'],
            dtype=torch.float32
        )
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Load Dataset
train_dataset =GoEmotionDataset(Dataset_encoded['train'])
val_dataset = GoEmotionDataset(Dataset_encoded['validation'])

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")  # Debug line

# Load model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28,  problem_type="multi_label_classification"  ).to(device)

# Training arguments (with GPU optimizations)
training_args = TrainingArguments(
    output_dir="./emotion_detection",
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",  # Added to match eval strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    metric_for_best_model='eval_micro_f1',
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none"  # Disables WandB logging
)

from sklearn.metrics import f1_score, accuracy_score

# ===== METRICS =====
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5)

    preds_np = preds.cpu().numpy().astype(int)
    labels_np = labels.astype(int)

    return {
        'micro_f1': f1_score(labels_np, preds_np, average='micro', zero_division=0),
        'macro_f1': f1_score(labels_np, preds_np, average='macro', zero_division=0),
        'accuracy': accuracy_score(labels_np, preds_np)
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the Model
trainer.train()


# Save the Model
model.save_pretrained("./emotion_detection_model")

# Example Prediction
model.eval()

# Save model
model.save_pretrained("./emotion_detection_model")
tokenizer.save_pretrained("./emotion_detection_model")
print("Phase 3 done")

# #Important but cant work in my pc
# #Printing Classification Report
# # Prepare the input tensors
# test_input = torch.tensor(Dataset_encoded['test']['input_ids']).to(device)
# test_mask = torch.tensor(Dataset_encoded['test']['attention_mask']).to(device)
#
# # Perform inference (UPDATED FOR MULTI-LABEL)
# with torch.no_grad():
#     outputs = model(input_ids=test_input, attention_mask=test_mask)
#     # Apply sigmoid and threshold for multi-label classification
#     probs = torch.sigmoid(outputs.logits)
#     y_pred = (probs > 0.5).int().cpu().numpy()  # Threshold at 0.5
#
# # Convert to evaluation format (UPDATED)
# y_true = np.array(Dataset_encoded['test']['labels'])  # Already multi-hot encoded
#
# # Evaluate the Model
# from sklearn.metrics import classification_report
#
# emotion_labels = [ "admiration", "amusement", "anger", "annoyance", "approval", "caring",
#     "confusion", "curiosity", "desire", "disappointment", "disapproval",
#     "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
#     "joy", "love", "nervousness", "optimism", "pride", "realization",
#     "relief", "remorse", "sadness", "surprise", "neutral"
# ]
# print(classification_report(y_true, y_pred, target_names=emotion_labels,zero_division=0))
# # Example usage
# test_index = 0
# input_content = dataset['test'][test_index]['text']
# detected_emotions = [emotion_labels[i] for i, val in enumerate(y_pred[test_index]) if val == 1]
#
# print(f"Input: {input_content}")
# print(f"Detected Emotions: {detected_emotions}")
#print("Phase 4 Done")

