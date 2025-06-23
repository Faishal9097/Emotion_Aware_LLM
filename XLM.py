import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, Value, Features
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
GO_EMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]
NUM_LABELS = len(GO_EMOTIONS_LABELS)


# =====================
# DATA PREPROCESSING
# =====================

def load_and_preprocess_data():
    """Load and preprocess all datasets with label alignment"""
    all_dfs = []

    # 1. GoEmotions Dataset (28 labels)
    print("Loading GoEmotions dataset...")
    go_emotions = load_dataset("go_emotions")

    # Process GoEmotions
    for split in ['train', 'validation', 'test']:
        df = go_emotions[split].to_pandas()

        # Create binary columns for each emotion
        for i, emotion in enumerate(GO_EMOTIONS_LABELS):
            df[emotion] = df['labels'].apply(lambda labels: 1 if i in labels else 0)

        all_dfs.append(df[['text'] + GO_EMOTIONS_LABELS])

    # 2. Hinglish Dataset (local CSV)
    print("Loading Hinglish dataset...")
    hinglish_df = pd.read_csv("./hinglish_dataset.csv")

    # Map to GoEmotions labels
    hinglish_mapping = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'joy',
        'other': 'neutral',
        'sadness': 'sadness',
        'surprise': 'surprise'
    }
    hinglish_df['mapped_label'] = hinglish_df['labels'].map(hinglish_mapping)

    # Create binary columns
    for emotion in GO_EMOTIONS_LABELS:
        hinglish_df[emotion] = (hinglish_df['mapped_label'] == emotion).astype(int)

    all_dfs.append(hinglish_df[['text'] + GO_EMOTIONS_LABELS])

    # 3. Hindi (BHAAV) Dataset - Handle severe imbalance
    print("Loading Hindi dataset...")
    hindi_df = pd.read_excel("./Bhaav-Dataset.xlsx", engine='openpyxl')  # Update path

    # Map to GoEmotions labels
    hindi_mapping = {
        'anger': 'anger',
        'joy': 'joy',
        'sad': 'sadness',
        'suspense': 'surprise',  # Assuming typo for 'suspense/surprise'
        'neutral': 'neutral'
    }
    hindi_df['mapped_label'] = hindi_df['label'].map(hindi_mapping)

    # Handle severe imbalance in Hindi dataset
    print(f"Original Hindi class distribution:\n{hindi_df['mapped_label'].value_counts()}")

    # Undersample neutral class
    neutral_samples = hindi_df[hindi_df['mapped_label'] == 'neutral']
    non_neutral_samples = hindi_df[hindi_df['mapped_label'] != 'neutral']

    # Keep all non-neutral samples, undersample neutral to match
    undersampled_neutral = neutral_samples.sample(
        n=len(non_neutral_samples),
        random_state=42
    )
    hindi_df = pd.concat([non_neutral_samples, undersampled_neutral])

    print(f"Balanced Hindi class distribution:\n{hindi_df['mapped_label'].value_counts()}")

    # Create binary columns
    for emotion in GO_EMOTIONS_LABELS:
        hindi_df[emotion] = (hindi_df['mapped_label'] == emotion).astype(int)

    all_dfs.append(hindi_df[['text'] + GO_EMOTIONS_LABELS])

    # Combine all datasets
    print("Combining datasets...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Train-validation split
    print("Performing train-validation split...")
    train_df, val_df = train_test_split(
        combined_df,
        test_size=0.1,
        random_state=42
    )

    # Reset index to avoid pandas index column
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Convert to Dataset objects
    features = Features({
        'text': Value('string'),
        **{label: Value('int64') for label in GO_EMOTIONS_LABELS}
    })

    train_ds = Dataset.from_pandas(train_df, features=features)
    val_ds = Dataset.from_pandas(val_df, features=features)

    return DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })


# =====================
# ADVANCED LOSS FUNCTIONS
# =====================

class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted BCE Loss with class weights"""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(inputs.device)
        return F.binary_cross_entropy_with_logits(
            inputs,
            targets.float(),
            pos_weight=self.pos_weight
        )


# =====================
# MODEL & TRAINING SETUP
# =====================

def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset using PyTorch"""
    label_counts = torch.zeros(NUM_LABELS)
    total_samples = len(dataset)

    # Sum counts for each label using PyTorch
    for i, label in enumerate(GO_EMOTIONS_LABELS):
        # Access the entire label column as a PyTorch tensor
        label_counts[i] = torch.sum(dataset[label])

    # Avoid division by zero
    label_counts = torch.clamp(label_counts, min=1)

    # Weight inversely proportional to class frequency
    weights = total_samples / (NUM_LABELS * label_counts)
    return weights.float()


# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# =====================
# METRICS
# =====================

def compute_metrics(p: EvalPrediction):
    """Compute accuracy, precision, recall, and F1 for multi-label classification"""
    preds = (torch.sigmoid(torch.tensor(p.predictions)) > 0.5).int().numpy()
    labels = p.label_ids

    # Calculate metrics per class
    results = {}
    for i, label_name in enumerate(GO_EMOTIONS_LABELS):
        tp = np.sum((preds[:, i] == 1) & (labels[:, i] == 1))
        fp = np.sum((preds[:, i] == 1) & (labels[:, i] == 0))
        fn = np.sum((preds[:, i] == 0) & (labels[:, i] == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        results[f"{label_name}_precision"] = precision
        results[f"{label_name}_recall"] = recall
        results[f"{label_name}_f1"] = f1

    # Calculate macro averages
    avg_precision = np.mean([results[f"{n}_precision"] for n in GO_EMOTIONS_LABELS])
    avg_recall = np.mean([results[f"{n}_recall"] for n in GO_EMOTIONS_LABELS])
    avg_f1 = np.mean([results[f"{n}_f1"] for n in GO_EMOTIONS_LABELS])

    results.update({
        "macro_precision": avg_precision,
        "macro_recall": avg_recall,
        "macro_f1": avg_f1
    })

    return results


# =====================
# CUSTOM TRAINER
# =====================

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels and create label tensor
        labels = torch.stack([inputs[label] for label in GO_EMOTIONS_LABELS], dim=1)

        # Create filtered inputs with only model-expected keys
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        outputs = model(**model_inputs)
        logits = outputs.logits

        # Compute loss
        loss_fct = WeightedBCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to handle custom inputs properly"""
        # Extract labels for metrics computation
        if all(label in inputs for label in GO_EMOTIONS_LABELS):
            labels = torch.stack([inputs[label] for label in GO_EMOTIONS_LABELS], dim=1)
        else:
            labels = None

        # Create filtered inputs with only model-expected keys
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        # Move inputs to the same device as the model
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        if labels is not None:
            labels = labels.to(model.device)

        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits

            # Compute loss if labels are available
            loss = None
            if labels is not None and not prediction_loss_only:
                loss_fct = WeightedBCEWithLogitsLoss(pos_weight=self.class_weights)
                loss = loss_fct(logits, labels.float())

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits.detach().cpu(), labels.detach().cpu() if labels is not None else None)


# =====================
# MAIN TRAINING SCRIPT
# =====================

def train():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataset = load_and_preprocess_data()

    # Tokenize datasets
    print("Tokenizing data...")
    tokenized_ds = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']  # Only remove text column, keep all emotion labels
    )

    # Format for PyTorch
    tokenized_ds.set_format("torch", columns=[
        "input_ids",
        "attention_mask",
        *GO_EMOTIONS_LABELS
    ])

    # Compute class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(tokenized_ds["train"])
    print("Class weights:")
    for label, weight in zip(GO_EMOTIONS_LABELS, class_weights):
        print(f"{label}: {weight:.2f}")

    # Initialize model
    print("Initializing model...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        gradient_accumulation_steps=8,
        fp16=True,
        remove_unused_columns=False  # Crucial for custom datasets with labels
    )

    # Initialize Custom Trainer with class weights
    print("Starting training...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    # Start training
    trainer.train()

    # Save final model
    print("Saving model...")
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Training complete!")


if __name__ == "__main__":
    train()