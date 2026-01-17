# ================================
# Multi-Modal Fake News Detection
# Binary Classification (Fake vs Real)
# Framework: PyTorch + BERT
# ================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Device Configuration
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# Load Dataset
# ------------------------
df = pd.read_csv("fake_news_dataset.csv").sample(3000, random_state=42)

# Map labels
df["label"] = df["label"].map({"fake": 0, "real": 1})

# ------------------------
# Train-Test Split
# ------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# ------------------------
# Tokenizer
# ------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ------------------------
# Dataset Class
# ------------------------
class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.metadata = dataframe[["followers", "retweets", "likes", "account_age"]].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ------------------------
# DataLoaders
# ------------------------
train_dataset = FakeNewsDataset(train_df, tokenizer)
test_dataset = FakeNewsDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ------------------------
# Multi-Modal Model
# ------------------------
class MultiModalFakeNewsModel(nn.Module):
    def __init__(self):
        super(MultiModalFakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.meta_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, metadata):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output
        meta_features = self.meta_fc(metadata)
        combined = torch.cat((text_features, meta_features), dim=1)
        output = self.classifier(combined)
        return output

# ------------------------
# Initialize Model
# ------------------------
model = MultiModalFakeNewsModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ------------------------
# Training
# ------------------------
def train_model(model, dataloader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ------------------------
# Evaluation
# ------------------------
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, metadata)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Fake", "Real"]))

    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    roc = roc_auc_score(true_labels, predictions)
    print("ROC-AUC:", roc)

# ------------------------
# Run
# ------------------------
train_model(model, train_loader, optimizer, criterion, epochs=3)
evaluate_model(model, test_loader)
