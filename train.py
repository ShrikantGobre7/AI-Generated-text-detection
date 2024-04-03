import pandas as pd
import numpy as np
import re
import tkinter as tk
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import nltk
#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################

# ... (rest of your code)

def train_and_save_bert_model(model, train_data, num_epochs=5, learning_rate=1e-4):
    def Train():
        
        result = pd.read_csv(r"E:/project 2023/AI Generated text/Training_Essay_Data.csv")

        result.head()
            
        result['text_without_stopwords'] = result['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for text, label in train_data:
            input_ids, attention_mask = preprocess_for_bert(text)
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()

            optimizer.zero_grad()
            output = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))[0]
            loss = criterion(output, torch.tensor([label]))  # Assuming label is a scalar
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')

    # Save the trained BERT model
    torch.save(model.state_dict(), 'bert_model.pth')

# Assuming 'train_data' is a list of tuples containing text and labels
# Example: train_data = [('text1', 0), ('text2', 1), ...]

# Define your BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)  # Adjust 'num_labels' based on your classification task

# Tokenization function
def preprocess_for_bert(text):
    tokens = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    return tokens['input_ids'], tokens['attention_mask']

# Sample train_data
train_data = [("This is a positive sentence.", 1), ("This is a negative sentence.", 0)]

# Use the defined BERT architecture
train_and_save_bert_model(bert_model, train_data)