
import pandas as pd
import numpy as np
import re
import tkinter as tk
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from tkinter import ttk
from PIL import Image, ImageTk
import nltk
from tkinter import filedialog
import fitz
#######################################################################################################
# Download NLTK resources
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################

# Create a Tkinter window
root = tk.Tk()
root.title("Detect AI Generated Text")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

bg = Image.open(r"image3.png")
bg.resize((1200,300),Image.ANTIALIAS)
print(w,h)
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=93)
#, relwidth=1, relheight=1)

w = tk.Label(root, text="AI generated text detector ",width=40,background="#212F3D",foreground="white",height=2,font=("Times new roman",19,"bold"))
w.place(x=0,y=15)
###########################################################################################################
# Define BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)
# Load the trained weights
bert_model.load_state_dict(torch.load('bert_model.pth'))
bert_model.eval()


# Label for the GUI
lbl = tk.Label(root, text="Detect AI Generated Text ", font=('times', 35, 'bold'), height=1, width=65, bg="#FFBF40",
               fg="black")
lbl.place(x=0, y=10)


def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf_document:
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()
    return text

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        text = extract_text_from_pdf(file_path)
        # text_output.delete(1.0, tk.END)
        # text_output.insert(tk.END, text)
        
        # Tokenize and preprocess the text for BERT model
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        if predicted_class == 0:
            result_label.config(text="Human Generated Text", bg='#46C646', fg='black')
        else:
            result_label.config(text="AI Generated Text", bg='#FF3C3C', fg='black')

# GUI Setup
frame = tk.LabelFrame(root, text="Control Panel", width=250, height=450, bd=3, background="black", foreground="white",
                      font=("Tempus Sanc ITC", 15, "bold"))
frame.place(x=15, y=100)

button3 = tk.Button(frame, command=open_file_dialog, text="Test", bg="#E46EE4", fg="white", width=15,
                    font=("Times New Roman", 15, "bold"))
button3.place(x=25, y=250)

result_label = tk.Label(root, text="", font=('times', 20), height=2, width=30, bg='khaki', fg='black')
result_label.place(x=450, y=500)

root.mainloop()