import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import fitz
from joblib import load
import pickle

root = tk.Tk()
root.title("PDF to Text Extractor")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

img = ImageTk.PhotoImage(Image.open("s1.jpg"))
img2 = ImageTk.PhotoImage(Image.open("s2.jpg"))
img3 = ImageTk.PhotoImage(Image.open("s3.png"))

logo_label = tk.Label(root)
logo_label.place(x=0, y=0)

background_label = tk.Label(root, text="Background Label")  # You are missing this label

x = 1

def move():
    global x
    if x == 4:
        x = 1
    if x == 1:
        logo_label.config(image=img)
    elif x == 2:
        logo_label.config(image=img2)
    elif x == 3:
        logo_label.config(image=img3)
    x = x + 1
    root.after(2000, move)

move()
background_label.place(x=0, y=0)

lbl = tk.Label(root, text="Detect AI Generated Text ", font=('times', 35, 'bold'), height=1, width=65, bg="light blue",
               fg="black")
lbl.place(x=0, y=0)

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()

    pdf_document.close()
    return text

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        text = extract_text_from_pdf(file_path)
        text_output.delete(6.0, tk.END)
        text_output.insert(tk.END, text)
        
        # label4 = tk.Label(root, text="Human Generated Text", width=40, height=2, bg='#46C646', fg='black',
        #                   font=("Tempus Sanc ITC", 25))
        # label4.place(x=450, y=550)
        
        predictor = load("model.joblib")
        Given_text = text
        vec = open('vectorizer.pickle', 'rb')
        tf_vect = pickle.load(vec)
        X_test_tf = tf_vect.transform([Given_text])
        y_predict = predictor.predict(X_test_tf)
        
        if y_predict[0] == 0:
            label4 = tk.Label(root, text="Human Generated Text", width=40, height=2, bg='#46C646', fg='black',
                              font=("Tempus Sanc ITC", 25))
            label4.place(x=450, y=550)
        else:
            label4 = tk.Label(root, text="AI Generated Text", width=40, height=2, bg='#FF3C3C', fg='black',
                              font=("Tempus Sanc ITC", 25))
            label4.place(x=450, y=550)

# Open File button
label4 = tk.Button(root, text="Open PDF",height=1,bg='light blue', fg='black',font=("Tempus Sanc ITC", 25), command=open_file_dialog)
label4.place(x=700, y=450)

# Text output area
text_output = tk.Text(root, wrap=tk.WORD, width=60, height=20)
text_output.place(x=600, y=100)

# Start the Tkinter main loop
root.mainloop()
