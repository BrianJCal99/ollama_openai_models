import gradio as gr
import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
from pathlib import Path
import json

# Function to convert PDF to text and append to vault.txt
def upload_pdf(filepath):
    file_path = filepath.name
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            if page.extract_text():
                text += page.extract_text() + " "
        
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Check if the current sentence plus the current chunk exceeds the limit
            if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                # When the chunk exceeds 1000 characters, store it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:  # Don't forget the last chunk!
            chunks.append(current_chunk)
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

# Function to upload a text file and append to vault.txt
def upload_txt(filepath):
    file_path = filepath.name
    with open(file_path, 'r', encoding="utf-8") as txt_file:
        text = txt_file.read()
        
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Check if the current sentence plus the current chunk exceeds the limit
            if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                # When the chunk exceeds 1000 characters, store it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:  # Don't forget the last chunk!
            chunks.append(current_chunk)
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

# Function to upload a JSON file and append to vault.txt
def upload_json(filepath):
    file_path = filepath.name
    with open(file_path, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)
        
        # Flatten the JSON data into a single string
        text = json.dumps(data, ensure_ascii=False)
        
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Check if the current sentence plus the current chunk exceeds the limit
            if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                # When the chunk exceeds 1000 characters, store it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:  # Don't forget the last chunk!
            chunks.append(current_chunk)
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

def upload_file(filepath):
    return filepath

def convert_to(filepath):
    file_type = Path(filepath).suffix
    file_name = Path(filepath).name
    list_of_supported_file_types = [".pdf", ".txt", ".json"]

    if file_type in list_of_supported_file_types:
        if file_type == ".pdf":
            upload_pdf(filepath)
        if file_type == ".txt":
            upload_txt(filepath)
        if file_type == ".json":
            upload_json(filepath)
        return "✔️ Content from '" + file_name + "' has been successfully appended to vault.txt, with each chunk on a separate line!"
    else:
        return "❌ Invalid file type. Only pdf, txt and json file types are supported at the moment. Please upload again."

with gr.Blocks(title="Upload Documents", theme="soft") as demo:
    file_output = gr.File(show_label=False)
    convert_button = gr.Button("Convert")
    conversion_status = gr.Textbox(show_label=False, container=False, placeholder="Conversion status")
    convert_button.click(convert_to, inputs=file_output, outputs=conversion_status)

demo.launch()