import gradio as gr
import torch
import ollama
import os
from openai import OpenAI
import argparse
import PyPDF2
import re
from pathlib import Path
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
RED = '\033[91m'

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
# Create the argument parser
parser = argparse.ArgumentParser(description="Ollama Chat")
# Add the model argument
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
# Add the vault argument
parser.add_argument("--vault", default="vault.txt", help="Path to the vault file (default: vault.txt)")
# Add the query rewrite argument
parser.add_argument("--mode", default="chat", help="Enable or disable query rewrite (default: False)")
# Parse the arguments
args = parser.parse_args()

conversation_history = []
multimodal = False 

if args.mode == "rag" :
    print(NEON_GREEN + "Retrieval-Augmented Generation (RAG) Mode." + RESET_COLOR)
    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text."
    multimodal = True

if args.mode == "chat":
    print(NEON_GREEN + "Chat Completeion Mode." + RESET_COLOR)
    system_message = "You are a helpful assistant."

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='dolphin-llama3'
)

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Ollama model
def chat_completion(user_input, system_message, ollama_model, conversation_history):
        
        conversation_history.append({"role": "user", "content": user_input})

        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]

        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages
        )

        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content

def chat_rag(user_input, system_message, vault_embeddings_tensor, vault_content, ollama_model, conversation_history):
    print(PINK + "User query received." + RESET_COLOR)
    print(YELLOW + "Looking for relevant text in the vault." + RESET_COLOR)
    #print(PINK + "\nUser Query: " + user_input + RESET_COLOR)
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=3)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print(CYAN + "Relevant context found." + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    print(YELLOW + "Creating AI response."+ RESET_COLOR)

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})
    
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    # Return the content of the response from the model
    print(CYAN + "AI response created."+ RESET_COLOR)
    #print(NEON_GREEN + "AI Response: " + response.choices[0].message.content + RESET_COLOR)
    return response.choices[0].message.content

# Function to convert PDF to text and append to vault
def upload_pdf(filepath):
    with open(filepath, 'rb') as pdf_file:
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
        with open(args.vault, "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

# Function to upload a text file and append to vault
def upload_txt(filepath):
    with open(filepath, 'r', encoding="utf-8") as txt_file:
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
        with open(args.vault, "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

# Function to upload a JSON file and append to vault
def upload_json(filepath):
    with open(filepath, 'r', encoding="utf-8") as json_file:
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
        with open(args.vault, "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

# Load the vault content
def load_vault_content(vault):
    print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
    vault_content = []
    if os.path.exists(vault):
        with open(vault, "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
    #print(PINK + "Vault content..." + RESET_COLOR)
    #print(vault_content)
    return vault_content

# Generate embeddings for the vault content using Ollama
def generate_vault_embeddings(vault_content):
    print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    #print(PINK + "Embeddings for each line in the vault:" + RESET_COLOR)
    #print(vault_embeddings)
    return vault_embeddings

# Convert to tensor
def convert_to_tensor(vault_embeddings):
    print(NEON_GREEN + "Converting embeddings to tensor..." + RESET_COLOR + RESET_COLOR)
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    #print(PINK + "Embeddings for each line in the vault:" + RESET_COLOR)
    #print(vault_embeddings_tensor)
    print(NEON_GREEN + "Memory updated and ready to chat!" + RESET_COLOR)
    return vault_embeddings_tensor

def upload_file(filepath):
    global vault_content
    global vault_embeddings
    global vault_embeddings_tensor
    
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

        vault_content = load_vault_content(args.vault)

        vault_embeddings = generate_vault_embeddings(vault_content)

        print(NEON_GREEN + "Converting embeddings to tensor..." + RESET_COLOR + RESET_COLOR)
        vault_embeddings_tensor = convert_to_tensor(vault_embeddings)

        print(NEON_GREEN + "Content from '" + file_name + "' has been successfully appended to vault, with each chunk on a separate line!" + RESET_COLOR)
        print(NEON_GREEN + "Memory updated and ready to chat!" + RESET_COLOR)

        return "✔️ <strong>'" + file_name + "'</strong> uploaded successfully!"
    else:
        print(RED + file_type + "' file type is not supported. Only pdf, txt and json file types are supported at the moment. Please upload again." + RESET_COLOR)
        return "❌ <strong>'" + file_name + "'</strong> could not upload successfully."

vault_content = load_vault_content(args.vault)

vault_embeddings = generate_vault_embeddings(vault_content)

vault_embeddings_tensor = convert_to_tensor(vault_embeddings)

def rag_response(input, history):
        global vault_content
        global vault_embeddings
        global vault_embeddings_tensor

        file_upload_status = ""
        response = ""
        
        if (input['text']) and not len(input['files']):
            response = chat_rag(input['text'], system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
        elif len(input['files']) and not (input['text']):
            if len(input['files']) > 1:
                for file in input['files']:
                    single_status  = upload_file(file)
                    file_upload_status += single_status + "<br>"
            else:
                file_upload_status = upload_file(input["files"][0])
            response = file_upload_status
        elif (input['text']) and len(input['files']):
            if len(input['files']) > 1:
                for i in input['files']:
                    single_status  = upload_file(i)
                    file_upload_status += single_status + "<br>"
            else:
                file_upload_status = upload_file(input["files"][0])
            response = file_upload_status + "<br><br>" + chat_rag(input['text'], system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
        else:
            print(RED + "Empty input received. Not proceeding forward. Aborting..." + RESET_COLOR)
            response =  "It seems like your message is empty. Could you please resend it or let me know how I can assist you?"

        return response

def chat_response(input, history):
        response = chat_completion(input, system_message, args.model, conversation_history)
        return response

def main(input,history):
    if args.mode == "rag":
        response = rag_response(input,history)
    if args.mode == "chat":
        response = chat_response(input,history)
    return response

# Gradio chatbot interface to handle the conversation
local_rag = gr.ChatInterface(
    main, 
    type="messages",
    theme="soft",
    show_progress='hidden',
    fill_height=True, 
    fill_width=True,
    multimodal=multimodal,
    title=args.model + " | " + args.mode
    )

local_rag.chatbot.show_copy_all_button=True
local_rag.chatbot.show_copy_button=True
local_rag.chatbot.show_label=False
local_rag.chatbot.bubble_full_width=False
local_rag.textbox.placeholder="Type your message here..."
local_rag.chatbot.placeholder="<center><h1>What can I help with?</h1></center>"

local_rag.launch()