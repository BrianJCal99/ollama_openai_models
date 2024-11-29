import gradio as gr
from openai import OpenAI
import argparse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

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
# Parse the arguments
args = parser.parse_args()

print(NEON_GREEN + "Connecting to MongoDB..." + RESET_COLOR)

uri = "mongodb://127.0.0.1:27017"

# Create a new client and connect to the server
db_client = MongoClient(uri, server_api=ServerApi('1'))

db_name = "conversation_history"
db_collection_name = "test"

OPENAI_API_KEY = "llama3"
OPENAI_BASE_URL = "http://127.0.0.1:11434/v1"

# Send a ping to confirm a successful connection
try:
    db_client.admin.command('ping')
    print(NEON_GREEN + "Pinged your deployment. You successfully connected to MongoDB!" + RESET_COLOR)
    db = db_client[db_name]  # Replace with your database name
    print(NEON_GREEN + "Database: " + CYAN + db.name + RESET_COLOR)
    db_collection = db[db_collection_name] # Define the collection
    print(NEON_GREEN + "Database Collection: " +  CYAN + db_collection.name + RESET_COLOR)
except Exception as e:
    print(e)

system_message = "You are a helpful assistant."

# Configuration for the Ollama API client
ai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

# Function to interact with the Ollama model
def chat_completion(user_input, system_message, ollama_model, conversation_history):
        
        db_collection.insert_one({"role": "user", "content": user_input})
        conversation_history = list(db_collection.find({}, {"_id": 0, "role": 1, "content": 1})) 

        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]

        response = ai_client.chat.completions.create(
            model=ollama_model,
            messages=messages
        )

        db_collection.insert_one({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content

def chat_response(input, history):
        response = chat_completion(input, system_message, args.model, db_collection)
        return response

# Gradio chatbot interface to handle the conversation
local_rag = gr.ChatInterface(
    chat_response, 
    type="messages",
    theme="soft",
    show_progress='hidden',
    fill_height=True, 
    fill_width=True,
    title=args.model
    )

local_rag.chatbot.show_copy_all_button=True
local_rag.chatbot.show_copy_button=True
local_rag.chatbot.show_label=False
local_rag.chatbot.bubble_full_width=False
local_rag.textbox.placeholder="Type your message here..."
local_rag.chatbot.placeholder="<center><h1>What can I help with?</h1></center>"

local_rag.launch()