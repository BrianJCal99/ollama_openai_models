import gradio as gr
import ollama
import time
import argparse
from openai import OpenAI

conversation_history = []
system_message = "You are a helpful assistant."

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

def ollama_chat(user_input, system_message, ollama_model, conversation_history):
        
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

def respond(user_input, chat_history):
        response = ollama_chat(user_input, system_message, args.model, conversation_history)
        for i in range(len(response)):
            time.sleep(0.01)
            yield response[: i+1]

chatbot = gr.ChatInterface(
      respond, 
        type="messages",
        theme="soft",
        show_progress='hidden',
        fill_height=True, 
        fill_width=True,
        examples=["I need help with a querry.", "Can you assist me with a task?"],
        title=args.model
        )

chatbot.chatbot.show_copy_all_button=True
chatbot.chatbot.show_copy_button=True
chatbot.chatbot.show_label=False
chatbot.chatbot.bubble_full_width=False
chatbot.textbox.placeholder="Type your message here..."
chatbot.chatbot.placeholder="<center><strong>Sart chatting with your AI assistant</strong><br>" + args.model + "<br>🦙🤖🦙</center>"

chatbot.launch()
