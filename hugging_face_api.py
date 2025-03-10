from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import json
import yaml
from transformers import pipeline

# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

HF_API_TOKEN = config["api_keys"]["huggingface"]
# HF_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
# HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
HF_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-V3"
# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI()


# ------------------------------
# Pydantic Models
# ------------------------------
class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: List[Message]


# ------------------------------
# Helper Function to Call the HF API
# ------------------------------
def query_huggingface_api(prompt: str):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    }

    response = requests.post(HF_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.text}")

    result = response.json()
    return result


# ------------------------------
# Endpoint to Accept Messages
# ------------------------------


@app.post("/ask")
def ask_question(conversation: Conversation):
    # Correct access to object attributes
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation.messages])

    # Call the hosted HF model
    output = query_huggingface_api(prompt)

    # Extract the generated text (adjust parsing if needed)
    try:
        generated_text = output[0]["generated_text"]
    except (IndexError, KeyError):
        generated_text = "No valid response."

    return {"answer": generated_text}


# ------------------------------
# Start the FastAPI Server
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
