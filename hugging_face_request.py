from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import uvicorn
import yaml
from transformers import pipeline

# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

HF_API_TOKEN = config["api_keys"]["huggingface"]


# Example endpoint for a Llama model on Hugging Face Inference API.
# Replace with the actual URL for the model you have access to!
# e.g., "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
# or your own private endpoint if you're hosting a Space or a custom Inference Endpoint.
HF_API_URL = "https://api-inference.huggingface.co/models/REPLACE_ME_WITH_LLAMA_MODEL"

# ------------------------------
# FastAPI application
# ------------------------------
app = FastAPI()


# ------------------------------
# Pydantic model for request body
# ------------------------------
class Question(BaseModel):
    question: str


# ------------------------------
# Helper function to call the HF Inference API
# ------------------------------
def query_huggingface_api(prompt: str):
    """
    Calls the Hugging Face Inference API for a given prompt.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    # You can adjust parameters depending on model.
    # e.g., for conversation models that expect {"inputs": {"text": "..."}}
    # or for text-generation models that expect {"inputs": "..."}
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
        raise ValueError(
            f"Hugging Face API error {response.status_code}: {response.text}"
        )

    # The structure of the output depends on the model and pipeline used.
    # For most text-generation endpoints, the response is a list of dicts:
    # [{"generated_text": "..."}]
    result = response.json()
    return result


# ------------------------------
# API endpoint: POST /ask
# ------------------------------
@app.post("/ask")
def ask_question(payload: Question):
    """
    Accepts a JSON body like:
        {
            "question": "Your question here..."
        }
    Returns a JSON object with the generated answer.
    """
    question_text = payload.question

    # Optionally build a prompt if needed by the model
    # (for example, if using a Chat model you might want a
    # system/user instruction format). In simplest form:
    prompt = f"Question: {question_text}\nAnswer:"

    # Query the Hugging Face Inference API
    output = query_huggingface_api(prompt)

    # You may need to adjust how you extract the answer
    # depending on the model's return format.
    try:
        # For text-generation models:
        generated_text = output[0]["generated_text"]
    except (IndexError, KeyError):
        generated_text = "No answer could be generated."

    # Return JSON response
    return {"answer": generated_text}


# ------------------------------
# Run the server
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
