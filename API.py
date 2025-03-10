from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline
import uvicorn


class Message(BaseModel):
    role: str  # e.g., "user", "system", "assistant"
    content: str


class Conversation(BaseModel):
    messages: List[Message]


app = FastAPI()

# Load the pipeline with DeepSeek-R1
deepseek_pipe = pipeline(
    "text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True
)


@app.post("/ask")
def ask_question(conversation: Conversation):
    """
    Accepts JSON with a list of messages, for example:
    {
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ]
    }
    Returns a JSON object with the generated answer.
    """
    # In principle, you can pass messages directly to the pipeline
    messages_input = conversation.messages
    output = deepseek_pipe(messages_input)

    # Extract generated text from the pipeline output
    try:
        generated_text = output[0]["generated_text"]
    except (IndexError, KeyError, TypeError):
        generated_text = "No valid response."

    return {"answer": generated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
