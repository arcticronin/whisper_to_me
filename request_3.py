from huggingface_hub import InferenceClient
import yaml


# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DS_API_TOKEN = config["api_keys"]["huggingface"]

client = InferenceClient(
    # provider="hyperbolic",
    provider="hf-inference",
    api_key=DS_API_TOKEN,
)

messages = [{"role": "user", "content": "What is the capital of France?"}]

completion = client.chat.completions.create(
    # model="deepseek-ai/DeepSeek-R1",
    # model="meta-llama/Meta-Llama-3-70B-Instruct",
    # model="meta-llama/Llama-2-7b-chat-hf",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    messages=messages,
    max_tokens=500,
)

print(completion.choices[0].message.content)
