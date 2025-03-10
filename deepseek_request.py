from openai import OpenAI
import yaml

# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DS_API_TOKEN = config["api_keys"]["deepseek"]

client = OpenAI(api_key=DS_API_TOKEN, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
)

print(response.choices[0].message.content)
