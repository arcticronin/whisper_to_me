import requests
import json


def call_deepseek_api(messages, output_filename="response.txt"):
    """
    Sends 'messages' (list of {role, content}) to the DeepSeek FastAPI endpoint
    running locally, and saves the response to 'output_filename'.
    """
    url = "http://localhost:8000/ask"
    payload = {"messages": messages}

    # Make a POST request to the /ask endpoint
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print("Response text:", response.text)
        return

    # Parse the JSON response
    data = response.json()
    answer = data.get("answer", "No answer found.")

    # Print the answer to the console (optional)
    print("Model Answer:", answer)

    # Save answer to file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"Answer saved to '{output_filename}'")


if __name__ == "__main__":
    # Example usage: ask a question
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    call_deepseek_api(messages, output_filename="deepseek_response.txt")
