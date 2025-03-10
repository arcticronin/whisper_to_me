from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import yaml

# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access keys and settings
OPENAI_API_KEY = config["api_keys"]["openai"]
ELEVENLABS_API_KEY = config["api_keys"]["elevenlabs"]
MODEL = config["settings"]["model"]
LANGUAGE = config["settings"]["language"]
VOICE = config["settings"]["voice"]


def play_text(t: str):
    load_dotenv()
    client = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )
    audio = client.text_to_speech.convert(
        text=t,
        # voice_id="JBFqnCBsd6RMkjVDRZzb",
        voice_id="gfKKsLN1k0oYYN9n2dXX",  # violetta
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)
