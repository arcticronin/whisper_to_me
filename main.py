from elevenlabs.client import ElevenLabs
from elevenlabs import play
import yaml
from importlib import reload
import player
import recorder_gui

reload(player)

# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access keys and settings
OPENAI_API_KEY = config["api_keys"]["openai"]
ELEVENLABS_API_KEY = config["api_keys"]["elevenlabs"]
MODEL = config["settings"]["model"]
LANGUAGE = config["settings"]["language"]
VOICE = config["settings"]["voice"]


# Example usage
if __name__ == "__main__":

    print("Starting the recorder GUI...")
    recorder_gui.main()

    # produce the transcription
    with open("transcriptions/transcription.txt", "r") as file:
        t = file.read()

    player.play_text(t=t)
