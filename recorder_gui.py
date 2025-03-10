import tkinter as tk
import threading
import sounddevice as sd
import numpy as np
import wave
import whisper
import os
import yaml
import player

# Load the YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access keys and settings
OPENAI_API_KEY = config["api_keys"]["openai"]
ELEVENLABS_API_KEY = config["api_keys"]["elevenlabs"]
MODEL = config["settings"]["model"]
LANGUAGE = config["settings"]["language"]
VOICE = config["settings"]["voice"]

# --- Settings ---
SAMPLE_RATE = 44100
CHANNELS = 1
MODEL = whisper.load_model("base")

# --- File paths ---
OUTPUT_WAV = "transcriptions/voce.wav"
TRANSCRIPTION_FILE = "transcriptions/transcription.txt"

# --- Global State ---
recording = False
audio_data = []


def record_audio():
    """Continuously reads audio while `recording` is True."""
    global recording, audio_data
    audio_data = []

    def callback(indata, frames, time, status):
        if recording:
            audio_data.append(indata.copy())

    # Open a recording stream and keep it active until recording = False
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=callback
    ):
        while recording:
            sd.sleep(100)


def start_recording(status_label):
    """Start the recording in a background thread."""
    global recording
    if recording:
        return  # Already recording

    recording = True
    status_label.config(text="Status: Recording...")
    threading.Thread(target=record_audio, daemon=True).start()


def stop_recording(status_label):
    """Stop the recording, save the WAV file, and transcribe."""
    global recording
    if not recording:
        return  # Not currently recording

    recording = False
    status_label.config(text="Status: Saving & Transcribing...")

    # Wait a moment for the recording thread to fully stop
    # Typically you might do something more robust than just a short sleep
    sd.sleep(500)

    save_recording()
    transcribe_audio()

    # TODO get answer

    # Read it
    with open("transcriptions/transcription.txt", "r") as file:
        t = file.read()
    player.play_text(t=t)

    status_label.config(text="Status: Ready")


def save_recording():
    """Save the audio_data to a WAV file."""
    if not audio_data:
        print("No audio recorded.")
        return

    audio = np.concatenate(audio_data, axis=0)
    with wave.open(OUTPUT_WAV, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    print(f"Audio saved to {OUTPUT_WAV}")


def transcribe_audio():
    """Use Whisper to transcribe the saved audio and write to a text file."""
    if not os.path.exists(OUTPUT_WAV):
        print("No WAV file found to transcribe.")
        return

    print("Transcribing...")
    result = MODEL.transcribe(OUTPUT_WAV)
    with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Transcription saved to {TRANSCRIPTION_FILE}")
    print(f"Transcription Text: {result['text']}")


def main():
    """Create the GUI and run the Tkinter main event loop."""
    # Create the main window
    root = tk.Tk()
    root.title("Audio Recorder")

    # Create a frame to hold widgets
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack()

    # Label to show status
    status_label = tk.Label(frame, text="Status: Ready")
    status_label.pack(pady=5)

    # Registre button
    start_button = tk.Button(
        frame, text="Registra", command=lambda: start_recording(status_label)
    )
    start_button.pack(pady=5, fill="x")

    # Stop button
    stop_button = tk.Button(
        frame, text="Stop", command=lambda: stop_recording(status_label)
    )
    stop_button.pack(pady=5, fill="x")

    root.mainloop()


if __name__ == "__main__":
    main()
