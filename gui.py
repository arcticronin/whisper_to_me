from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import threading
import sounddevice as sd
import numpy as np
import wave
import whisper
import os
import yaml
import player
from request_from_provider_hugging_face import query_huggingface_api

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
is_recording = False


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


def apply_material_style(root):
    """
    Create a custom ttk theme that approximates Material Design,
    but uses white for button background so icons appear with
    a white background.
    """
    style = ttk.Style(root)

    # Instead of the purple primary color, we set white for the button.
    primary_color = "#FFFFFF"
    on_primary_color = "#000000"  # text color if you had text
    background_color = "#FAFAFA"
    text_color = "#212121"

    style.theme_create(
        "material",
        parent="clam",
        settings={
            "TFrame": {
                "configure": {
                    "background": background_color,
                }
            },
            "TLabel": {
                "configure": {
                    "background": background_color,
                    "foreground": text_color,
                    "font": ("Segoe UI", 10),
                }
            },
            "TButton": {
                "configure": {
                    "background": primary_color,  # White
                    "foreground": on_primary_color,
                    "font": ("Segoe UI", 10, "bold"),
                    "padding": 0,  # Let the image define shape
                    "borderwidth": 0,  # No border => round images look clean
                },
                "map": {
                    "background": [
                        ("active", "#E0E0E0"),  # Slightly grey on click
                        ("disabled", "#CCCCCC"),
                    ],
                    "foreground": [
                        ("active", on_primary_color),
                        ("disabled", "#666666"),
                    ],
                },
            },
        },
    )
    style.theme_use("material")


def start_recording(status_label):
    """Start the recording in a background thread."""
    status_label.config(text="Status: Recording...")
    global recording
    if recording:
        return  # Already recording

    recording = True
    status_label.config(text="Status: Recording...")
    threading.Thread(target=record_audio, daemon=True).start()


def stop_recording(status_label):
    """Stop the recording, save the WAV file, and transcribe."""
    status_label.config(text="Status: Stopping...")
    # ... stopping, saving, transcribing ...

    global recording
    if not recording:
        return  # Not currently recording

    recording = False
    status_label.config(text="Status: Saving & Transcribing...")

    # Wait a moment for the recording thread to fully stop
    # Try do something more robust than just a sleep
    sd.sleep(500)

    save_recording()
    transcribe_audio()

    # get question
    with open("transcriptions/transcription.txt", "r") as file:
        t = file.read()

    # query the model
    query_huggingface_api(content=t)

    # read it out
    with open("transcriptions/response.txt", "r") as file:
        t = file.read()
    player.play_text(t=t)

    # Reset the label after done
    status_label.config(text="Status: Ready")


def toggle_recording(status_label, button, record_icon, stop_icon):
    """
    Toggle between record and stop states for the single button.
    """
    global is_recording
    if not is_recording:
        # Start recording
        is_recording = True
        start_recording(status_label)
        button.config(image=stop_icon)
        button.image = stop_icon
    else:
        # Stop recording
        is_recording = False
        stop_recording(status_label)
        button.config(image=record_icon)
        button.image = record_icon


def main():
    root = tk.Tk()
    root.title("Carlotta tutta Potta")
    root.geometry("440x200")
    root.resizable(False, False)

    # Apply Material-like theme
    apply_material_style(root)

    # Create main frame
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill="both", expand=True)

    # Status label near the top
    status_label = ttk.Label(main_frame, text="Status: Ready")
    status_label.pack(pady=(0, 8))

    # Load the circular icons (record, stop) with high-quality LANCZOS.
    record_img_raw = Image.open("icons/record_icon.png").resize((48, 48), Image.LANCZOS)
    record_icon = ImageTk.PhotoImage(record_img_raw)

    stop_img_raw = Image.open("icons/stop_recording_icon.png").resize(
        (48, 48), Image.LANCZOS
    )
    stop_icon = ImageTk.PhotoImage(stop_img_raw)

    # Create one central button
    toggle_button = ttk.Button(
        main_frame,
        image=record_icon,
        command=lambda: toggle_recording(
            status_label, toggle_button, record_icon, stop_icon
        ),
    )
    toggle_button.pack(expand=True, anchor="center")

    # Keep references to images to prevent garbage collection
    toggle_button.image = record_icon

    root.mainloop()


if __name__ == "__main__":
    main()
