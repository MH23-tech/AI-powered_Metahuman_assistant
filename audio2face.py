import queue
import sys
import json
import requests
import torch
import numpy as np
import time
import threading
import subprocess
import soundfile as sf  # For reading/writing WAV files
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
from py_audio2face import Audio2Face
import sounddevice as sd
from typing import Generator


def get_player_instance():
    """
    Retrieve the available player instances using GET /A2F/Player/GetInstances.
    Prefer the "regular" instance if available; otherwise, fall back to a default core instance.
    """
    url = "http://localhost:8011/A2F/Player/GetInstances"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            regular = data.get("result", {}).get("regular", [])
            if regular:
                instance = regular[0]
                print(f"[INFO] Regular player instance retrieved: {instance}")
                return instance
            else:
                print("[WARNING] No regular player instance found. Using default core instance.")
        else:
            print(f"[WARNING] Failed to get player instances: {response.text}. Using default core instance.")
    except Exception as e:
        print(f"[ERROR] Exception while getting player instances: {e}. Using default core instance.")
    default_instance = "/World/audio2face/CoreFullface"
    print(f"[INFO] Using default instance: {default_instance}")
    return default_instance


def enable_auto_emotion_streaming(instance_identifier):
    """
    Enable auto emotion streaming using POST /A2F/A2E/EnableStreaming.
    Uses the provided instance identifier.
    """
    url = "http://localhost:8011/A2F/A2E/EnableStreaming"
    payload = {
        "a2f_instance": instance_identifier,
        "enable": True
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("Auto emotion streaming enabled successfully.")
            print("Response:", response.text)
        else:
            print("Failed to enable auto emotion streaming:")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] Exception while enabling auto emotion streaming: {e}")


def get_blendshape_solver_node():
    """
    Retrieve the blendshape solver node path using GET /A2F/Exporter/GetBlendShapeSolvers.
    Expected response:
    {
      "status": "OK",
      "result": ["/World/audio2face/BlendshapeSolve"]
    }
    """
    url = "http://localhost:8011/A2F/Exporter/GetBlendShapeSolvers"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "result" in data and data["result"]:
                node_path = data["result"][0]
                print(f"[INFO] Blendshape solver node retrieved: {node_path}")
                return node_path
            else:
                print("[WARNING] No blendshape solver node found in response. Using default.")
                return "/World/audio2face/BlendshapeSolve"
        else:
            print(f"[WARNING] Failed to get blendshape solver node: {response.text}. Using default.")
            return "/World/audio2face/BlendshapeSolve"
    except Exception as e:
        print(f"[ERROR] Exception while getting blendshape solver node: {e}. Using default.")
        return "/World/audio2face/BlendshapeSolve"


def activate_stream_livelink(node_path):
    """
    Activate the StreamLivelink node in Audio2Face by sending a POST request.
    Uses the provided node_path.
    """
    url = "http://localhost:8011/A2F/Exporter/ActivateStreamLivelink"
    payload = {
        "node_path": node_path,
        "value": True
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("Stream Livelink activated successfully.")
            print("Response:", response.text)
        else:
            print("Failed to activate Stream Livelink:")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] Exception while activating Stream Livelink: {e}")


def file_audio_stream_manual(file_path: str) -> Generator[np.ndarray, None, None]:
    """
    Read the entire WAV file (in its original FLOAT format) and yield it as a single NumPy array.
    Before streaming, configure live link settings using GET endpoints.

    Workaround: yield a very short audio clip first to populate blendshape names,
    then yield the remainder of the audio.
    """
    data, samplerate = sf.read(file_path, dtype='float32')
    print(f"[DEBUG] Samplerate: {samplerate}, Total samples: {len(data)}")

    # Retrieve valid player instance and enable auto emotion streaming
    player_instance = get_player_instance()
    enable_auto_emotion_streaming(player_instance)

    # Retrieve blendshape solver node and override if needed
    node_path = get_blendshape_solver_node()
    if node_path == "/World/audio2face/BlendshapeSolve":
        print("[INFO] Overriding blendshape solver node to /World/audio2face/StreamLivelink")
        node_path = "/World/audio2face/StreamLivelink"
    if node_path:
        activate_stream_livelink(node_path)
    else:
        print("[ERROR] Cannot configure live link without a valid node path.")

    # Workaround: push a very short audio clip to populate blendshape names input.
    short_clip = data[:100]  # adjust sample count as needed
    print("[INFO] Pushing short audio clip to populate blendshape names input.")
    yield short_clip
    time.sleep(1)  # wait a moment to let the system update

    # Now yield the rest of the audio (which combined gives the full audio)
    remaining_clip = data[100:]
    yield remaining_clip


class Assistant:
    def __init__(self):
        # Initialize TTS using Coqui VITS model.
        self.tts = TTS(
            model_name="tts_models/en/ljspeech/vits",
            progress_bar=False,
            gpu=torch.cuda.is_available()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts.to(self.device)
        print(f"[INFO] TTS running on {self.device.upper()}")

        # Initialize Audio2Face.
        self.audio2face = Audio2Face()
        # Live link settings will be configured before streaming.

        # Use the TTS native sample rate.
        self.target_sample_rate = self.tts.synthesizer.output_sample_rate
        self.output_file = "output.wav"
        self.output_animation = "output_animation.usd"  # (Optional)

        # Conversation state.
        self.is_speaking = False
        self.audio_queue = queue.Queue()
        self.history = [{
            "role": "system",
            "content": ("You are a friendly assistant called Eve. Keep answers concise. "
                        "Address the user as Mohammad.")
        }]

    def append_history(self, message):
        self.history.append(message)

    def speak(self, text):
        """
        Generate TTS audio from text, save it as a FLOAT WAV file, and stream it to Audio2Face.
        """
        self.is_speaking = True
        print(f"EVE: {text}")
        try:
            self.generate_and_save_audio(text)
            self.stream_audio_file()
        except Exception as e:
            print(f"[ERROR] Speech generation failed: {e}")
        finally:
            self.is_speaking = False

    def generate_and_save_audio(self, text):
        """
        Generate audio via TTS and save it as a FLOAT WAV file.
        """
        audio_data = self.tts.tts(text=text)
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)

        self.target_sample_rate = self.tts.synthesizer.output_sample_rate
        print(f"[INFO] Using TTS native sample rate: {self.target_sample_rate} Hz")

        sf.write(self.output_file, audio_data, self.target_sample_rate, subtype='FLOAT')
        print(f"[INFO] Audio saved to {self.output_file} as FLOAT at {self.target_sample_rate} Hz")

    def stream_audio_file(self):
        """
        Configure live link and stream the saved audio file to Audio2Face.
        """
        try:
            print("[INFO] Streaming audio file in one complete chunk...")
            stream_gen = file_audio_stream_manual(self.output_file)
            info = sf.info(self.output_file)
            samplerate = info.samplerate
            print(f"[DEBUG] Streaming samplerate: {samplerate}")
            self.audio2face.stream_audio(
                audio_stream=stream_gen,
                samplerate=samplerate,
                block_until_playback_is_finished=True
            )
            print("[INFO] Audio streaming completed")
        except Exception as e:
            print(f"[ERROR] Audio2Face streaming failed: {e}")

    def chat(self, text):
        """
        Send the user's text to the chat API and speak the response.
        """
        self.append_history({"role": "user", "content": text})
        payload = {
            "model": "llama3.2",
            "messages": self.history,
            "stream": False
        }
        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
            response_data = response.json()
            if "message" in response_data:
                self.append_history(response_data["message"])
                self.speak(response_data["message"]["content"])
            else:
                print("[ERROR] Invalid response format")
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")

    def cleanup(self):
        """Clean up resources on shutdown."""
        print("[INFO] System shutdown")


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    global assistant
    if not assistant.is_speaking:
        assistant.audio_queue.put(bytes(indata))


def main():
    global assistant
    assistant = Assistant()
    vosk_model = Model("models/vosk-model-small-en-us-0.15")
    input_device = sd.query_devices(None, "input")
    recognizer = KaldiRecognizer(vosk_model, input_device["default_samplerate"])
    try:
        with sd.RawInputStream(
                samplerate=input_device["default_samplerate"],
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=audio_callback
        ):
            print("\n" + "#" * 40)
            print(" Voice Assistant Ready")
            print(" Press CTRL+C to exit")
            print("#" * 40 + "\n")
            while True:
                if not assistant.is_speaking:
                    try:
                        audio_chunk = assistant.audio_queue.get_nowait()
                        if recognizer.AcceptWaveform(audio_chunk):
                            result = json.loads(recognizer.Result())
                            if text := result.get("text", "").strip():
                                print(f"USER: {text}")
                                assistant.chat(text)
                    except queue.Empty:
                        continue
    except KeyboardInterrupt:
        assistant.cleanup()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
