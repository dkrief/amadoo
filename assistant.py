#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import struct
import math
import threading
import random
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import wave
import yaml
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QMetaObject, Q_ARG, QEvent, QTimer, QRectF, QPropertyAnimation, QEasingCurve
import traceback
from openai import AsyncOpenAI, OpenAI
import webrtcvad

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets  # or PySide2
from pocketsphinx import Pocketsphinx
from dotenv import load_dotenv
import openai
import pyaudio

# Import Whisper only if needed locally
try:
    import whisper
except ImportError:
    whisper = None

# Load environment variables from .env, including OPENAI_API_KEY if needed for ChatCompletion
load_dotenv()

#####################################################
# Global Color Definitions
#####################################################

BACKGROUND_COLOR_HEX = "#000"      # Black
DOMINANT_COLOR_HEX = "#52C2B1"    # Dominant color for the text and swirl

BACKGROUND_COLOR = QtGui.QColor(BACKGROUND_COLOR_HEX)
DOMINANT_COLOR = QtGui.QColor(DOMINANT_COLOR_HEX)

#####################################################
# Load YAML Configuration
#####################################################

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml file not found.")
    sys.exit(1)
except yaml.YAMLError as exc:
    print(f"Error parsing config.yaml: {exc}")
    sys.exit(1)

# Required keys for normal operation:
required_keys = [
    'ASSISTANT_NAME',
    'WAKE_WORD',
    'RESPONSE_LANGUAGE',
    'USE_WHISPER_API',
    'ASSISTANT_PURPOSE',
    'OPENAI_MODEL'
]
missing_keys = [key for key in required_keys if key not in config]
if missing_keys:
    raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

ASSISTANT_NAME = config['ASSISTANT_NAME']
WAKE_WORD = config['WAKE_WORD'].lower()
RESPONSE_LANGUAGE = config['RESPONSE_LANGUAGE'].lower()
USE_WHISPER_API = config['USE_WHISPER_API']
ASSISTANT_PURPOSE = config['ASSISTANT_PURPOSE']
OPENAI_MODEL = config.get('OPENAI_MODEL', 'gpt-4o')

# TTS Configuration
ENABLE_TTS = config.get('ENABLE_TTS', False)
TTS_VOICE = config.get('TTS_VOICE', 'alloy')  # Default to 'alloy'

if ASSISTANT_NAME.lower() != "amadou" or WAKE_WORD != "amadou":
    raise NotImplementedError("Assistant name and wake word can only be 'Amadou'.")


print(OPENAI_MODEL)
print(f"TTS Enabled: {ENABLE_TTS}, Voice: {TTS_VOICE}")


# Make sure we have an OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#####################################################
# Part 1: Setup for Pocketsphinx Wake Word Detection
#####################################################

# Define paths to acoustic models based on language
LANGUAGE_MODELS = {
    "fr": {
        "acoustic_model_dir": os.path.join(os.getcwd(), "models", "fr-ptm-5.2"),
        "dictionary_path": os.path.join(os.getcwd(), "models", "fr.dict"),
        "kws_file_path": os.path.join(os.getcwd(), "models", "kws.list"),
        "kws_file_path_strict": os.path.join(os.getcwd(), "models", "kws2.list"),
        "error_sentence": "Désole, je n'ai pas pu traiter votre demande."
    },
    "en": {
        "acoustic_model_dir": os.path.join(os.getcwd(), "models", "en-us"),
        "dictionary_path": os.path.join(os.getcwd(), "models", "cmudict-en-us.dict"),
        "kws_file_path": os.path.join(os.getcwd(), "models", "kws_en.list"),
        "kws_file_path_strict": os.path.join(os.getcwd(), "models", "kws2_en.list"),
        "error_sentence": "Sorry, I couldn't process your request."
    }
}


if RESPONSE_LANGUAGE not in LANGUAGE_MODELS:
    raise ValueError(f"Unsupported language: {RESPONSE_LANGUAGE}")

selected_model = LANGUAGE_MODELS[RESPONSE_LANGUAGE]
acoustic_model_dir = selected_model["acoustic_model_dir"]
dictionary_path = selected_model["dictionary_path"]
kws_file_path = selected_model["kws_file_path"]
kws_file_path_strict = selected_model["kws_file_path_strict"]

# Define global audio constants
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
AMPLITUDE_SMOOTHING = 0.23  # Adjust between 0.0 (no smoothing) and 1.0 (high smoothing)
SILENCE_PADDING_DURATION_MS = 300  # Duration to wait after speech ends (in milliseconds)
VAD_AGGRESSIVENESS = 3           # VAD aggressiveness mode (0-3)

# Pocketsphinx config for tolerant detection
ps_config = {
    "verbose": False,
    "hmm": acoustic_model_dir,
    "lm": False,
    "dict": dictionary_path,
    "kws": kws_file_path,
    "kws_threshold": 1e-30,  # Tolerant threshold
}

#####################################################
# Part 1.1: Audio and Transcription Utilities
#####################################################


def listen_for_potential_wake(pa):
    """
    Blocks until we get a possible wake word (tolerant).
    Returns a buffered audio snippet if something is detected.
    """
    try:
        stream = pa.open(
            format=FORMAT,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=2048
        )
    except Exception as e:
        print(f"Failed to open audio stream in listen_for_potential_wake: {e}")
        return None

    decoder = Pocketsphinx(**ps_config)
    decoder.start_utt()

    buffered_audio = b""
    try:
        while True:
            data = stream.read(2048, exception_on_overflow=False)
            buffered_audio += data
            decoder.process_raw(data, no_search=False, full_utt=False)
            if decoder.hyp() is not None:
                hypothesis = decoder.hyp().hypstr.lower()
                if WAKE_WORD in hypothesis:
                    decoder.end_utt()
                    stream.stop_stream()
                    stream.close()
                    return buffered_audio
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in listen_for_potential_wake.")
        decoder.end_utt()
        stream.stop_stream()
        stream.close()
        raise
    except Exception as e:
        print(f"Error in listen_for_potential_wake: {e}")
        decoder.end_utt()
        stream.stop_stream()
        stream.close()
        return None


def save_audio_buffer(audio_buffer, file_path, sample_rate=16000, channels=1, sample_format=pyaudio.paInt16):
    """
    Saves the audio buffer (bytes) to a WAV file for debugging.
    """
    try:
        pa_temp = pyaudio.PyAudio()
        sample_width = pa_temp.get_sample_size(sample_format)
        pa_temp.terminate()
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_buffer)
        print(f"Debug audio saved to: {file_path}")
    except Exception as e:
        print(f"Error saving debug audio: {e}")


def confirm_wake_word(pa, buffered_audio, additional_time=0.3, debug_save_file="debug_wake.wav"):
    """
    Quick check with a stricter threshold that the wake word was truly spoken.
    """
    try:
        stream = pa.open(
            format=FORMAT,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=2048
        )
    except Exception as e:
        print(f"Failed to open audio stream in confirm_wake_word: {e}")
        traceback.print_exc()
        return False

    # Stricter threshold
    confirm_ps_config = ps_config.copy()
    confirm_ps_config["kws_threshold"] = 8e-5
    confirm_ps_config["kws"] = kws_file_path_strict

    decoder = Pocketsphinx(**confirm_ps_config)
    decoder.start_utt()

    # Process the buffered audio first
    decoder.process_raw(buffered_audio, no_search=False, full_utt=False)

    combined_audio = buffered_audio
    frames_to_read = int(16000 / 2048 * additional_time)
    try:
        for _ in range(frames_to_read):
            data = stream.read(2048, exception_on_overflow=False)
            combined_audio += data
            decoder.process_raw(data, no_search=False, full_utt=False)

        decoder.end_utt()
        stream.stop_stream()
        stream.close()

        # Save combined audio for debugging
        save_audio_buffer(
            combined_audio,
            debug_save_file,
            sample_rate=16000,
            channels=1,
            sample_format=pyaudio.paInt16
        )

        if decoder.hyp() is not None:
            hypothesis = decoder.hyp().hypstr.lower()
            print("Confirmation hypothesis: ", hypothesis)
            return (WAKE_WORD in hypothesis)
        else:
            return False

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in confirm_wake_word.")
        decoder.end_utt()
        stream.stop_stream()
        stream.close()
        return False
    except Exception as e:
        print(f"Error in confirm_wake_word: {e}")
        traceback.print_exc()
        decoder.end_utt()
        stream.stop_stream()
        return False


def record_command(pa, aggressiveness=2, frame_duration=30, padding_duration=300):
    """
    Records audio until silence is detected using Voice Activity Detection (VAD).
    aggressiveness: VAD aggressiveness mode (0-3)
    frame_duration: duration of a frame in ms
    padding_duration: duration of padding in ms after speech stops
    """
    vad = webrtcvad.Vad()
    vad.set_mode(aggressiveness)

    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=int(RATE * frame_duration / 1000),
        )
    except Exception as e:
        print(f"Failed to open audio stream in record_command: {e}")
        return None

    frames = []
    is_speaking = False
    padding_frames = int(padding_duration / frame_duration)
    padding_counter = 0

    print("Listening for your request...")

    try:
        while True:
            frame = stream.read(int(RATE * frame_duration / 1000), exception_on_overflow=False)
            if not frame:
                break
            is_voice = vad.is_speech(frame, RATE)
            if is_voice:
                if not is_speaking:
                    print("Speech started")
                    is_speaking = True
                frames.append(frame)
                padding_counter = 0
            elif is_speaking:
                padding_counter += 1
                if padding_counter > padding_frames:
                    print("Speech ended")
                    break
                frames.append(frame)
            else:
                # Not speaking and no speech before
                pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in record_command.")
        stream.stop_stream()
        stream.close()
        return None
    except Exception as e:
        print(f"Error in record_command: {e}")
        stream.stop_stream()
        stream.close()
        return None

    stream.stop_stream()
    stream.close()

    if not frames:
        return None

    tmp_wave = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        with wave.open(tmp_wave.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        return tmp_wave.name
    except Exception as e:
        print(f"Error writing temporary WAV file: {e}")
        return None


def transcribe_audio(wav_path):
    """
    Transcribes audio either via the Whisper API (if USE_WHISPER_API is True)
    or locally via the Whisper package.
    """
    if USE_WHISPER_API:
        # Use OpenAI Whisper API
        print("Transcribing using Whisper API endpoint...")
        try:
            with open(wav_path, "rb") as audio_file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            # Attempt to handle different response shapes
            if hasattr(transcription, "text"):
                return transcription.text
            elif isinstance(transcription, dict) and "text" in transcription:
                return transcription["text"]
            else:
                return transcription
        except Exception as e:
            print(f"Error during transcription via API: {e}")
            return ""
    else:
        # Use local Whisper
        if whisper is None:
            raise ImportError("Please install the Whisper package or enable USE_WHISPER_API.")
        try:
            print("Loading local Whisper model 'turbo'...")
            model = whisper.load_model("turbo")
            result = model.transcribe(wav_path)
            return result["text"]
        except Exception as e:
            print(f"Error during local transcription: {e}")
            return ""


def get_chat_completion(prompt):
    """
    Uses OpenAI ChatCompletion API to generate a GPT response.
    """
    print("Getting GPT response...")
    full_prompt = (
        f"You are a **{ASSISTANT_PURPOSE}** Your name is 'Amadou'.",
        f"Always respond in **{RESPONSE_LANGUAGE}** even if the user speaks another language.",
        f"User says: '{prompt}'"
    )
    try:
        openai_response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": " ".join(full_prompt)}],
            max_tokens=400,
            temperature=0.5,
        )
        return openai_response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return "Désolé, je n'ai pas pu traiter votre demande."

#####################################################
# Part 1.2: A Brief, Soothing Sound
#####################################################


def play_wave_file(file_path):
    """
    Plays a .wav audio file using PyAudio.
    """
    chunk = 1024
    wf = wave.open(file_path, 'rb')
    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()

#####################################################
# Part 2: Enhanced Visualization
#####################################################


class Dot:
    def __init__(self, angle, distance, width, height, rotation, color):
        self.angle = angle
        self.init_distance = distance
        self.distance = distance
        self.width = width
        self.height = height
        self.rotation = rotation
        self.color = color


class DotsLayer:
    """
    Represents a single layer of dots with its own color palette.
    """

    def __init__(self, num_dots, base_radius, max_extra, color_palette, dominant=False):
        self.num_dots = num_dots
        self.base_radius = base_radius
        self.max_extra = max_extra
        self.color_palette = color_palette
        self.dots = self.init_dots(dominant)
        self.current_amp = 0.0

    def init_dots(self, dominant):
        dots = []
        for _ in range(self.num_dots):
            angle = random.uniform(0, 2 * math.pi)
            distance = self.base_radius
            if dominant:
                width = random.uniform(2, 6)
                height = random.uniform(2, 6)
            else:
                width = random.uniform(1, 4)
                height = random.uniform(1, 4)
            rotation = random.uniform(0, 360)
            color = DOMINANT_COLOR if dominant else random.choice(self.color_palette)
            dots.append(Dot(angle, distance, width, height, rotation, color))
        return dots

    def update(self, amplitude):
        """
        Update each dot's distance based on amplitude (so they expand/contract).
        """
        self.current_amp = amplitude
        expansion = self.max_extra * amplitude
        for dot in self.dots:
            dot.distance = dot.init_distance + expansion

    def draw(self, painter, overall_alpha):
        for dot in self.dots:
            # Scale dot distance by painter scale, so amplitude "breathes" continuously
            x = dot.distance * math.cos(dot.angle)
            y = dot.distance * math.sin(dot.angle)
            scaled_width = dot.width * (1.0 + 0.5 * self.current_amp)
            scaled_height = dot.height * (1.0 + 0.5 * self.current_amp)

            painter.save()
            painter.translate(x, y)
            painter.rotate(dot.rotation)

            # Adjust alpha according to the current state
            dot_color = QtGui.QColor(dot.color)
            dot_color.setAlpha(overall_alpha)

            # Halo
            halo_size = max(scaled_width, scaled_height) * 1.5
            halo_color = QtGui.QColor(dot_color)
            halo_color.setAlpha(int(0.2 * overall_alpha))  # a fraction for the halo
            painter.setBrush(QtGui.QBrush(halo_color))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(0, 0), halo_size, halo_size)

            # Main ellipse
            painter.setBrush(QtGui.QBrush(dot_color))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(0, 0), scaled_width, scaled_height)

            painter.restore()


class DotsBowlItem(QtWidgets.QGraphicsObject):
    """
    Visualization of multiple layers of dots arranged in circular patterns,
    expanding with the audio amplitude. Handles states for transparency,
    scale, and swirling (rotation).
    """

    def __init__(self, layers=5, dots_per_layer=100, parent=None):
        super(DotsBowlItem, self).__init__(parent)
        self.setAcceptedMouseButtons(Qt.NoButton)

        # Real-time amplitude from mic
        self.amplitude = 0.0
        # Layers for the normal bowl
        self.layers = []

        # State can be "LISTENING", "AWAKENED", or "GPT_LOADING"
        self.state = "LISTENING"

        # Scale factor property
        self._scaleFactor = 1.0
        # Dot alpha property (0 = fully transparent, 255 = fully opaque)
        self._dotAlpha = 128  # Start with semi-transparent

        # For swirling animation
        self.isSwirling = False
        self.swirlTimer = None
        self.swirlAngle = 0.0

        # Animation reference to prevent garbage collection
        self.current_animation = None

        # Build the layers
        self.init_layers(layers, dots_per_layer)

        # Timer for chaotic movement
        self.chaotic_timer = QtCore.QTimer()
        self.chaotic_timer.timeout.connect(self.apply_chaotic_movement)
        self.chaotic_movement_intensity = 5.0  # Adjust as needed for desired effect

    def init_layers(self, num_layers, dots_per_layer):
        color_palettes = [
            [QtGui.QColor(25, 25, 112, 180), QtGui.QColor(0, 0, 139, 150)],
            [QtGui.QColor(65, 105, 225, 150), QtGui.QColor(72, 61, 139, 130)],
            [QtGui.QColor(138, 43, 226, 120), QtGui.QColor(123, 104, 238, 110)],
            [QtGui.QColor(106, 90, 205, 100), QtGui.QColor(75, 0, 130, 90)],
            [QtGui.QColor(72, 61, 139, 80), QtGui.QColor(65, 105, 225, 70)]
        ]
        for i in range(num_layers):
            palette = color_palettes[i % len(color_palettes)]
            dominant = (i == 0)
            layer = DotsLayer(
                num_dots=dots_per_layer + (50 if dominant else 0),
                base_radius=30 + (i * 20),
                max_extra=100,
                color_palette=palette,
                dominant=dominant
            )
            self.layers.append(layer)

    # ----------------------
    # QGraphicsItem overrides
    # ----------------------
    def boundingRect(self):
        max_radius = 0
        for lyr in self.layers:
            candidate = lyr.base_radius + lyr.max_extra
            if candidate > max_radius:
                max_radius = candidate
        # Multiply by 1.5 to accommodate scaling
        max_radius *= 1.5
        return QRectF(-max_radius, -max_radius, 2 * max_radius, 2 * max_radius)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Apply scale
        painter.save()
        painter.scale(self._scaleFactor, self._scaleFactor)

        # If swirling, rotate
        if self.isSwirling:
            painter.rotate(self.swirlAngle)

        # Each layer draws with the current alpha
        effective_amp = self.amplitude
        for layer in self.layers:
            layer.update(effective_amp)
            layer.draw(painter, overall_alpha=self._dotAlpha)

        painter.restore()

    # -----------------------------
    # Property Definitions
    # -----------------------------
    @QtCore.pyqtProperty(float)
    def scaleFactor(self):
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, value):
        self._scaleFactor = value
        self.update()

    @QtCore.pyqtProperty(int)
    def dotAlpha(self):
        return self._dotAlpha

    @dotAlpha.setter
    def dotAlpha(self, value):
        self._dotAlpha = value
        self.update()

    def updateAmplitude(self, amp):
        """
        Called by AudioProcessor's signal to update amplitude in real-time.
        """
        if self.state == "GPT_LOADING":
            # Ignore amplitude updates during GPT_LOADING state
            return
        self.amplitude = amp
        self.update()

    # --------------------------------
    # Swirl (rotation) control
    # --------------------------------
    def startSwirl(self):
        if self.swirlTimer:
            self.swirlTimer.stop()
        self.isSwirling = True
        self.swirlAngle = 0.0

        self.swirlTimer = QtCore.QTimer()
        self.swirlTimer.timeout.connect(self._updateSwirl)
        self.swirlTimer.start(30)  # ~33 fps

    def _updateSwirl(self):
        if not self.isSwirling:
            return
        # Rotate by a small increment
        self.swirlAngle += 3.0
        self.update()

    def stopSwirl(self):
        self.isSwirling = False
        if self.swirlTimer:
            self.swirlTimer.stop()

    # --------------------------------
    # State management
    # --------------------------------
    def setState(self, new_state):
        """
        Transitions the bowl to a new state:
         - LISTENING: dot alpha=128, scale=1.0, no swirl
         - AWAKENED: animate alpha from 128->255, scale from 1.0->1.5 over 500ms and start chaotic movement
         - GPT_LOADING: start swirling animation without changing alpha or scale
        """
        self.state = new_state
        if new_state == "LISTENING":
            self.stopSwirl()
            self.chaotic_timer.stop()
            # Animate alpha from current to 128 and scale from current to 1.0
            self.animateScaleAndAlpha(0, 1.2, self._dotAlpha, 60, 60)
            self.update()

        elif new_state == "AWAKENED":
            self.stopSwirl()  # No swirl while just awakened
            self.chaotic_timer.start(50)  # Update every 50ms
            # Animate alpha from 128 -> 255, scale from 1.0 -> 1.5
            self.animateScaleAndAlpha(1.0, 1.5, 128, 255, 500)

        elif new_state == "GPT_LOADING":
            self.chaotic_timer.stop()  # Stop chaotic movement during loading
            self.startSwirl()          # Start swirling animation
            # Optionally reset amplitude to prevent residual effects
            self.amplitude = 0.0
            self.update()

    def animateScaleAndAlpha(self, startScale, endScale, startAlpha, endAlpha, duration=500):
        """
        Helper to animate scaleFactor and dotAlpha in parallel over 'duration' ms.
        """
        # Stop any ongoing animation
        if self.current_animation is not None:
            self.current_animation.stop()

        # Create a parallel animation group
        group = QtCore.QParallelAnimationGroup(self)

        # Scale animation
        scaleAnim = QPropertyAnimation(self, b"scaleFactor")
        scaleAnim.setStartValue(startScale)
        scaleAnim.setEndValue(endScale)
        scaleAnim.setDuration(duration)
        scaleAnim.setEasingCurve(QEasingCurve.InOutCubic)

        # Alpha animation
        alphaAnim = QPropertyAnimation(self, b"dotAlpha")
        alphaAnim.setStartValue(startAlpha)
        alphaAnim.setEndValue(endAlpha)
        alphaAnim.setDuration(duration)
        alphaAnim.setEasingCurve(QEasingCurve.InOutCubic)

        # Add animations to the group
        group.addAnimation(scaleAnim)
        group.addAnimation(alphaAnim)

        # Connect the finished signal to clear the reference
        group.finished.connect(lambda: setattr(self, 'current_animation', None))

        # Store the reference to prevent garbage collection
        self.current_animation = group

        # Start the animation
        group.start()

    def apply_chaotic_movement(self):
        if self.state != "AWAKENED":
            return

        for layer in self.layers:
            for dot in layer.dots:
                # Apply small random perturbations to the angle
                angle_perturb = random.uniform(-0.05, 0.05)  # radians
                dot.angle += angle_perturb

                # Apply small perturbations to distance for vibrational effect
                distance_perturb = random.uniform(-1.0, 1.0) * self.chaotic_movement_intensity
                dot.distance = dot.init_distance + distance_perturb

        self.update()


class MyGraphicsScene(QtWidgets.QGraphicsScene):
    def prepareForPaint(self):
        pass

    def leaveEvent(self, event):
        pass


#####################################################
# TTS Functions
#####################################################

def generate_speech(text, voice=TTS_VOICE):
    """
    Generates speech audio from text using OpenAI's TTS API.

    Args:
        text (str): The text to convert to speech.
        voice (str): The voice to use for TTS.

    Returns:
        str: Path to the generated speech audio file.
    """
    if not ENABLE_TTS:
        print("TTS is disabled in the configuration.")
        return None

    try:
        # Initialize OpenAI client if not already
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Generate speech
        mp3 = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )

        # Save the audio to a temporary file
        buffer = mp3.read()
        speech_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(speech_file.name, "wb") as f:
            f.write(buffer)

        print(f"Speech audio generated at: {speech_file.name}")
        return speech_file.name

    except Exception as e:
        print(f"Error generating speech: {e}")
        return None


def play_mp3_file(file_path):
    """
    Plays an MP3 file using pydub.

    Args:
        file_path (str): Path to the MP3 file.
    """
    try:
        audio = AudioSegment.from_mp3(file_path)
        play(audio)
        os.remove(file_path)
        print(f"Played and deleted speech audio file: {file_path}")
    except FileNotFoundError:
        print(f"MP3 file not found: {file_path}")
    except ImportError:
        print("pydub is not installed. Please install it using 'pip install pydub'.")
    except Exception as e:
        print(f"Error playing MP3 file: {e}")


#####################################################
# Part 2.1: Audio Processor for Amplitude
#####################################################


class AudioProcessor(QtCore.QObject):
    amplitude_signal = pyqtSignal(float)

    def __init__(self, pa, smoothing: float = 0.15):
        super().__init__()
        self.running = True
        self.pa = pa
        self.smoothed_amp = 0.0
        self.smoothing = smoothing  # This will now be set from config

    def start(self):
        try:
            self.stream = self.pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            print(f"Failed to open audio stream in AudioProcessor: {e}")
            self.running = False
            return

        threading.Thread(target=self.process_audio, daemon=True).start()

    def stop(self):
        self.running = False
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass

    def process_audio(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
            except IOError as e:
                if e.errno == -9988:
                    print("Stream closed. Stopping audio processing.")
                    break
                else:
                    print("Audio stream error:", e)
                    continue
            except Exception as e:
                print("Audio stream error:", e)
                continue

            amplitude = self.compute_rms(data)
            # Normalize roughly
            current_amp = min(amplitude / 3000, 1.0)

            # Exponential smoothing
            self.smoothed_amp += self.smoothing * (current_amp - self.smoothed_amp)
            self.amplitude_signal.emit(self.smoothed_amp)

            time.sleep(0.01)

    def compute_rms(self, frame_data):
        count = len(frame_data) // 2
        if count == 0:
            return 0.0
        fmt = f"{count}h"
        try:
            samples = struct.unpack(fmt, frame_data)
        except struct.error as e:
            print(f"Error unpacking audio data: {e}")
            return 0.0
        sum_squares = sum(sample * sample for sample in samples)
        rms = math.sqrt(sum_squares / count)
        return rms

#####################################################
# Part 2.2: A scrollable overlay to show GPT text
#####################################################


class ResponseOverlayWidget(QtWidgets.QWidget):
    """
    An overlay that appears on top of the main window content,
    showing the GPT response with progressive "streaming" text,
    a scrollbar, and a close button.
    """

    closed_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 220); /* semi-transparent black */
            }
        """)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        # Increase padding around entire overlay
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(0)

        # Close button at top-right
        self.close_button = QtWidgets.QPushButton("✕")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.close_button.setFixedWidth(30)
        self.close_button.setFixedHeight(30)
        self.close_button.clicked.connect(self.on_close_clicked)

        # A small horizontal layout for the close button (aligned right)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(self.close_button)

        self.layout.addLayout(top_layout)

        # Scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        # Remove border and make it fully transparent:
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Inside the scroll area: a widget with QVBoxLayout
        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        # Additional padding inside
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        self.label = QtWidgets.QLabel("")
        self.label.setStyleSheet(f"QLabel {{ color: {DOMINANT_COLOR_HEX}; font-size: 24px; }}")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Ensure label expands
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        font = QtGui.QFont()
        font.setPointSize(24)
        self.label.setFont(font)

        self.content_layout.addWidget(self.label)
        self.content_layout.addStretch()  # push text to the top

        self.scroll_area.setWidget(self.content_widget)

        # For progressive text
        self.full_text = ""
        self.displayed_text = ""
        self.char_index = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.stream_text)
        self.setGeometry(parent.rect())

        # For fade animations
        self._fade_anim = None

    def on_close_clicked(self):
        self.closeOverlay()

    def closeOverlay(self):
        self.timer.stop()
        self.hide()
        self.closed_signal.emit()

    def fadeOut(self, duration_ms=300):
        """
        Fades out the overlay over duration_ms, then closes it.
        """
        if self._fade_anim:
            self._fade_anim.stop()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(duration_ms)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._fade_anim.finished.connect(self.closeOverlay)
        self._fade_anim.start()

    def startOverlay(self, text):
        """
        Show the overlay, reset opacity to 1.0, and begin progressive text streaming.
        """
        # Cancel any ongoing fade
        if self._fade_anim:
            self._fade_anim.stop()
        self.setWindowOpacity(1.0)

        # Add disclosure if TTS is enabled
        if ENABLE_TTS:
            disclaimer = "Note: The following audio is generated by AI."
            full_display_text = f"{text}\n\n{disclaimer}"
        else:
            full_display_text = text

        self.full_text = full_display_text
        self.displayed_text = ""
        self.char_index = 0
        self.label.setText("")
        self.show()  # ensure widget is visible

        # Start streaming
        self.timer.start(20)

    def stream_text(self):
        """
        Add a small chunk of text each step, with some random jitter in speed.
        """
        # random chunk size
        chunk_size = random.randint(2, 4)
        if self.char_index < len(self.full_text):
            next_index = min(self.char_index + chunk_size, len(self.full_text))
            self.displayed_text += self.full_text[self.char_index:next_index]
            self.char_index = next_index
            self.label.setText(self.displayed_text)

            # Make the text scroll to bottom with each addition
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )

            # Randomize the next timer interval to vary speeds
            new_interval = random.randint(10, 30)
            self.timer.setInterval(new_interval)
        else:
            self.timer.stop()

#####################################################
# Part 2.3: Main Window
#####################################################


class SoundBowlWindow(QtWidgets.QMainWindow):
    def __init__(self, pa):
        super().__init__()
        self.pa = pa
        self.setWindowTitle(f"{ASSISTANT_NAME} Voice Volume")

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BACKGROUND_COLOR_HEX};
            }}
            QGraphicsView {{
                border: none;
                background-color: {BACKGROUND_COLOR_HEX};
            }}
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.view = QtWidgets.QGraphicsView()
        self.scene = MyGraphicsScene()
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        # Create the dot bowl
        self.dots_bowl = DotsBowlItem(layers=5, dots_per_layer=200)
        self.scene.addItem(self.dots_bowl)
        # Center it (adjust as per your window size)
        self.dots_bowl.setPos(400, 400)

        # Initialize AudioProcessor
        self.audio = AudioProcessor(pa)
        self.audio.amplitude_signal.connect(self.dots_bowl.updateAmplitude)
        self.audio.start()

        # Ensure the initial state is LISTENING
        self.dots_bowl.setState("LISTENING")

        # Overlay to show GPT response
        self.response_overlay = ResponseOverlayWidget(self)
        self.response_overlay.hide()
        self.response_overlay.closed_signal.connect(self.on_overlay_closed)

    @pyqtSlot()
    def wakeWordDetected(self):
        """
        Called from the assistant thread to trigger "Awakened" animation + beep.
        Also fade out any currently displayed text.
        """
        print("wakeWordDetected slot called in GUI.")

        # If text is visible, fade it out
        if self.response_overlay.isVisible():
            self.response_overlay.fadeOut(300)

        self.dots_bowl.setState("AWAKENED")
        play_wave_file("amadou.wav")

    @pyqtSlot()
    def startGptLoading(self):
        """
        Called from the assistant thread once we start calling GPT => swirl.
        Also play the 'whirling' sound.
        """
        print("Starting GPT loading state.")
        self.dots_bowl.setState("GPT_LOADING")
        play_wave_file("ok.wav")  # <--- New sound before loading

    @pyqtSlot(str)
    def showGPTResponse(self, text):
        """
        Called from the assistant thread once GPT has responded.
        The bowl should go back to listening mode behind text, and we show the overlay.
        """
        # Set state back to LISTENING
        self.dots_bowl.setState("LISTENING")

        # Show the GPT response overlay
        self.response_overlay.startOverlay(text)

        # Generate and play speech if TTS is enabled
        if ENABLE_TTS:
            speech_file = generate_speech(text, voice=TTS_VOICE)
            if speech_file:
                threading.Thread(target=play_mp3_file, args=(speech_file,), daemon=True).start()
            else:
                print("Failed to generate speech audio.")

    @pyqtSlot()
    def wakeWordDetectedDuringLoading(self):
        """
        Called from the assistant thread if a wake word is detected during GPT loading.
        Fades out any existing text if shown.
        """
        print("wakeWordDetectedDuringLoading slot called in GUI.")

        if self.response_overlay.isVisible():
            self.response_overlay.fadeOut(300)

        self.dots_bowl.setState("AWAKENED")
        play_wave_file("amadou.wav")

    def on_overlay_closed(self):
        """
        Called when overlay is closed by the user. The bowl is already in listening mode.
        """
        pass

    def closeEvent(self, event):
        self.audio.stop()
        event.accept()

#####################################################
# Part 3: Assistant Thread + Main Entry
#####################################################


def assistant_loop(window, pa, interrupt_event):
    print(f"{ASSISTANT_NAME} started. Listening for wake word: '{WAKE_WORD}'")
    try:
        while True:
            buffered_audio = listen_for_potential_wake(pa)
            if buffered_audio is None:
                print("Failed to capture buffered audio. Restarting loop.")
                continue

            print("Potential wake word detected. Confirming...")
            if confirm_wake_word(pa, buffered_audio, additional_time=0.2):
                print("Wake word confirmed! I'm awake!")
                # Animate & beep in GUI
                QMetaObject.invokeMethod(
                    window,
                    "wakeWordDetected",
                    Qt.QueuedConnection
                )

                # Record the user command using VAD
                recorded_wav = record_command(
                    pa,
                    aggressiveness=VAD_AGGRESSIVENESS,        # Use the configurable aggressiveness
                    frame_duration=30,                        # You can also make this configurable if needed
                    padding_duration=SILENCE_PADDING_DURATION_MS  # Use the configurable padding duration
                )
                if recorded_wav is None:
                    print("Failed to record command. Returning to listen mode.")
                    window.dots_bowl.setState("LISTENING")  # Ensure state is reset
                    continue

                user_text = transcribe_audio(recorded_wav)
                print(f"User said: {user_text}")

                # Let the GUI know we are loading GPT => swirl
                QMetaObject.invokeMethod(
                    window,
                    "startGptLoading",
                    Qt.QueuedConnection
                )

                # Reset the interrupt event before starting GPT processing
                interrupt_event.clear()

                # Start GPT processing in a separate thread
                gpt_thread = threading.Thread(
                    target=process_gpt,
                    args=(user_text, window, interrupt_event),
                    daemon=True
                )
                gpt_thread.start()

                # Start listening for wake word during GPT loading in a separate thread
                wake_during_gpt_thread = threading.Thread(
                    target=listen_for_wake_during_gpt,
                    args=(pa, interrupt_event),
                    daemon=True
                )
                wake_during_gpt_thread.start()

                # Wait for GPT thread to finish or interruption
                while gpt_thread.is_alive():
                    if interrupt_event.is_set():
                        print("GPT loading interrupted by wake word.")
                        QMetaObject.invokeMethod(
                            window,
                            "wakeWordDetectedDuringLoading",
                            Qt.QueuedConnection
                        )
                        # No direct way to stop the GPT thread; it will complete in background
                        break
                    time.sleep(0.1)  # Check every 100ms

                print(f"Returning to listen mode. Say '{WAKE_WORD}' again to wake me.")
            else:
                print("False alarm. Back to listening...")
    except KeyboardInterrupt:
        print("\nCtrl+C detected in assistant loop. Exiting.")
        return
    except Exception as e:
        print(f"Unexpected error in assistant loop: {e}")
        return


def process_gpt(user_text, window, interrupt_event):
    """
    Processes GPT completion. If interrupted, it will still complete but the UI will have been updated.
    """
    response = get_chat_completion(user_text)
    if not interrupt_event.is_set():
        print(f"{ASSISTANT_NAME} says: {response}")

        # Send GPT response to the GUI for progressive display
        QMetaObject.invokeMethod(
            window,
            "showGPTResponse",
            Qt.QueuedConnection,
            Q_ARG(str, response)
        )
    else:
        print("GPT response received but was interrupted.")


def listen_for_wake_during_gpt(pa, interrupt_event):
    """
    Listens for the wake word during GPT loading.
    If detected, sets the interrupt_event to signal GPT loading interruption.
    """
    try:
        stream = pa.open(
            format=FORMAT,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=2048
        )
    except Exception as e:
        print(f"Failed to open audio stream in listen_for_wake_during_gpt: {e}")
        return

    decoder = Pocketsphinx(**ps_config)
    decoder.start_utt()

    try:
        while not interrupt_event.is_set():
            data = stream.read(2048, exception_on_overflow=False)
            decoder.process_raw(data, no_search=False, full_utt=False)
            if decoder.hyp() is not None:
                hypothesis = decoder.hyp().hypstr.lower()
                if WAKE_WORD in hypothesis:
                    print("Wake word detected during GPT loading.")
                    interrupt_event.set()
                    break
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in listen_for_wake_during_gpt.")
    except Exception as e:
        print(f"Error in listen_for_wake_during_gpt: {e}")
    finally:
        decoder.end_utt()
        stream.stop_stream()
        stream.close()


def main():
    pa = pyaudio.PyAudio()
    interrupt_event = threading.Event()
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = SoundBowlWindow(pa)
        window.show()

        # Assistant in background
        assistant_thread = threading.Thread(target=assistant_loop, args=(window, pa, interrupt_event), daemon=True)
        assistant_thread.start()

        sys.exit(app.exec_())
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
