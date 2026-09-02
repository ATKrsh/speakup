import sys
import os
import ctypes

# Programmatically hide console window at startup if running compiled executable
try:
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
except Exception:
    pass

import time
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import pyttsx3

import tkinter as tk
from tkinter import font as tkfont

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_SIZE = "tiny"  # Use 'tiny' for fast CPU inference (~70MB download)
MIN_RMS_THRESHOLD = 0.003  # Minimum RMS energy to process audio (filters digital silence)
CHUNK_DURATION = 3.0  # Translate in 3-second blocks for real-time response

# Global Queues and States
audio_queue = queue.Queue()
tts_queue = queue.Queue()

is_running = True
current_mode = "subtitle"  # 'subtitle' or 'voiceover'

# ---------------------------------------------------------
# Audio Converter Helpers
# ---------------------------------------------------------
def convert_audio(data_bytes, channels, sample_width, from_rate, to_rate=16000):
    if sample_width == 2:
        audio = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(data_bytes, dtype=np.float32)
    else:
        return None
    
    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = audio.mean(axis=1)
        
    if from_rate != to_rate:
        duration = len(audio) / from_rate
        new_len = int(duration * to_rate)
        audio = np.interp(
            np.linspace(0, len(audio), new_len, endpoint=False),
            np.arange(len(audio)),
            audio
        )
    return audio.astype(np.float32)

# ---------------------------------------------------------
# Thread 1: Audio Loopback Recording (Real-time Block Mode)
# ---------------------------------------------------------
def audio_record_thread():
    global is_running
    p = pyaudio.PyAudio()
    try:
        # Find WASAPI loopback device
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        loopback_device = None
        if default_speakers.get("isLoopbackDevice", False):
            loopback_device = default_speakers
        else:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    loopback_device = loopback
                    break
        
        if not loopback_device:
            print("[Audio] Error: No loopback device found.")
            return

        print(f"[Audio] Capturing from: {loopback_device['name']}")
        
        rate = int(loopback_device["defaultSampleRate"])
        channels = loopback_device["maxInputChannels"]
        chunk = 1024
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=loopback_device["index"],
            frames_per_buffer=chunk
        )
        
        # Calculate how many chunks make up the block duration
        chunks_per_block = int((CHUNK_DURATION * rate) / chunk)
        block_buffer = []
        
        while is_running:
            try:
                data = stream.read(chunk, exception_on_overflow=False)
            except IOError:
                continue
                
            block_buffer.append(data)
            
            if len(block_buffer) >= chunks_per_block:
                # Process the block
                raw_bytes = b"".join(block_buffer)
                
                # Check RMS energy to verify if sound was actually playing
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(samples**2)) if len(samples) > 0 else 0.0
                
                if rms > MIN_RMS_THRESHOLD:
                    converted = convert_audio(raw_bytes, channels, 2, rate)
                    if converted is not None:
                        audio_queue.put((converted, CHUNK_DURATION))
                
                # Reset buffer
                block_buffer = []
                
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"[Audio Error] {e}")
    finally:
        p.terminate()

# ---------------------------------------------------------
# Thread 2: Whisper Translation
# ---------------------------------------------------------
def translation_thread(gui_app):
    global is_running
    print("[Whisper] Loading Whisper model...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="float32")
    print("[Whisper] Model loaded successfully.")
    
    while is_running:
        try:
            audio_np, original_duration = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
        try:
            # Task 'translate' automatically transcribes non-English speech to English
            segments, info = model.transcribe(audio_np, task="translate", beam_size=5)
            text = " ".join([seg.text for seg in segments]).strip()
            
            if text:
                print(f"[Translated] ({info.language} -> EN): {text}")
                gui_app.update_subtitle(text)
                if current_mode == "voiceover":
                    tts_queue.put((text, original_duration))
        except Exception as e:
            print(f"[Translation Error] {e}")

# ---------------------------------------------------------
# Thread 3: TTS Playback
# ---------------------------------------------------------
def tts_thread():
    global is_running
    engine = pyttsx3.init()
    
    while is_running:
        try:
            text, duration = tts_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
        try:
            words = len(text.split())
            if duration > 0.1:
                target_rate = int((words / duration) * 60)
                target_rate = max(120, min(target_rate, 280))
            else:
                target_rate = 200
                
            engine.setProperty('rate', target_rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")

# ---------------------------------------------------------
# Tkinter GUI App
# ---------------------------------------------------------
class SpeakUpGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main window
        
        # Subtitle Window
        self.sub_win = tk.Toplevel(self.root)
        self.sub_win.overrideredirect(True)
        self.sub_win.attributes("-topmost", True)
        
        # Use Windows built-in click-through color transparency key
        self.sub_win.configure(bg="#010101")
        self.sub_win.attributes("-transparentcolor", "#010101")
        
        # Screen geometry
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        sub_width = int(self.screen_w * 0.8)
        sub_height = 120
        sub_x = int((self.screen_w - sub_width) / 2)
        sub_y = int(self.screen_h - sub_height - 100)
        self.sub_win.geometry(f"{sub_width}x{sub_height}+{sub_x}+{sub_y}")
        
        self.sub_font = tkfont.Font(family="Outfit", size=24, weight="bold")
        
        # Subtitle label with black background (which is transparent) and white text
        # Draw text with a clean dark shadow or background frame
        self.sub_label = tk.Label(
            self.sub_win, 
            text="", 
            font=self.sub_font, 
            fg="yellow",        # Yellow is highly readable for subtitles
            bg="#010101", 
            wraplength=sub_width - 40,
            justify="center"
        )
        self.sub_label.pack(expand=True, fill="both")
        
        # Floating Widget Button Window
        self.widget_win = tk.Toplevel(self.root)
        self.widget_win.overrideredirect(True)
        self.widget_win.attributes("-topmost", True)
        self.widget_win.attributes("-alpha", 0.7)  # Idle opacity
        self.widget_win.geometry(f"60x60+{self.screen_w - 100}+150")
        
        # Use transparent color key for widget
        self.widget_win.configure(bg="#020202")
        self.widget_win.attributes("-transparentcolor", "#020202")
        
        self.canvas = tk.Canvas(self.widget_win, width=60, height=60, bg="#020202", highlightthickness=0)
        self.canvas.pack()
        
        self.draw_widget()
        
        # Dragging bindings
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_click)
        self.widget_win.bind("<Enter>", self.on_enter)
        self.widget_win.bind("<Leave>", self.on_leave)
        
        self.start_x = 0
        self.start_y = 0
        self.has_dragged = False
        
        self.fade_job = None

    def draw_widget(self):
        self.canvas.delete("all")
        
        if current_mode == "subtitle":
            fill_color = "#2ecc71"
            border_color = "#27ae60"
            mode_text = "SUB"
        else:
            fill_color = "#e67e22"
            border_color = "#d35400"
            mode_text = "VO"
            
        self.canvas.create_oval(3, 3, 57, 57, fill=fill_color, outline=border_color, width=2)
        self.canvas.create_text(30, 30, text=mode_text, fill="white", font=("Outfit", 10, "bold"))

    def on_drag_start(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.win_x = self.widget_win.winfo_x()
        self.win_y = self.widget_win.winfo_y()
        self.has_dragged = False

    def on_drag_motion(self, event):
        dx = event.x_root - self.start_x
        dy = event.y_root - self.start_y
        if abs(dx) > 3 or abs(dy) > 3:
            self.has_dragged = True
        self.widget_win.geometry(f"+{self.win_x + dx}+{self.win_y + dy}")

    def on_click(self, event):
        global current_mode
        if not self.has_dragged:
            if current_mode == "subtitle":
                current_mode = "voiceover"
                print("[Mode] Switched to Voice-Over Mode (Overlay English TTS)")
            else:
                current_mode = "subtitle"
                print("[Mode] Switched to Subtitle Mode (Visual Only)")
            self.draw_widget()

    def on_enter(self, event):
        self.widget_win.attributes("-alpha", 1.0)

    def on_leave(self, event):
        self.widget_win.attributes("-alpha", 0.7)

    def update_subtitle(self, text):
        self.root.after(0, self._set_subtitle_text, text)

    def _set_subtitle_text(self, text):
        if self.fade_job:
            self.root.after_cancel(self.fade_job)
        self.sub_label.config(text=text)
        self.fade_job = self.root.after(5000, self._clear_subtitle)

    def _clear_subtitle(self):
        self.sub_label.config(text="")

    def start(self):
        self.root.mainloop()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    import traceback
    
    def log_error(msg):
        try:
            with open("E:/workspace/speakup/speakup_error.log", "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except:
            pass

    log_error("--- App Starting ---")
    
    def thread_wrapper(target, name, *args):
        try:
            target(*args)
        except Exception as e:
            err_str = f"Exception in thread {name}:\n{traceback.format_exc()}"
            print(err_str)
            log_error(err_str)

    gui = SpeakUpGUI()
    
    t1 = threading.Thread(target=thread_wrapper, args=(audio_record_thread, "AudioRecord"), daemon=True)
    t2 = threading.Thread(target=thread_wrapper, args=(translation_thread, "Translation", gui), daemon=True)
    t3 = threading.Thread(target=thread_wrapper, args=(tts_thread, "TTS"), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    print("[SpeakUp] Floating translator active (Tkinter version).")
    log_error("UI and threads started successfully.")
    
    try:
        gui.start()
    except Exception as e:
        err_str = f"Exception in main GUI loop:\n{traceback.format_exc()}"
        print(err_str)
        log_error(err_str)
    finally:
        is_running = False
        log_error("--- App Exiting ---")
