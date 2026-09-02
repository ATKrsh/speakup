import numpy as np
from faster_whisper import WhisperModel
import time

print("Loading model...")
try:
    model = WhisperModel("tiny", device="cpu", compute_type="float32")
    print("Model loaded successfully!")
    
    # Create dummy audio (1 second of silence at 16000Hz)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    print("Transcribing dummy audio...")
    segments, info = model.transcribe(dummy_audio, beam_size=1)
    results = list(segments)
    print("Transcription completed successfully! Results:", results)
except Exception as e:
    import traceback
    print("Caught Python Exception:")
    traceback.print_exc()
