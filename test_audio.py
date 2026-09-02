import pyaudiowpatch as pyaudio
import numpy as np
import time

print("Initializing PyAudio...")
p = pyaudio.PyAudio()

try:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    print("Default speakers:", default_speakers["name"])
    
    loopback_device = None
    if default_speakers.get("isLoopbackDevice", False):
        loopback_device = default_speakers
    else:
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                loopback_device = loopback
                break
                
    if not loopback_device:
        print("Error: No loopback device found.")
        exit(1)
        
    print(f"Found loopback device: {loopback_device['name']} (Index: {loopback_device['index']})")
    
    rate = int(loopback_device["defaultSampleRate"])
    channels = loopback_device["maxInputChannels"]
    print(f"Opening stream at rate={rate}, channels={channels}...")
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=loopback_device["index"],
        frames_per_buffer=1024
    )
    print("Stream opened successfully!")
    
    print("Reading 5 buffers...")
    for i in range(5):
        data = stream.read(1024, exception_on_overflow=False)
        print(f"Buffer {i} read successfully, size={len(data)}")
        
    print("Closing stream...")
    stream.stop_stream()
    stream.close()
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    p.terminate()
