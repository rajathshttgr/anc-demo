
import numpy as np
import soundfile as sf
import wave
import contextlib

def load_audio(path, target_sr=None):
    """Load WAV/FLAC/OGG with soundfile. Returns mono float32 signal and samplerate.
    If target_sr is set, resample using scipy."""
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    if (target_sr is not None) and (target_sr != sr):
        from scipy.signal import resample_poly
        # Use rational approximation to resample
        import math
        g = math.gcd(target_sr, sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)
        sr = target_sr
    return data, sr

def save_wav(path, y, sr):
    y = np.asarray(y, dtype=np.float32)
    sf.write(path, y, sr, subtype="PCM_16")

def record_seconds_pyaudio(seconds=5, sr=16000, device_index=None):
    """Record mono audio using PyAudio. Returns float32 numpy array."""
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, input=True,
                     frames_per_buffer=1024, input_device_index=device_index)
    frames = []
    for _ in range(int(sr / 1024 * seconds)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream(); stream.close(); pa.terminate()
    return np.concatenate(frames, axis=0)

def record_seconds_sounddevice(seconds=5, sr=16000, device=None):
    """Record mono audio using sounddevice as a fallback."""
    import sounddevice as sd
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return rec[:,0]

