
import numpy as np
from scipy.signal import stft, istft

def _stft(y, sr, n_fft=512, hop=128, win="hann"):
    f, t, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=win, boundary=None)
    return f, t, Z

def _istft(Z, sr, n_fft=512, hop=128, win="hann"):
    _, y = istft(Z, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window=win, boundary=None)
    return y

def estimate_noise_power_mag(Z, percentile=20):
    """Estimate noise power from low-magnitude time frames (percentile across time)."""
    mag = np.abs(Z)
    thr = np.percentile(mag, percentile, axis=1, keepdims=True)
    noise_mag = np.minimum(mag, thr)
    noise_pow = (noise_mag**2).mean(axis=1, keepdims=True)
    return noise_pow

def spectral_subtraction(y, sr, n_fft=512, hop=128, alpha=1.0, floor=0.002):
    f, t, Z = _stft(y, sr, n_fft, hop)
    noise_pow = estimate_noise_power_mag(Z)  # shape (freq, 1)
    Y = Z.copy()
    S_mag = np.maximum(np.abs(Y)**2 - alpha*noise_pow, floor)
    phase = np.angle(Y)
    S = np.sqrt(S_mag) * np.exp(1j*phase)
    out = _istft(S, sr, n_fft, hop)
    return out.astype(np.float32)

def wiener_filter(y, sr, n_fft=512, hop=128, floor=0.002):
    f, t, Z = _stft(y, sr, n_fft, hop)
    mag2 = np.abs(Z)**2
    noise_pow = estimate_noise_power_mag(Z)
    Sxx = np.maximum(mag2 - noise_pow, 0.0)
    H = Sxx / (Sxx + noise_pow + 1e-10)
    S = H * Z
    out = _istft(S, sr, n_fft, hop)
    return out.astype(np.float32)
