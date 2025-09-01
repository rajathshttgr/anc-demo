
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_waveform(y, sr, title="Waveform"):
    t = np.arange(len(y))/sr
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()

def plot_spectrogram(y, sr, nperseg=512, noverlap=256, title="Spectrogram"):
    f, t, Sxx = spectrogram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10*np.log10(Sxx + 1e-10)
    plt.figure()
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.tight_layout()
