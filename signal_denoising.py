import librosa
import noisereduce as nr
import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, sr, order=5):
    # butterWorth bandpass filter
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, sr, lowcut=300.0, highcut=3400.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y_filtered = lfilter(b, a, data)
    return y_filtered


def rms_normalize(y, target_level=0.1, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2) + eps)
    return y * (target_level / rms)


def denoise_signal(signal, sample_rate):
    """
    Complete audio denoising pipeline:
    1. Noise reduction
    2. Silence removal
    3. RMS normalization
    4. Bandpass filtering (Butterworth)
    """
    # 1. Noise reduction Spectral gating
    # It works by computing a spectrogram of a signal (and optionally a noise signal)
    # and estimating a noise threshold (or gate) for each frequency band of that signal/noise
    # That threshold is used to compute a mask, which gates noise below the frequency-varying threshold.
    y_denoised = nr.reduce_noise(y=signal, sr=sample_rate)

    # 2. Silence removal
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)

    # 3. Volume normalization
    y_normalized = rms_normalize(y_trimmed)

    # 4. Bandpass filtering (for microphone quality differences)
    y_filtered = apply_bandpass_filter(y_normalized, sample_rate)

    # clip to avoid amplitude overflow
    y_filtered = np.clip(y_filtered, -1.0, 1.0)

    return y_filtered
