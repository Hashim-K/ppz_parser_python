import numpy as np
from scipy.signal import welch


def get_psd(data, dt):
    """
    Calculates the Power Spectral Density (PSD) of a time-series signal.

    Args:
        data: A numpy array containing the signal data.
        dt: The time step (1 / sampling frequency).

    Returns:
        A tuple containing the frequencies (f) and the PSD (Pxx).
    """
    # Use Welch's method to compute the PSD, which is robust for noisy signals.
    # fs is the sampling frequency.
    fs = 1.0 / dt
    # nperseg defines the length of each segment for averaging.
    f, Pxx = welch(data, fs, nperseg=1024)
    return f, Pxx
