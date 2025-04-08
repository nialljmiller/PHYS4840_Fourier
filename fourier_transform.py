#!/usr/bin/env python3
"""
Fourier Transform Implementation
-------------------------------
A clean, pedagogical implementation of Fourier Transform for teaching purposes.
This module provides functions to compute DFT, inverse DFT, and spectral analysis.

PHYS 4840 - Mathematical and Computational Methods II
"""

import numpy as np


def dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of the input signal.
    
    Parameters:
        x (array): Input signal (time domain)
    
    Returns:
        array: Fourier Transform of x (frequency domain, complex values)
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X


def idft(X):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of the input spectrum.
    
    Parameters:
        X (array): Input spectrum (frequency domain)
    
    Returns:
        array: Inverse Fourier Transform of X (time domain)
    """
    N = len(X)
    x = np.zeros(N, dtype=complex)
    
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    
    # Normalize by N
    x = x / N
    
    return x


def fft(x):
    """
    Compute the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.
    This implementation works for signal lengths that are powers of 2.
    
    Parameters:
        x (array): Input signal (time domain)
    
    Returns:
        array: Fourier Transform of x (frequency domain)
    """
    N = len(x)
    
    # Base case: FFT of a single point is the point itself
    if N == 1:
        return x
    
    # Check if N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Split even and odd indices
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
    # Twiddle factors
    twiddle = np.exp(-2j * np.pi * np.arange(N//2) / N)
    
    # Combine using butterfly pattern
    result = np.zeros(N, dtype=complex)
    half_N = N // 2
    
    for k in range(half_N):
        result[k] = even[k] + twiddle[k] * odd[k]
        result[k + half_N] = even[k] - twiddle[k] * odd[k]
    
    return result


def ifft(X):
    """
    Compute the Inverse Fast Fourier Transform (IFFT).
    
    Parameters:
        X (array): Input spectrum (frequency domain)
    
    Returns:
        array: Inverse Fourier Transform of X (time domain)
    """
    N = len(X)
    
    # Compute the FFT of the conjugate, then conjugate the result and scale
    x = np.conj(fft(np.conj(X))) / N
    
    return x


def compute_periodogram(x, fs=1.0):
    """
    Compute the periodogram (power spectral density estimate) of the input signal.
    
    Parameters:
        x (array): Input signal (time domain)
        fs (float): Sampling frequency in Hz
    
    Returns:
        tuple: (frequencies, power spectrum)
    """
    N = len(x)
    
    # Compute FFT (using NumPy for efficiency)
    X = np.fft.fft(x)
    
    # Compute power spectrum (periodogram)
    power_spectrum = np.abs(X)**2 / N
    
    # Compute frequency axis
    frequencies = np.fft.fftfreq(N, d=1/fs)
    
    return frequencies, power_spectrum


def filter_spectrum(X, frequencies, band_type='low', cutoff=None, band=(None, None)):
    """
    Filter a spectrum in the frequency domain.
    
    Parameters:
        X (array): Input spectrum (from FFT)
        frequencies (array): Frequency axis
        band_type (str): 'low', 'high', or 'band'
        cutoff (float): Cutoff frequency for low/high pass
        band (tuple): (low_cutoff, high_cutoff) for bandpass
    
    Returns:
        array: Filtered spectrum
    """
    X_filtered = X.copy()
    
    if band_type == 'low':
        # Low-pass filter
        X_filtered[np.abs(frequencies) > cutoff] = 0
    
    elif band_type == 'high':
        # High-pass filter
        X_filtered[np.abs(frequencies) < cutoff] = 0
    
    elif band_type == 'band':
        # Band-pass filter
        low, high = band
        X_filtered[(np.abs(frequencies) < low) | (np.abs(frequencies) > high)] = 0
        
    return X_filtered


def reconstruct_signal(X_filtered):
    """
    Reconstruct a time-domain signal from a filtered spectrum.
    
    Parameters:
        X_filtered (array): Filtered spectrum
    
    Returns:
        array: Reconstructed time-domain signal
    """
    # Use NumPy's IFFT for efficiency
    return np.fft.ifft(X_filtered)


def spectrum_to_magnitude_phase(X):
    """
    Convert a complex spectrum to magnitude and phase representation.
    
    Parameters:
        X (array): Complex spectrum from FFT
    
    Returns:
        tuple: (magnitude_spectrum, phase_spectrum)
    """
    magnitude = np.abs(X)
    phase = np.angle(X)
    
    return magnitude, phase