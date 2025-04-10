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





def fft_bluestein(x):
    N = len(x)
    M = 2**int(np.ceil(np.log2(2*N - 1)))  # Next power of 2 >= 2N - 1
    a = np.array(x, dtype=complex)

    # Chirp signal
    n = np.arange(N)
    chirp = np.exp(1j * np.pi * (n**2) / N)
    
    a_chirp = a * chirp
    b = np.zeros(M, dtype=complex)
    b[:N] = np.exp(-1j * np.pi * (n**2) / N)
    b[-(N-1):] = np.exp(-1j * np.pi * (n[1:][::-1]**2) / N)

    A = np.fft.fft(a_chirp, n=M)
    B = np.fft.fft(b, n=M)
    C = A * B
    c = np.fft.ifft(C)[:N]
    return c * chirp



def fft_zeropad(x):
    N = len(x)
    next_pow2 = 1 << (N - 1).bit_length()
    x_padded = np.pad(x, (0, next_pow2 - N), mode='constant')
    return fft_radix2(x_padded)



def fft_ct(x):
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






def compress_audio_fft(audio, keep_ratio=0.1):
    """
    Compress audio by keeping only the top `keep_ratio` frequency components (by magnitude).
    
    Parameters:
        audio (np.ndarray): Time-domain audio signal
        keep_ratio (float): Fraction of strongest frequencies to keep (0 < keep_ratio <= 1)
        
    Returns:
        compressed_audio (np.ndarray): Reconstructed audio from compressed frequency domain
        X_compressed (np.ndarray): The compressed spectrum (mostly zero)
    """
    N = len(audio)
    X = np.fft.fft(audio)
    magnitudes = np.abs(X)
    
    # Determine how many frequencies to keep
    #new_N = N * keep_ratio? something like that...

    # Get indices of top frequencies by magnitude
    #can we assume its already sorted? np.argsort() might be useful here...

    # Create a compressed version of the spectrum
    #x_compressed = x (but only the kept indices)

    # Inverse FFT to get time-domain signal
    #remember we did all of this on the frequency domain, turn it back into a signal...
    
    return 
