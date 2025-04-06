#!/usr/bin/python3
#########################
# Discrete Fourier Transform (DFT) Implementation
# PHYS 4840 - Math and Computational Methods II
# Week 11 - Fourier Analysis: Theory & Discrete FT
#########################

import numpy as np
import matplotlib.pyplot as plt
import time

def dft_naive(x):
    """
    Compute the Discrete Fourier Transform (DFT) of input signal x using a naive
    implementation (direct application of the definition).
    
    Parameters:
        x (array): Input signal array (complex or real)
        
    Returns:
        array: DFT of x (complex array)
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    # For each output element X[k]
    for k in range(N):
        # For each input element x[n]
        for n in range(N):
            # Compute the DFT sum term by term
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X

def idft_naive(X):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of input frequency domain signal X
    using a naive implementation (direct application of the definition).
    
    Parameters:
        X (array): Input frequency domain signal array (complex or real)
        
    Returns:
        array: IDFT of X (complex array)
    """
    N = len(X)
    x = np.zeros(N, dtype=complex)
    
    # For each output element x[n]
    for n in range(N):
        # For each input element X[k]
        for k in range(N):
            # Compute the IDFT sum term by term
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    
    # Normalize by N
    x = x / N
    
    return x

def dft_matrix_form(x):
    """
    Compute the DFT using matrix multiplication.
    This is still O(N²) complexity but helps visualize the transform.
    
    Parameters:
        x (array): Input signal array (complex or real)
        
    Returns:
        array: DFT of x (complex array)
    """
    N = len(x)
    
    # Create DFT matrix
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    
    # Apply the transform
    X = np.dot(M, x)
    
    return X

def compare_dft_with_numpy(signal_length=128):
    """
    Compare our DFT implementations with NumPy's FFT.
    Also measures execution time.
    
    Parameters:
        signal_length (int): Length of the signal to test
    """
    # Create a test signal (sum of sines)
    t = np.linspace(0, 1, signal_length, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Time our naive DFT implementation
    start_time = time.time()
    X_naive = dft_naive(x)
    naive_time = time.time() - start_time
    
    # Time our matrix DFT implementation
    start_time = time.time()
    X_matrix = dft_matrix_form(x)
    matrix_time = time.time() - start_time
    
    # Time NumPy's FFT implementation
    start_time = time.time()
    X_numpy = np.fft.fft(x)
    numpy_time = time.time() - start_time
    
    # Print timing results
    print(f"Signal length: {signal_length}")
    print(f"Naive DFT time: {naive_time:.6f} seconds")
    print(f"Matrix DFT time: {matrix_time:.6f} seconds")
    print(f"NumPy FFT time: {numpy_time:.6f} seconds")
    print(f"Speedup (Naive vs NumPy): {naive_time/numpy_time:.2f}x")
    print(f"Speedup (Matrix vs NumPy): {matrix_time/numpy_time:.2f}x")
    
    # Verify the results are nearly identical
    print(f"Max absolute difference (Naive vs NumPy): {np.max(np.abs(X_naive - X_numpy)):.6e}")
    print(f"Max absolute difference (Matrix vs NumPy): {np.max(np.abs(X_matrix - X_numpy)):.6e}")
    
    # Plot the magnitude spectra
    plt.figure(figsize=(12, 8))
    freq = np.fft.fftfreq(signal_length, d=t[1]-t[0])
    pos_freq_idx = np.where(freq >= 0)
    
    plt.subplot(3, 1, 1)
    plt.plot(freq[pos_freq_idx], np.abs(X_naive[pos_freq_idx]), label='Naive DFT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectrum - Naive DFT')
    
    plt.subplot(3, 1, 2)
    plt.plot(freq[pos_freq_idx], np.abs(X_matrix[pos_freq_idx]), label='Matrix DFT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectrum - Matrix DFT')
    
    plt.subplot(3, 1, 3)
    plt.plot(freq[pos_freq_idx], np.abs(X_numpy[pos_freq_idx]), label='NumPy FFT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectrum - NumPy FFT')
    
    plt.tight_layout()
    plt.savefig('dft_comparison.png')
    plt.show()

def dft_time_complexity():
    """
    Analyze the time complexity of DFT implementations by measuring
    execution time for different signal lengths.
    """
    signal_lengths = [2**i for i in range(4, 12)]  # 16 to 2048
    naive_times = []
    matrix_times = []
    numpy_times = []
    
    for N in signal_lengths:
        # Create a test signal
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        
        # Naive DFT (only for smaller sizes due to O(N²) complexity)
        if N <= 256:
            start_time = time.time()
            dft_naive(x)
            naive_times.append(time.time() - start_time)
        else:
            naive_times.append(np.nan)  # Too slow for larger signals
        
        # Matrix DFT
        if N <= 512:  # Still O(N²) but a bit faster
            start_time = time.time()
            dft_matrix_form(x)
            matrix_times.append(time.time() - start_time)
        else:
            matrix_times.append(np.nan)
        
        # NumPy FFT
        start_time = time.time()
        np.fft.fft(x)
        numpy_times.append(time.time() - start_time)
    
    # Plot the timing results on a log-log scale
    plt.figure(figsize=(10, 6))
    
    # Theoretical complexity curves for reference
    n_squared = np.array(signal_lengths)**2 / 1e6  # Scaled for visibility
    n_logn = np.array(signal_lengths) * np.log2(np.array(signal_lengths)) / 1e5
    
    plt.loglog(signal_lengths, naive_times, 'o-', label='Naive DFT')
    plt.loglog(signal_lengths, matrix_times, 's-', label='Matrix DFT')
    plt.loglog(signal_lengths, numpy_times, '^-', label='NumPy FFT')
    plt.loglog(signal_lengths, n_squared, '--', label='O(N²) Reference')
    plt.loglog(signal_lengths, n_logn, ':', label='O(N log N) Reference')
    
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.xlabel('Signal Length (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('DFT Implementation Time Complexity')
    
    plt.savefig('dft_time_complexity.png')
    plt.show()

def demonstrate_dft_properties():
    """
    Demonstrate important properties of the DFT:
    1. Linearity
    2. Circular shift (time shift property)
    3. Conjugate symmetry for real signals
    4. Parseval's theorem (energy conservation)
    """
    # Create two test signals
    N = 64
    t = np.linspace(0, 1, N, endpoint=False)
    x1 = np.sin(2 * np.pi * 5 * t)
    x2 = np.cos(2 * np.pi * 10 * t)
    
    # 1. Linearity: DFT(a*x1 + b*x2) = a*DFT(x1) + b*DFT(x2)
    a, b = 2, 3
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    X_combined = np.fft.fft(a * x1 + b * x2)
    X_linear = a * X1 + b * X2
    
    linearity_error = np.max(np.abs(X_combined - X_linear))
    print(f"1. Linearity property error: {linearity_error:.6e}")
    
    # 2. Circular shift
    shift = 10
    x_shifted = np.roll(x1, shift)  # Circular shift in time domain
    X1 = np.fft.fft(x1)
    X_shifted = np.fft.fft(x_shifted)
    
    # The magnitudes should be identical
    mag_error = np.max(np.abs(np.abs(X1) - np.abs(X_shifted)))
    print(f"2. Circular shift magnitude invariance error: {mag_error:.6e}")
    
    # 3. Conjugate symmetry for real signals
    X = np.fft.fft(x1)  # x1 is real
    conjugate_error = np.max(np.abs(X[1:N//2] - np.conj(X[N-1:N//2:-1])))
    print(f"3. Conjugate symmetry error: {conjugate_error:.6e}")
    
    # 4. Parseval's theorem (energy conservation)
    signal_energy = np.sum(np.abs(x1)**2)
    spectrum_energy = np.sum(np.abs(X1)**2) / N  # Normalize by N
    energy_error = np.abs(signal_energy - spectrum_energy)
    print(f"4. Parseval's theorem energy conservation error: {energy_error:.6e}")
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Original and shifted signals (time domain)
    plt.subplot(2, 2, 1)
    plt.plot(t, x1, label='Original')
    plt.plot(t, x_shifted, label=f'Shifted by {shift}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Time Domain Signals')
    
    # Magnitude spectra of original and shifted signals
    freq = np.fft.fftfreq(N, d=t[1]-t[0])
    plt.subplot(2, 2, 2)
    plt.plot(freq, np.abs(X1), label='Original')
    plt.plot(freq, np.abs(X_shifted), '--', label=f'Shifted by {shift}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectra (Shift Invariance)')
    
    # Real signal and its conjugate symmetry
    plt.subplot(2, 2, 3)
    plt.plot(freq, np.real(X), label='Real Part')
    plt.plot(freq, np.imag(X), label='Imaginary Part')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('DFT of Real Signal (Conjugate Symmetry)')
    
    # Linearity property
    plt.subplot(2, 2, 4)
    plt.plot(freq, np.abs(X_combined), label='DFT(a*x1 + b*x2)')
    plt.plot(freq, np.abs(X_linear), '--', label='a*DFT(x1) + b*DFT(x2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Linearity Property')
    
    plt.tight_layout()
    plt.savefig('dft_properties.png')
    plt.show()

def visualize_frequency_resolution():
    """
    Demonstrate frequency resolution in the DFT and 
    how zero-padding can improve it.
    """
    # Create a test signal with two close frequencies
    N = 64  # Original signal length
    t = np.linspace(0, 1, N, endpoint=False)
    f1, f2 = 10, 12  # Close frequencies (in Hz)
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Compute DFT with different amounts of zero-padding
    X_original = np.fft.fft(x)
    X_padded_2x = np.fft.fft(np.pad(x, (0, N)))  # 2x padding
    X_padded_4x = np.fft.fft(np.pad(x, (0, 3*N)))  # 4x padding
    
    # Prepare frequency axes
    freq_original = np.fft.fftfreq(N, d=t[1]-t[0])
    freq_padded_2x = np.fft.fftfreq(2*N, d=t[1]-t[0])
    freq_padded_4x = np.fft.fftfreq(4*N, d=t[1]-t[0])
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Original signal spectrum
    plt.subplot(3, 1, 1)
    plt.plot(freq_original, np.abs(X_original))
    plt.grid(True, alpha=0.3)
    plt.title(f'Original DFT (N={N}, Resolution={1/N:.2f} Hz)')
    plt.xlim(0, 20)  # Focus on the region of interest
    
    # 2x zero-padded spectrum
    plt.subplot(3, 1, 2)
    plt.plot(freq_padded_2x, np.abs(X_padded_2x))
    plt.grid(True, alpha=0.3)
    plt.title(f'2x Zero-Padded DFT (N={2*N}, Resolution={1/(2*N):.2f} Hz)')
    plt.xlim(0, 20)
    
    # 4x zero-padded spectrum
    plt.subplot(3, 1, 3)
    plt.plot(freq_padded_4x, np.abs(X_padded_4x))
    plt.grid(True, alpha=0.3)
    plt.title(f'4x Zero-Padded DFT (N={4*N}, Resolution={1/(4*N):.2f} Hz)')
    plt.xlim(0, 20)
    
    plt.tight_layout()
    plt.savefig('frequency_resolution.png')
    plt.show()
    
    print("Frequency Resolution Analysis:")
    print(f"Original DFT: {1/N:.4f} Hz")
    print(f"2x Zero-Padded: {1/(2*N):.4f} Hz")
    print(f"4x Zero-Padded: {1/(4*N):.4f} Hz")
    print(f"True frequency separation: {f2-f1:.4f} Hz")

# Example usage
if __name__ == "__main__":
    print("Demonstrating Discrete Fourier Transform (DFT) Implementation")
    
    # Compare our DFT implementations with NumPy
    print("\nComparing DFT implementations:")
    compare_dft_with_numpy(signal_length=64)
    
    # Analyze time complexity
    print("\nAnalyzing time complexity:")
    dft_time_complexity()
    
    # Demonstrate DFT properties
    print("\nDemonstrating DFT properties:")
    demonstrate_dft_properties()
    
    # Visualize frequency resolution
    print("\nVisualizing frequency resolution with zero-padding:")
    visualize_frequency_resolution()
