#!/usr/bin/python3
#########################
# Fast Fourier Transform (FFT) Implementation
# PHYS 4840 - Math and Computational Methods II
# Week 11 - Fourier Analysis: Theory & Discrete FT
#########################

import numpy as np
import matplotlib.pyplot as plt
import time

def fft_recursive(x):
    """
    Compute the FFT of input signal x using the Cooley-Tukey algorithm.
    This recursive implementation works for signal lengths that are powers of 2.
    
    Parameters:
        x (array): Input signal array (complex or real)
        
    Returns:
        array: FFT of x (complex array)
    """
    N = len(x)
    
    # Base case: DFT of a single point is the point itself
    if N == 1:
        return x
    
    # Check if N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Split even and odd indices
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    
    # Combine the results
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    X = np.zeros(N, dtype=complex)
    
    # Use the butterfly pattern
    half_N = N // 2
    X[:half_N] = even + factor[:half_N] * odd
    X[half_N:] = even + factor[half_N:] * odd
    
    return X

def ifft_recursive(X):
    """
    Compute the Inverse FFT of input spectrum X using the Cooley-Tukey algorithm.
    This recursive implementation works for signal lengths that are powers of 2.
    
    Parameters:
        X (array): Input spectrum array (complex)
        
    Returns:
        array: IFFT of X (complex array)
    """
    N = len(X)
    
    # Compute the FFT of the conjugate of X, then conjugate the result and scale
    x = np.conj(fft_recursive(np.conj(X))) / N
    
    return x

def fft_iterative(x):
    """
    Compute the FFT of input signal x using an iterative implementation
    of the Cooley-Tukey algorithm.
    This implementation works for signal lengths that are powers of 2.
    
    Parameters:
        x (array): Input signal array (complex or real)
        
    Returns:
        array: FFT of x (complex array)
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    # Check if N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Number of stages (log2(N))
    num_stages = int(np.log2(N))
    
    # Bit-reversed copy of the input
    X = x[bit_reversal_permutation(N)]
    
    # Butterfly computation
    for stage in range(1, num_stages + 1):
        butterfly_size = 2 ** stage  # Size of each butterfly
        half_size = butterfly_size // 2  # Half size for twiddle factors
        
        # Twiddle factors
        twiddle = np.exp(-2j * np.pi * np.arange(half_size) / butterfly_size)
        
        # Process each group
        for group in range(0, N, butterfly_size):
            for k in range(half_size):
                idx1 = group + k
                idx2 = group + k + half_size
                
                # Butterfly operation
                temp = X[idx2] * twiddle[k]
                X[idx2] = X[idx1] - temp
                X[idx1] = X[idx1] + temp
    
    return X

def bit_reversal_permutation(N):
    """
    Generate bit-reversal permutation indices for FFT.
    
    Parameters:
        N (int): Length of the signal (must be a power of 2)
        
    Returns:
        array: Bit-reversed indices
    """
    num_bits = int(np.log2(N))
    indices = np.arange(N)
    
    # Manually perform bit reversal for each index
    reversed_indices = np.zeros(N, dtype=int)
    for i in range(N):
        # Convert i to binary, reverse it, and convert back to decimal
        binary = format(i, f'0{num_bits}b')
        reversed_binary = binary[::-1]  # Reverse the binary representation
        reversed_indices[i] = int(reversed_binary, 2)
    
    return reversed_indices

def compare_fft_implementations(signal_length=1024):
    """
    Compare different FFT implementations and measure execution time.
    
    Parameters:
        signal_length (int): Length of the signal to test (must be a power of 2)
    """
    # Ensure signal_length is a power of 2
    if signal_length & (signal_length - 1) != 0:
        signal_length = 2 ** int(np.log2(signal_length))
        print(f"Adjusting signal length to nearest power of 2: {signal_length}")
    
    # Create a test signal (sum of sines)
    t = np.linspace(0, 1, signal_length, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.25 * np.sin(2 * np.pi * 50 * t)
    
    # Time our recursive FFT implementation
    start_time = time.time()
    X_recursive = fft_recursive(x)
    recursive_time = time.time() - start_time
    
    # Time our iterative FFT implementation
    start_time = time.time()
    X_iterative = fft_iterative(x)
    iterative_time = time.time() - start_time
    
    # Time NumPy's FFT implementation
    start_time = time.time()
    X_numpy = np.fft.fft(x)
    numpy_time = time.time() - start_time
    
    # Print timing results
    print(f"Signal length: {signal_length}")
    print(f"Recursive FFT time: {recursive_time:.6f} seconds")
    print(f"Iterative FFT time: {iterative_time:.6f} seconds")
    print(f"NumPy FFT time: {numpy_time:.6f} seconds")
    print(f"Recursive vs NumPy speedup: {numpy_time/recursive_time:.2f}x")
    print(f"Iterative vs NumPy speedup: {numpy_time/iterative_time:.2f}x")
    
    # Verify the results are nearly identical
    print(f"Max absolute difference (Recursive vs NumPy): {np.max(np.abs(X_recursive - X_numpy)):.6e}")
    print(f"Max absolute difference (Iterative vs NumPy): {np.max(np.abs(X_iterative - X_numpy)):.6e}")
    
    # Plot the magnitude spectra
    plt.figure(figsize=(12, 9))
    freq = np.fft.fftfreq(signal_length, d=t[1]-t[0])
    pos_freq_idx = np.where(freq >= 0)
    
    plt.subplot(4, 1, 1)
    plt.plot(t[:100], x[:100])  # Plot just a portion for clarity
    plt.grid(True, alpha=0.3)
    plt.title('Original Signal (first 100 samples)')
    
    plt.subplot(4, 1, 2)
    plt.plot(freq[pos_freq_idx], np.abs(X_recursive[pos_freq_idx]), label='Recursive FFT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectrum - Recursive FFT')
    
    plt.subplot(4, 1, 3)
    plt.plot(freq[pos_freq_idx], np.abs(X_iterative[pos_freq_idx]), label='Iterative FFT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectrum - Iterative FFT')
    
    plt.subplot(4, 1, 4)
    plt.plot(freq[pos_freq_idx], np.abs(X_numpy[pos_freq_idx]), label='NumPy FFT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Magnitude Spectrum - NumPy FFT')
    
    plt.tight_layout()
    plt.savefig('fft_comparison.png')
    plt.show()

def visualize_fft_complexity():
    """
    Visualize the time complexity of different Fourier transform implementations.
    """
    # Use powers of 2 for signal lengths
    signal_lengths = [2**i for i in range(3, 14)]  # 8 to 8192
    
    # Arrays to store timing results
    naive_times = []
    recursive_times = []
    iterative_times = []
    numpy_times = []
    
    # Define a separate naive DFT for this test
    def naive_dft(x):
        N = len(x)
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
        return X
    
    for N in signal_lengths:
        # Create a test signal
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        
        # Naive DFT (only for smaller sizes due to O(N²) complexity)
        if N <= 128:
            start_time = time.time()
            naive_dft(x)
            naive_times.append(time.time() - start_time)
        else:
            naive_times.append(np.nan)  # Too slow for larger signals
        
        # Recursive FFT
        start_time = time.time()
        fft_recursive(x)
        recursive_times.append(time.time() - start_time)
        
        # Iterative FFT
        start_time = time.time()
        fft_iterative(x)
        iterative_times.append(time.time() - start_time)
        
        # NumPy FFT
        start_time = time.time()
        np.fft.fft(x)
        numpy_times.append(time.time() - start_time)
    
    # Plot the timing results on a log-log scale
    plt.figure(figsize=(10, 6))
    
    # Theoretical complexity curves for reference
    n_squared = np.array(signal_lengths)**2 / 1e6  # Scaled for visibility
    n_logn = np.array(signal_lengths) * np.log2(np.array(signal_lengths)) / 1e5
    
    plt.loglog(signal_lengths, naive_times, 'o-', label='Naive DFT O(N²)')
    plt.loglog(signal_lengths, recursive_times, 's-', label='Recursive FFT O(N log N)')
    plt.loglog(signal_lengths, iterative_times, '^-', label='Iterative FFT O(N log N)')
    plt.loglog(signal_lengths, numpy_times, 'd-', label='NumPy FFT')
    plt.loglog(signal_lengths, n_squared, '--', label='O(N²) Reference')
    plt.loglog(signal_lengths, n_logn, ':', label='O(N log N) Reference')
    
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.xlabel('Signal Length (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Fourier Transform Implementation Time Complexity')
    
    plt.savefig('fft_time_complexity.png')
    plt.show()

def reconstruct_signal_from_fft():
    """
    Demonstrate signal reconstruction from FFT and IFFT.
    Also show partial reconstruction with different numbers of frequency components.
    """
    # Create a test signal
    N = 512
    t = np.linspace(0, 1, N, endpoint=False)
    
    # Sum of sines with different frequencies
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.25 * np.sin(2 * np.pi * 50 * t)
    
    # Add some noise
    np.random.seed(42)
    x_noisy = x + 0.1 * np.random.randn(N)
    
    # Compute FFT of the noisy signal
    X = np.fft.fft(x_noisy)
    
    # Generate frequency axis
    freq = np.fft.fftfreq(N, d=t[1]-t[0])
    
    # Create filtered versions with different numbers of frequency components
    X_filtered_5 = X.copy()
    X_filtered_10 = X.copy()
    X_filtered_20 = X.copy()
    
    # Keep only the top 5, 10, 20 frequency components (and their conjugates)
    mag_spectrum = np.abs(X)
    sorted_indices = np.argsort(mag_spectrum)[::-1]  # Sort in descending order
    
    # Set all other components to zero
    mask_5 = np.ones(N, dtype=bool)
    mask_5[sorted_indices[10:]] = False  # Keep top 5 components (and their conjugates)
    X_filtered_5[mask_5] = 0
    
    mask_10 = np.ones(N, dtype=bool)
    mask_10[sorted_indices[20:]] = False  # Keep top 10 components (and their conjugates)
    X_filtered_10[mask_10] = 0
    
    mask_20 = np.ones(N, dtype=bool)
    mask_20[sorted_indices[40:]] = False  # Keep top 20 components (and their conjugates)
    X_filtered_20[mask_20] = 0
    
    # Reconstruct signals using IFFT
    x_reconstructed = np.fft.ifft(X)
    x_filtered_5 = np.fft.ifft(X_filtered_5)
    x_filtered_10 = np.fft.ifft(X_filtered_10)
    x_filtered_20 = np.fft.ifft(X_filtered_20)
    
    # Plot the results
    plt.figure(figsize=(14, 10))
    
    # Original and noisy signals
    plt.subplot(3, 2, 1)
    plt.plot(t[:100], x[:100], label='Original')
    plt.plot(t[:100], x_noisy[:100], alpha=0.7, label='Noisy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Original vs Noisy Signal (first 100 samples)')
    
    # Magnitude spectrum
    plt.subplot(3, 2, 2)
    plt.plot(freq[freq >= 0], np.abs(X)[freq >= 0])
    plt.grid(True, alpha=0.3)
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    
    # Reconstruction with all components
    plt.subplot(3, 2, 3)
    plt.plot(t[:100], x_noisy[:100], label='Noisy')
    plt.plot(t[:100], np.real(x_reconstructed[:100]), label='Full Reconstruction')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Full FFT Reconstruction')
    
    # Reconstruction with top 5 components
    plt.subplot(3, 2, 4)
    plt.plot(t[:100], x[:100], label='Original')
    plt.plot(t[:100], np.real(x_filtered_5[:100]), label='5 Components')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Reconstruction with Top 5 Components')
    
    # Reconstruction with top 10 components
    plt.subplot(3, 2, 5)
    plt.plot(t[:100], x[:100], label='Original')
    plt.plot(t[:100], np.real(x_filtered_10[:100]), label='10 Components')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Reconstruction with Top 10 Components')
    
    # Reconstruction with top 20 components
    plt.subplot(3, 2, 6)
    plt.plot(t[:100], x[:100], label='Original')
    plt.plot(t[:100], np.real(x_filtered_20[:100]), label='20 Components')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Reconstruction with Top 20 Components')
    
    plt.tight_layout()
    plt.savefig('signal_reconstruction.png')
    plt.show()
    
    # Calculate and print reconstruction errors
    error_full = np.mean(np.abs(x - np.real(x_reconstructed))**2)
    error_5 = np.mean(np.abs(x - np.real(x_filtered_5))**2)
    error_10 = np.mean(np.abs(x - np.real(x_filtered_10))**2)
    error_20 = np.mean(np.abs(x - np.real(x_filtered_20))**2)
    
    print("Reconstruction Mean Squared Errors:")
    print(f"Full reconstruction: {error_full:.6e}")
    print(f"Top 5 components: {error_5:.6e}")
    print(f"Top 10 components: {error_10:.6e}")
    print(f"Top 20 components: {error_20:.6e}")

# Example usage
if __name__ == "__main__":
    print("Demonstrating Fast Fourier Transform (FFT) Implementation")
    
    # Compare FFT implementations
    print("\nComparing FFT implementations:")
    compare_fft_implementations(signal_length=512)
    
    # Visualize time complexity
    print("\nVisualizing FFT time complexity:")
    visualize_fft_complexity()
    
    # Demonstrate signal reconstruction
    print("\nDemonstrating signal reconstruction from FFT:")
    reconstruct_signal_from_fft()
