#!/usr/bin/env python3
"""
Compare Fourier Transform Implementations
PHYS 4840 - Minimal benchmarking
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import fourier_transform as ft

def compare_speeds():
    sizes = [2,4,6,8,16, 32, 64, 128, 256, 512, 1024]
    times_dft = []
    times_radix2 = []
    times_bluestein = []
    times_zeropad = []
    times_numpy = []

    for N in sizes:
        x = np.random.rand(N)

        # Naive DFT (only small N)
        if N <= 256:
            t0 = time.time()
            ft.dft(x)
            times_dft.append(time.time() - t0)
        else:
            times_dft.append(None)

        # FFT radix-2 (only powers of 2)
        try:
            t0 = time.time()
            ft.fft_radix2(x)
            times_radix2.append(time.time() - t0)
        except:
            times_radix2.append(None)

        # FFT Bluestein (works for any N)
        t0 = time.time()
        ft.fft_bluestein(x)
        times_bluestein.append(time.time() - t0)

        # FFT zero-padding (works for any N)
        t0 = time.time()
        ft.fft_zeropad(x)
        times_zeropad.append(time.time() - t0)

        # NumPy FFT
        t0 = time.time()
        np.fft.fft(x)
        times_numpy.append(time.time() - t0)

    # Plotting
    plt.figure()
    if any(t is not None for t in times_dft):
        plt.plot([s for s, t in zip(sizes, times_dft) if t], 
                 [t for t in times_dft if t], 'o-', label='DFT (naive)')

    plt.plot(sizes, times_radix2, 's-', label='FFT (radix-2)')
    plt.plot(sizes, times_bluestein, '^-', label='FFT (bluestein)')
    plt.plot(sizes, times_zeropad, 'v-', label='FFT (zeropad)')
    plt.plot(sizes, times_numpy, 'x-', label='FFT (NumPy)')

    # Add theory lines
    x = np.array(sizes)
    plt.plot(x, 1e-6 * x**2, '--', label=r'$\mathcal{O}(N^2)$ reference')
    plt.plot(x, 1e-7 * x * np.log2(x), ':', label=r'$\mathcal{O}(N \log N)$ reference')

    plt.xlabel("Signal Size N")
    plt.ylabel("Time (s)")
    plt.title("Fourier Transform Speed Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_speeds()
