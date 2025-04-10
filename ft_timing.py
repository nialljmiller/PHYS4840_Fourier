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


if __name__ == "__main__":
    compare_speeds()
