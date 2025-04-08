#!/usr/bin/env python3
"""
Simple Fourier Series Demonstration
----------------------------------
This script demonstrates Fourier series approximation with a fixed number of terms,
generating four key visualizations:
1. Series plot with terms set
2. PSD with only terms set
3. Convergence and error up to the terms set
4. Animation showing only terms set

PHYS 4840 - Mathematical and Computational Methods II
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fourier_series as fs


def do_fourier(TERMS, wave):

    # Create x values for plotting
    x_range = (-2*np.pi, 2*np.pi)
    num_points = 10000
    x = np.linspace(x_range[0], x_range[1], num_points, endpoint=False)
    y_exact = wave(x)
    


    # Compute Fourier series coefficients
    a0, an, bn = fs.compute_coefficients(wave, TERMS)
    
    # Calculate the Fourier approximation
    y_approx = fs.fourier_series_approximation(x, a0, an, bn)
    
    # Calculate partial approximations
    partial_approx = fs.compute_partial_approximations(x, a0, an, bn)
    




    # 1. Plot the series with TERMS set
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_exact, 'k-', label='Exact')
    plt.plot(x, y_approx, 'r-', label=f'Fourier ({TERMS} terms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()#plt.savefig('fourier_approximation.png')
    plt.close()
    




    # 2. Plot the power spectral density (PSD) / coefficient spectrum
    plt.figure(figsize=(10, 6))
    
    # Compute magnitude of coefficients
    n_values = np.arange(1, TERMS + 1)
    
    # Plot coefficient magnitudes
    plt.stem(n_values, an, 'g-', markerfmt='g^', label='an', basefmt=" ", linefmt='g--')
    plt.stem(n_values, bn, 'r-', markerfmt='rs', label='bn', basefmt=" ", linefmt='r--')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('Harmonic (n)')
    plt.ylabel('Coefficient Magnitude')
    plt.yscale('log')
    plt.show()#plt.savefig('coefficient_spectrum.png')
    plt.close()
    




    # 3. Convergence and error analysis
    # Calculate error for each partial approximation
    errors = []
    term_counts = range(1, TERMS + 1)
    
    for i, approx in enumerate(partial_approx):
        error = np.sqrt(np.mean((y_exact - approx)**2))
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(term_counts, errors, 'bo-')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Terms')
    plt.ylabel('RMS Error')
    plt.show()#plt.savefig('convergence_rate_linear.png')
    plt.close()
    
    # Log-log plot to better visualize error scaling
    plt.figure(figsize=(10, 6))
    plt.loglog(term_counts, errors, 'bo-')
    plt.grid(True, alpha=0.3, which='both')
    plt.xlabel('Number of Terms')
    plt.ylabel('RMS Error')
    plt.show()#plt.savefig('convergence_rate_log.png')
    plt.close()
    




    # 4. Create an animation showing how the approximation improves with terms
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exact_line, = ax.plot(x, y_exact, 'k-', label='Exact')
    approx_line, = ax.plot([], [], 'r-', label='Fourier Approximation')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set axis limits
    margin = 0.1
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(x_range)
    
    # Text to display current number of terms
    terms_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        """Initialize animation"""
        approx_line.set_data([], [])
        terms_text.set_text('')
        return approx_line, terms_text
    
    def update(frame):
        """Update animation for each frame"""
        n_terms = frame + 1
        # Use pre-computed partial approximation
        y_approx = partial_approx[n_terms - 1]
        
        approx_line.set_data(x, y_approx)
        terms_text.set_text(f'Terms: {n_terms}')
        return approx_line, terms_text
    
    ani = FuncAnimation(fig, update, frames=TERMS,
                       init_func=init, blit=True, interval=200)
    
    plt.show()
    ani.save('fourier_animation.gif', writer='pillow', fps=5)
    plt.close()
    


def square_wave(x):
    """Square wave: 1 for 0 <= x < pi, -1 for pi <= x < 2pi"""
    return np.where((x % (2*np.pi)) < np.pi, 1.0, -1.0)


def sawtooth_wave(x):
    """Sawtooth wave: from -1 to 1 over 2pi period"""
    return (x % (2*np.pi)) / np.pi - 1


def triangle_wave(x):
    """Triangle wave with period 2pi"""
    # Normalize to [0, 2pi]
    x_norm = x % (2*np.pi)
    # For 0 to pi, goes from 0 to 1
    # For pi to 2pi, goes from 1 to 0
    return np.where(x_norm < np.pi, 
                   x_norm / np.pi, 
                   2 - x_norm / np.pi)


def pulse_train(x):
    """Pulse train: 1 for small interval, 0 elsewhere"""
    x_norm = x % (2*np.pi)
    pulse_width = np.pi / 8  # Very narrow pulse
    return np.where(x_norm < pulse_width, 1.0, 0.0)


def half_rectified_sine(x):
    """Half-rectified sine wave: max(0, sin(x))"""
    return np.maximum(0, np.sin(x))


def ecg_like_signal(x):

    def r(val, variation=0.1):
        return val * np.random.uniform(1 - variation, 1 + variation)

    # Normalize x to [0, 2pi]
    x_norm = x % (2 * np.pi)
    # P-wave with randomized amplitude, center, and width
    p_wave = r(0.25, 0.2) * np.exp(-((x_norm - r(0.7 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
    # QRS complex: one positive peak and two negative deflections
    qrs1 = r(1.0, 0.2) * np.exp(-((x_norm - r(np.pi, 0.05))**2) / (r(0.05 * np.pi, 0.05)**2))
    qrs2 = r(-0.3, 0.2) * np.exp(-((x_norm - r(0.9 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
    qrs3 = r(-0.2, 0.2) * np.exp(-((x_norm - r(1.1 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
    # T-wave with random parameters
    t_wave = r(0.5, 0.2) * np.exp(-((x_norm - r(1.4 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
    
    return p_wave + qrs1 + qrs2 + qrs3 + t_wave

def my_signal(x):
    return np.sin(x) * np.cos(np.sin(10 * x))


def main():
    TERMS = 100
    wave = pulse_train
    do_fourier(TERMS, wave)



if __name__ == "__main__":
    main()