#!/usr/bin/python3
#########################
# Fourier Series Implementation and Visualization
# PHYS 4840 - Math and Computational Methods II
# Week 11 - Fourier Analysis: Theory & Discrete FT
#########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def fourier_series_approximation(x, n_terms, func_type='square'):
    """
    Calculate Fourier series approximation for different function types.
    
    Parameters:
        x (array): Input values (typically in range -π to π)
        n_terms (int): Number of terms in the Fourier series
        func_type (str): 'square', 'sawtooth', or 'triangle'
        
    Returns:
        array: Fourier series approximation of the specified function
    """
    result = np.zeros_like(x, dtype=float)
    
    if func_type == 'square':
        # Square wave Fourier series (odd harmonics only)
        for n in range(1, n_terms + 1, 2):
            result += (4 / (n * np.pi)) * np.sin(n * x)
    
    elif func_type == 'sawtooth':
        # Sawtooth wave Fourier series
        for n in range(1, n_terms + 1):
            result += (2 / n) * (-1)**(n+1) * np.sin(n * x)
    
    elif func_type == 'triangle':
        # Triangle wave Fourier series (odd harmonics only)
        for n in range(1, n_terms + 1, 2):
            result += (8 / (n * np.pi)**2) * (-1)**((n-1)//2) * np.sin(n * x)
    
    return result

def true_function(x, func_type='square'):
    """
    Calculate the true function values for comparison.
    
    Parameters:
        x (array): Input values
        func_type (str): 'square', 'sawtooth', or 'triangle'
        
    Returns:
        array: True function values
    """
    if func_type == 'square':
        return np.sign(np.sin(x))
    
    elif func_type == 'sawtooth':
        return (x % (2 * np.pi)) / np.pi - 1
    
    elif func_type == 'triangle':
        return 2 * np.abs((x % (2 * np.pi)) / np.pi - 1) - 1

def plot_fourier_series(func_type='square', max_terms=10):
    """
    Plot the Fourier series approximation with increasing number of terms.
    
    Parameters:
        func_type (str): 'square', 'sawtooth', or 'triangle'
        max_terms (int): Maximum number of terms to include
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    true_func = true_function(x, func_type)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, true_func, 'k--', label='True Function')
    
    colors = plt.cm.viridis(np.linspace(0, 1, max_terms))
    
    for i, n_terms in enumerate([1, 3, 5, 10, 20, 50]):
        if n_terms <= max_terms:
            approx = fourier_series_approximation(x, n_terms, func_type)
            plt.plot(x, approx, label=f'N = {n_terms}', color=colors[min(n_terms, max_terms-1)])
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Fourier Series Approximation of {func_type.capitalize()} Wave')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim(-2 * np.pi, 2 * np.pi)
    
    plt.savefig(f'{func_type}_fourier_series.png')
    plt.show()

def animate_fourier_series(func_type='square', max_terms=50):
    """
    Create an animation showing the Fourier series convergence.
    
    Parameters:
        func_type (str): 'square', 'sawtooth', or 'triangle'
        max_terms (int): Maximum number of terms to include
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    true_func = true_function(x, func_type)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, true_func, 'k--', label='True Function')
    line, = ax.plot([], [], 'r-', label='Fourier Approximation')
    
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    y_range = 1.5 if func_type == 'square' else 1.2
    ax.set_ylim(-y_range, y_range)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Fourier Series Approximation of {func_type.capitalize()} Wave')
    
    term_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        term_text.set_text('')
        return line, term_text
    
    def update(frame):
        n_terms = frame + 1
        approx = fourier_series_approximation(x, n_terms, func_type)
        line.set_data(x, approx)
        term_text.set_text(f'Terms: {n_terms}')
        return line, term_text
    
    ani = FuncAnimation(fig, update, frames=max_terms,
                        init_func=init, blit=True, interval=100)
    
    plt.legend()
    plt.tight_layout()
    
    # Save animation (optional)
    # ani.save(f'{func_type}_fourier_animation.mp4', writer='ffmpeg', fps=10)
    
    plt.show()

def gibbs_phenomenon_demo():
    """
    Demonstrate the Gibbs phenomenon at discontinuities.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 10000)  # Use a higher resolution
    func_type = 'square'  # Square wave has clear discontinuities
    
    plt.figure(figsize=(12, 8))
    
    # Focus on a transition region
    x_zoomed = np.linspace(-0.5, 0.5, 1000)
    true_zoomed = true_function(x_zoomed, func_type)
    
    for n_terms in [5, 10, 50, 100, 500]:
        approx = fourier_series_approximation(x_zoomed, n_terms, func_type)
        plt.plot(x_zoomed, approx, label=f'N = {n_terms}')
    
    plt.plot(x_zoomed, true_zoomed, 'k--', linewidth=2, label='True Function')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Gibbs Phenomenon at Discontinuity')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    plt.savefig('gibbs_phenomenon.png')
    plt.show()

def fourier_energy_spectrum(func_type='square', max_n=100):
    """
    Plot the energy spectrum of Fourier coefficients.
    
    Parameters:
        func_type (str): 'square', 'sawtooth', or 'triangle'
        max_n (int): Maximum harmonic to compute
    """
    n_values = np.arange(1, max_n + 1)
    coefficients = np.zeros(max_n)
    
    if func_type == 'square':
        for i, n in enumerate(n_values):
            if n % 2 == 1:  # Odd harmonics only
                coefficients[i] = (4 / (n * np.pi))**2
            # Even harmonics are zero
            
    elif func_type == 'sawtooth':
        for i, n in enumerate(n_values):
            coefficients[i] = (2 / n)**2
            
    elif func_type == 'triangle':
        for i, n in enumerate(n_values):
            if n % 2 == 1:  # Odd harmonics only
                coefficients[i] = (8 / (n * np.pi)**2)**2
            # Even harmonics are zero
    
    plt.figure(figsize=(12, 6))
    
    # Plot on log-log scale to see power law
    plt.loglog(n_values, coefficients, 'o-')
    
    # Add a reference line showing the expected power law decay
    if func_type == 'square':
        power = -2  # Square wave coefficients decay as 1/n^2
    elif func_type == 'sawtooth':
        power = -2  # Sawtooth wave coefficients decay as 1/n^2
    elif func_type == 'triangle':
        power = -4  # Triangle wave coefficients decay as 1/n^4
        
    # Plot reference line
    x_ref = np.array([1, max_n])
    y_ref = coefficients[0] * (x_ref / x_ref[0])**power
    plt.loglog(x_ref, y_ref, 'k--', label=f'1/n^{-power}')
    
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.title(f'Energy Spectrum of {func_type.capitalize()} Wave')
    plt.xlabel('Harmonic Number (n)')
    plt.ylabel('|Coefficient|²')
    
    plt.savefig(f'{func_type}_energy_spectrum.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    print("Demonstrating Fourier Series Approximations")
    
    # Plot Fourier series approximations for different functions
    for func in ['square', 'sawtooth', 'triangle']:
        print(f"\nPlotting {func} wave approximation...")
        plot_fourier_series(func_type=func, max_terms=50)
    
    # Demonstrate Gibbs phenomenon
    print("\nDemonstrating Gibbs phenomenon...")
    gibbs_phenomenon_demo()
    
    # Plot energy spectrum
    print("\nPlotting energy spectrum...")
    for func in ['square', 'sawtooth', 'triangle']:
        fourier_energy_spectrum(func_type=func)
    
    # Create animations (commented out as they're interactive)
    # print("\nCreating animations...")
    # animate_fourier_series('square')
