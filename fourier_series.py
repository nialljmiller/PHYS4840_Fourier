#!/usr/bin/env python3
"""
Fourier Series Implementation
----------------------------
A clean, pedagogical implementation of Fourier series for teaching purposes.
This module provides functions to compute Fourier coefficients and series
for arbitrary functions.

PHYS 4840 - Mathematical and Computational Methods II
"""

import numpy as np



def compute_a0(func, period=2*np.pi, num_points=1000):
    """
    Compute the a0 Fourier coefficient (constant term).
    
    Parameters:
        func (callable): Function to approximate
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        float: a0 coefficient (divided by 2)
    """
    x = np.linspace(0, period, num_points)
    y = func(x)
    result = np.trapz(y, x)
    return (1 / period) * result


def compute_an(func, n, period=2*np.pi, num_points=1000):
    """
    Compute the an Fourier coefficient for cosine terms using NumPy's trapz.
    
    Parameters:
        func (callable): Function to approximate
        n (int): Harmonic number
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        float: an coefficient
    """
    x = np.linspace(0, period, num_points)
    y = func(x)
    
    # Create the integrand: f(x) * cos(2πnx/period)
    integrand = y * np.cos(2 * np.pi * n * x / period)
    
    # Use NumPy's trapz function
    result = np.trapz(integrand, x)
    
    # Scale by 2/period for the Fourier coefficient
    return (2/period) * result



def compute_bn(func, n, period=2*np.pi, num_points=1000):
    """
    Compute the bn Fourier coefficient for sine terms using NumPy's trapz.
    
    Parameters:
        func (callable): Function to approximate
        n (int): Harmonic number
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        float: bn coefficient
    """
    x = np.linspace(0, period, num_points)
    y = func(x)
    
    # Create the integrand: f(x) * sin(2πnx/period)
    integrand = y * np.sin(2 * np.pi * n * x / period)
    
    # Use NumPy's trapz function
    result = np.trapz(integrand, x)
    
    # Scale by 2/period for the Fourier coefficient
    return (2/period) * result



def compute_coefficients(func, n_terms, period=2*np.pi, num_points=1000):
    """
    Compute all Fourier coefficients up to a specified number of terms.
    
    Parameters:
        func (callable): Function to approximate
        n_terms (int): Number of terms in the Fourier series
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        tuple: (a0, an_coefficients, bn_coefficients)
    """
    a0 = compute_a0(func, period, num_points)
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)
    
    for n in range(1, n_terms + 1):
        an[n-1] = compute_an(func, n, period, num_points)
        bn[n-1] = compute_bn(func, n, period, num_points)
    
    return a0, an, bn


def fourier_series_approximation(x, a0, an, bn, period=2*np.pi):
    """
    Compute the Fourier series approximation using precomputed coefficients.
    
    Parameters:
        x (array): Points where to evaluate the approximation
        a0 (float): Constant coefficient (divided by 2)
        an (array): Cosine coefficients
        bn (array): Sine coefficients
        period (float): Period of the function
    
    Returns:
        array: Fourier series approximation at points x
    """
    result = np.ones_like(x) * a0
    
    for n in range(1, len(an) + 1):
        result += an[n-1] * np.cos(2 * np.pi * n * x / period)
        result += bn[n-1] * np.sin(2 * np.pi * n * x / period)
    
    return result


def compute_partial_approximations(x, a0, an, bn, period=2*np.pi):
    """
    Compute partial Fourier approximations with increasing number of terms.
    
    Parameters:
        x (array): Points where to evaluate the approximation
        a0 (float): Constant coefficient (divided by 2)
        an (array): Cosine coefficients
        bn (array): Sine coefficients
        period (float): Period of the function
    
    Returns:
        list: Approximations with increasing number of terms
    """
    approximations = []
    for k in range(1, len(an) + 1):
        approx = np.ones_like(x) * a0
        for n in range(k):
            approx += an[n] * np.cos(2 * np.pi * (n+1) * x / period)
            approx += bn[n] * np.sin(2 * np.pi * (n+1) * x / period)
        approximations.append(approx)
    
    return approximations