# Week 11 - Fourier Analysis & Signal Processing
**PHYS 4840 - Math and Computational Methods II - Spring 2025**
**Sections: 7.1, 7.2, 7.3, 7.4 -- COMPUTATIONAL PHYSICS - Mark Newman**
**MSMA 11.5 - Modern Statistical Methods for Astronomy**

## Introduction to Fourier Analysis

Fourier analysis is a mathematical technique that decomposes functions into oscillatory components (sines and cosines). This powerful approach allows us to transform signals between the time/space domain and the frequency domain, revealing insights that may not be apparent in the original representation.

### Why Fourier Analysis is Important in Physics

In physics, many systems exhibit oscillatory or wave-like behavior. Fourier analysis provides us with the tools to:

1. Decompose complex signals into their fundamental frequency components
2. Identify dominant frequencies in experimental data
3. Simplify differential equations by transforming them to algebraic equations
4. Analyze the frequency response of physical systems
5. Extract periodic signals from noisy data

For computational physicists, Fourier techniques are essential because they:
- Provide efficient algorithms for solving certain types of problems
- Allow for filtering and processing of data in ways not possible in the original domain
- Enable compression of data by focusing on the most significant frequency components

## Mathematical Foundations

### Fourier Series: Breaking Down Periodic Functions Step-by-Step

Let's start with a basic question: How can we represent any periodic function using simple building blocks? This is the central idea of Fourier series.

#### The Basic Building Blocks: Sines and Cosines

Think of sines and cosines as the fundamental "shapes" of oscillation. These functions repeat forever and have several important properties:
- They are periodic (they repeat at regular intervals)
- They are smooth (continuously differentiable)
- They have well-defined frequencies (how fast they oscillate)

Here are some examples of these building blocks:
- $\cos(x)$: Oscillates once over $2\pi$ interval
- $\sin(2x)$: Oscillates twice over $2\pi$ interval 
- $\cos(3x)$: Oscillates three times over $2\pi$ interval

#### Combining the Building Blocks

The central insight of Fourier analysis is that we can add together sines and cosines of different frequencies to create almost any shape we want. For a function with period $2\pi$, the Fourier series is:

$f(x) = \frac{a_0}{2} + a_1\cos(x) + b_1\sin(x) + a_2\cos(2x) + b_2\sin(2x) + a_3\cos(3x) + b_3\sin(3x) + \ldots$

Or more compactly:

$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$

#### Finding the Coefficients

How do we find the values of $a_0$, $a_1$, $b_1$, $a_2$, $b_2$, etc.? The beauty of sines and cosines is that they are orthogonal to each other, meaning different frequency components don't interfere with each other. This gives us formulas for the coefficients:

$a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \, dx$

$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$

$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$

#### What These Coefficients Mean:

- $a_0/2$: This is simply the average value of the function over one period
- $a_1$: Strength of the "once-per-period" cosine oscillation
- $b_1$: Strength of the "once-per-period" sine oscillation
- $a_2$ and $b_2$: Strength of the "twice-per-period" oscillations
- Higher $n$ values: Represent faster and faster oscillations

#### Visualizing the Process

Let's imagine we're approximating a square wave:
1. First, we include just $a_0/2$ (the average value)
2. Then we add $a_1\cos(x) + b_1\sin(x)$ (the fundamental frequency)
3. Then we add $a_2\cos(2x) + b_2\sin(2x)$ (the second harmonic)
4. And so on...

With each term we add, our approximation gets closer to the square wave. This is like adding finer and finer details to a picture.

#### Alternative: Amplitude and Phase Form

Instead of using separate sine and cosine terms, we can combine them into a single sinusoid with an amplitude and phase:

$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} A_n \cos(nx - \phi_n)$

Where:
- $A_n = \sqrt{a_n^2 + b_n^2}$ is the amplitude of frequency $n$
- $\phi_n = \arctan(b_n/a_n)$ is the phase shift of frequency $n$

This form makes it easier to visualize each frequency component's strength and timing.

### Fourier Transform: What Happens When a Signal Isn't Periodic?

#### Moving from Periodic to Non-Periodic Functions

What if our signal doesn't repeat? For example, a single pulse, or a damped oscillation that eventually dies out? For these non-periodic functions, we need the Fourier transform.

#### Conceptual Development

Think about what happens when a function's period gets longer and longer:
1. As the period increases, the discrete frequency components get closer together
2. In the limit of infinite period (non-periodic function), these discrete frequencies become continuous
3. Instead of summing over discrete frequencies, we need to integrate over a continuous range

#### The Fourier Transform Using Trigonometric Functions

For a non-periodic function, we can express the Fourier transform in terms of sines and cosines:

$F_c(k) = \int_{-\infty}^{\infty} f(x) \cos(2\pi k x) \, dx$
$F_s(k) = \int_{-\infty}^{\infty} f(x) \sin(2\pi k x) \, dx$

Where:
- $F_c(k)$ is the cosine component at frequency $k$
- $F_s(k)$ is the sine component at frequency $k$

The complete Fourier transform combines these:
$F(k) = F_c(k) - i F_s(k)$

And the inverse transform to recover the original function:
$f(x) = \int_{0}^{\infty} [F_c(k)\cos(2\pi k x) + F_s(k)\sin(2\pi k x)] \, dk$

#### Understanding the Fourier Transform Intuitively

The Fourier transform essentially measures "how much" of each frequency is present in a signal:

1. For each possible frequency $k$:
   - Multiply the signal by sine and cosine waves of that frequency
   - Integrate the result over all time
   - Large positive or negative values indicate strong presence of that frequency
   
2. The resulting function $F(k)$ tells us:
   - Which frequencies are present in the original signal
   - How strong each frequency component is
   - What phase each frequency component has

#### Key Differences from Fourier Series:
- Analyzes non-repeating signals (potentially infinite in length)
- Produces a continuous spectrum instead of discrete frequency spikes
- Each infinitesimal frequency range $dk$ contributes to the signal

#### Real-World Interpretation:
- The magnitude $\sqrt{F_c(k)^2 + F_s(k)^2}$ tells you the strength of frequency $k$
- The relative sizes of $F_c(k)$ and $F_s(k)$ tell you the phase
- The Fourier transform preserves the total energy of the signal

### Discrete Fourier Transform (DFT): The Real-World Version for Computers

#### Why We Need the Discrete Version

In real-world physics problems, we don't have continuous functions. Instead, we have:
- Measurements taken at specific time points
- A finite number of data points
- Digital data stored in computers

This is where the Discrete Fourier Transform (DFT) comes in. It's the practical version of Fourier analysis for actual computational work.

#### Understanding the DFT Step by Step

Let's say we have a signal measured at N equally spaced points: $\{x_0, x_1, ..., x_{N-1}\}$

The DFT transforms this into a set of N frequency components: $\{X_0, X_1, ..., X_{N-1}\}$

Each frequency component $X_k$ tells us how much of frequency $k$ exists in our original signal.

#### The DFT Using Sines and Cosines

The DFT can be written in terms of sines and cosines as follows:

$X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{2\pi k n}{N}\right) - i \sum_{n=0}^{N-1} x_n \sin\left(\frac{2\pi k n}{N}\right)$

This gives us:
- The real part of $X_k$: Correlation with a cosine of frequency $k$
- The imaginary part of $X_k$: Correlation with a sine of frequency $k$

#### Going Back: The Inverse DFT

To recover our original signal from the frequency components, we use:

$x_n = \frac{1}{N} \sum_{k=0}^{N-1} \left[ \text{Re}(X_k) \cos\left(\frac{2\pi k n}{N}\right) + \text{Im}(X_k) \sin\left(\frac{2\pi k n}{N}\right) \right]$

Where $\text{Re}(X_k)$ is the real part and $\text{Im}(X_k)$ is the imaginary part of $X_k$.

#### What the DFT Frequencies Actually Mean

For a signal sampled at rate $f_s$ (samples per second):
- $X_0$ represents the DC component (average value)
- $X_1$ through $X_{N/2-1}$ represent positive frequencies from $f_s/N$ to $f_s/2 - f_s/N$
- $X_{N/2}$ represents the Nyquist frequency ($f_s/2$)
- $X_{N/2+1}$ through $X_{N-1}$ represent negative frequencies

#### Example with Numbers:
If we sample at 1000 Hz for 1 second (N=1000):
- $X_0$ is the average value
- $X_1$ represents 1 Hz
- $X_{10}$ represents 10 Hz
- $X_{500}$ represents 500 Hz (Nyquist frequency)

#### Key Properties of the DFT:

1. **Finite Resolution**: The DFT can only resolve frequencies that are multiples of $f_s/N$
2. **Periodicity**: The DFT implicitly assumes the signal repeats every N samples
3. **Symmetry**: For real input signals, the frequency components have symmetry: $X_{N-k} = X_k^*$
4. **Linearity**: The DFT of a sum equals the sum of the DFTs
5. **Energy Conservation**: The total energy in the time domain equals the total energy in the frequency domain (Parseval's theorem)

#### Computational Considerations:
- Direct calculation requires $N^2$ operations (slow for large N)
- The Fast Fourier Transform (FFT) algorithm reduces this to $N \log N$ operations
- The DFT frequencies are evenly spaced from 0 to $f_s$, with resolution $f_s/N$

## Fast Fourier Transform (FFT): The Breakthrough Algorithm That Made Fourier Analysis Practical

### Why We Need a Faster Algorithm

Computing the DFT directly is very slow for large datasets:
- For N = 1,000 points: ~1,000,000 operations (direct DFT)
- For N = 1,000,000 points: ~1,000,000,000,000 operations (direct DFT)

This made Fourier analysis impractical for many real-world applications until the development of the Fast Fourier Transform (FFT) algorithm.

### The FFT Breakthrough

The FFT algorithm reduces computation from $O(N^2)$ to $O(N \log N)$:
- For N = 1,000 points: ~10,000 operations (FFT)
- For N = 1,000,000 points: ~20,000,000 operations (FFT)

This is a reduction from trillion to million operations - making previously impossible calculations very feasible!

### How the FFT Works: A Simple Explanation

The key insight of the FFT is that we can break down a big calculation into smaller pieces that reuse previous work. Here's the general idea:

1. **Divide the problem**: Split the N-point DFT into two N/2-point DFTs
   - One DFT for even-indexed points: $x_0, x_2, x_4, ...$
   - One DFT for odd-indexed points: $x_1, x_3, x_5, ...$

2. **Reuse calculations**: The smaller DFTs have patterns we can exploit
   - Many calculations repeat in specific patterns
   - We can store and reuse these intermediate results

3. **Combine results**: Merge the two N/2-point DFTs to get the full N-point DFT
   - This step uses the "butterfly" pattern (a simple sum and difference operation)

### The Butterfly: The Core of the FFT

The term "butterfly" comes from the shape of the diagram used to represent the basic operation:

```
x[a] ────────┬──────── X[a]
             │
             │ W_N^k
             │
x[b] ────────┴──╳──── X[b]
```

In this diagram:
- We combine two input values to create two output values
- Each butterfly operation requires just one complex multiplication and two additions
- This is far more efficient than recomputing everything from scratch

### The Decimation-in-Time (DIT) Approach

Using the butterfly concept, the full FFT calculation works like this:

1. Separate the input into even and odd indices
2. Compute the N/2-point DFT of each group (recursively applying the same technique)
3. Combine the results using butterfly operations

The formula for this is:

$X_k = \text{DFT of even elements} + W_N^k × \text{DFT of odd elements}$

Where $W_N^k = e^{-2\pi i k / N}$ is called a "twiddle factor" - essentially a rotation in the complex plane.

For simplicity, using $E_k$ for the DFT of even elements and $O_k$ for the DFT of odd elements:

$X_k = E_k + W_N^k O_k$

### Why Power-of-2 Sizes Are Efficient

The FFT works best when N is a power of 2 because:
- We can divide the problem exactly in half at each step
- This division can continue until we reach trivial 1-point DFTs
- The butterfly pattern works perfectly with these sizes

This is why you'll often see FFT implementations require or recommend sizes like 1024, 2048, or 4096.

### Implementation Example:

```python
def fft_recursive(x):
    """Basic recursive FFT implementation for signals with length = power of 2"""
    N = len(x)
    
    # Base case
    if N == 1:
        return x
    
    # Divide
    even = fft_recursive(x[0::2])  # Even-indexed elements
    odd = fft_recursive(x[1::2])   # Odd-indexed elements
    
    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    result = np.zeros(N, dtype=complex)
    half_N = N // 2
    
    for k in range(half_N):
        result[k] = even[k] + factor[k] * odd[k]
        result[k + half_N] = even[k] - factor[k] * odd[k]
    
    return result
```

### FFT Performance Considerations:

- Most efficient when N is a power of 2
- Memory access patterns can significantly affect performance
- Modern FFT libraries (like FFTW or NumPy's implementation) use various optimizations
- Can be parallelized for further performance improvements

## Critical Sampling and Frequency Considerations: Why Sampling Rate Matters

### The Nyquist-Shannon Sampling Theorem: A Fundamental Limit

The Nyquist-Shannon sampling theorem is one of the most important principles in digital signal processing. In simple terms:

**To properly capture a signal, you must sample at least twice as fast as the highest frequency present in the signal.**

#### A Simple Way to Understand This:

Imagine a sine wave oscillating at frequency $f$. To properly capture one cycle of this wave, we need at least two points:
- One point to catch the peak
- One point to catch the trough

If we sample any slower, we might miss the peaks and troughs entirely, leading to a completely wrong representation of the signal.

#### The Mathematical Statement:

For a signal with maximum frequency $f_{max}$, the sampling rate $f_s$ must satisfy:

$f_s > 2f_{max}$

The maximum frequency we can accurately represent with a given sampling rate is called the **Nyquist frequency**:

$f_{Nyquist} = \frac{f_s}{2}$

#### An Example with Numbers:

- If we sample at 1000 Hz (1000 samples per second):
  - We can accurately capture frequencies up to 500 Hz
  - Any frequency above 500 Hz will not be represented correctly

- If our signal contains a 700 Hz component:
  - We cannot capture it correctly with a 1000 Hz sampling rate
  - It will appear as a 300 Hz component in our sampled data (aliasing)

#### Why Aliasing Happens: The Folding Effect

Aliasing occurs when frequencies above the Nyquist frequency "fold back" into the lower frequency range:

1. A frequency $f$ above the Nyquist frequency will appear as $|f - f_s|$ if $f_s < f < 2f_s$
2. This folding pattern continues for higher multiples of $f_s$

For example, with $f_s = 1000$ Hz:
- A 700 Hz component appears as 300 Hz (1000 - 700 = 300)
- A 1200 Hz component also appears as 200 Hz (1200 - 1000 = 200)
- A 1800 Hz component appears as 200 Hz (2000 - 1800 = 200)

#### Real-World Implications:

1. **Audio Processing**: CD quality audio uses 44.1 kHz sampling to capture frequencies up to 22.05 kHz (just above human hearing limit)

2. **Image Processing**: Digital cameras need enough pixels (spatial sampling) to resolve the finest details

3. **Scientific Measurements**: When measuring physical phenomena, ensure your sampling rate exceeds twice the highest frequency you expect to observe

4. **Anti-Aliasing Filters**: In practice, we apply low-pass filters before sampling to remove frequencies above the Nyquist limit

### Frequency Resolution and Spectral Leakage: The Practical Challenges

#### Frequency Resolution: How Closely We Can Distinguish Different Frequencies

Frequency resolution refers to our ability to tell apart two close frequencies in our analysis. This is determined by how long we observe the signal.

##### A Simple Example:
- To distinguish between 100 Hz and 101 Hz signals, we need to observe for at least 1 second
- To distinguish between 100 Hz and 100.1 Hz, we need to observe for at least 10 seconds

##### The Mathematical Formula:
The minimum frequency difference we can resolve is:

$\Delta f = \frac{f_s}{N} = \frac{1}{T}$

Where:
- $f_s$ is the sampling rate
- $N$ is the number of samples
- $T$ is the total observation time

##### Real-World Implications:
1. **Short recordings have poor frequency resolution**:
   - A 0.1-second recording can only resolve frequencies that differ by at least 10 Hz
   - This might be insufficient for many applications (music, speech, etc.)

2. **Long recordings improve resolution**:
   - A 10-second recording can resolve frequencies that differ by 0.1 Hz
   - Astronomical observations might run for hours or days to detect tiny frequency variations

#### Spectral Leakage: When Frequencies Don't Line Up with Our Analysis Bins

Spectral leakage is a phenomenon where energy from one frequency "spills over" into neighboring frequency bins.

##### Why It Happens:
The DFT assumes our signal repeats perfectly over the observation window. If a frequency doesn't complete a whole number of cycles in our window, the apparent discontinuity at the edges causes leakage.

##### Visual Explanation:
1. **Ideal case**: A 10 Hz signal observed for exactly 0.1 seconds completes exactly one cycle
   - Result: All energy is concentrated at the 10 Hz bin

2. **Leakage case**: A 10.5 Hz signal observed for exactly 0.1 seconds completes 1.05 cycles
   - Result: Energy spreads across multiple frequency bins
   - The true frequency peak becomes blurred
   - Additional false peaks appear

##### Consequences of Spectral Leakage:
1. **Reduced peak amplitude**: The true signal strength is underestimated
2. **Decreased frequency precision**: The exact frequency location becomes uncertain
3. **Masking of weaker signals**: Leakage from strong signals can hide nearby weak signals

##### Solutions for Spectral Leakage:
1. **Ensure coherent sampling**: Try to capture whole numbers of cycles (not always possible)
2. **Apply windowing functions**: Gradually taper the signal at the edges (we'll discuss this next)
3. **Use longer observation windows**: Reduces the percentage impact of edge effects
4. **Zero-padding**: Add zeros to increase the apparent frequency resolution (but doesn't actually improve fundamental resolution)

## Windowing Functions: Reducing Spectral Leakage

Windowing functions multiply the time-domain signal before performing the Fourier transform to reduce the discontinuities at the boundaries.

### Popular Windowing Functions:

| Window Type | Main Lobe Width | Side Lobe Attenuation | Trade-offs |
|-------------|-----------------|------------------------|----------|
| Rectangular | Narrowest | Poorest (-13 dB) | Best frequency resolution, worst spectral leakage |
| Hamming | Medium | Better (-43 dB) | Good general-purpose window |
| Hann | Medium | Good (-32 dB) | Reduces side lobes at cost of main lobe width |
| Blackman | Widest | Best (-58 dB) | Excellent side lobe suppression, worst frequency resolution |

### Implementation Example:

```python
def apply_window(signal, window_type='hann'):
    """Apply a windowing function to a signal"""
    N = len(signal)
    
    if window_type == 'rectangular':
        window = np.ones(N)
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    elif window_type == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    elif window_type == 'blackman':
        window = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)) + 
                 0.08 * np.cos(4 * np.pi * np.arange(N) / (N - 1)))
    
    return signal * window
```

### Window Selection Guidelines:

- Choose **rectangular** when analyzing transients or measuring exact amplitudes in high SNR signals
- Choose **Hamming** for general-purpose spectral analysis
- Choose **Hann** when the signal contains closely spaced frequencies of similar amplitudes
- Choose **Blackman** when looking for weak spectral features near strong ones

## Zero-Padding and Interpolation

Zero-padding involves appending zeros to the time-domain signal before computing the DFT.

### Benefits of Zero-Padding:
1. Increases the number of frequency points in the DFT output
2. Provides smoother appearance of the spectrum (interpolation)
3. Can help identify peaks that fall between frequency bins

### Limitations of Zero-Padding:
1. Does NOT improve the fundamental frequency resolution (determined by signal duration)
2. Cannot reveal frequency components that aren't present in the original data
3. Introduces computational overhead

### Implementation Example:

```python
def zero_pad_fft(signal, pad_factor=2):
    """Compute FFT with zero padding for smoother spectrum"""
    N = len(signal)
    padded_signal = np.zeros(N * pad_factor)
    padded_signal[:N] = signal
    return np.fft.fft(padded_signal)
```

## Applications of Fourier Analysis in Physics

### 1. Signal Processing
- **Filtering**: Remove noise or unwanted frequency components
- **Deconvolution**: Extract the original signal from a convolved or blurred version
- **Correlation Analysis**: Find patterns or similarities in signals

### 2. Wave Phenomena
- **Wave Propagation**: Analyze and predict wave behavior
- **Optics**: Study diffraction and interference patterns
- **Acoustic Analysis**: Identify resonant frequencies and harmonics

### 3. Astronomy
- **Periodicity Detection**: Find periodic sources like pulsars or variable stars
- **Redshift Analysis**: Analyze spectral lines from distant galaxies
- **Time Series Analysis**: Process data from telescopes and other instruments

### 4. Quantum Mechanics
- **Momentum Space Representations**: Transform between position and momentum space
- **Energy Level Analysis**: Study energy spectra of quantum systems
- **Quantum Computing Algorithms**: Some quantum algorithms rely on Fourier transforms

### 5. Computational Fluid Dynamics
- **Spectral Methods**: Solve PDEs using Fourier basis functions
- **Turbulence Analysis**: Study energy cascades across different scales
- **Flow Visualization**: Identify dominant flow structures

## Practical Implementation in Python

### Basic FFT Example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a test signal (sum of sine waves + noise)
fs = 1000  # Sampling rate (Hz)
T = 1      # Duration (seconds)
t = np.linspace(0, T, fs*T, endpoint=False)  # Time vector

# Signal with 50 Hz and 120 Hz components
signal = (1.0 * np.sin(2 * np.pi * 50 * t) + 
          0.5 * np.sin(2 * np.pi * 120 * t) + 
          0.2 * np.random.randn(len(t)))  # Add noise

# Compute the FFT
N = len(signal)
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, 1/fs)  # Frequency bins

# Plot only the positive frequencies (since signal is real)
positive_freq_mask = frequencies >= 0
plt.figure(figsize=(10, 6))
plt.plot(frequencies[positive_freq_mask], np.abs(fft_result[positive_freq_mask])/N)
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.xlim(0, fs/2)  # Display up to Nyquist frequency
```

### FFT-Based Filtering:

```python
def bandpass_filter(signal, fs, lowcut, highcut):
    """Apply a frequency domain bandpass filter to a signal"""
    # Compute FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/fs)
    
    # Create a mask for the bandpass region
    mask = (abs(frequencies) >= lowcut) & (abs(frequencies) <= highcut)
    
    # Apply the filter
    filtered_fft = fft_result * mask
    
    # Inverse FFT to return to time domain
    return np.real(np.fft.ifft(filtered_fft))
```

## Error Considerations in Fourier Analysis

### 1. Aliasing

Aliasing occurs when a signal contains frequencies above the Nyquist frequency, causing those components to appear as lower frequencies in the sampled signal.

#### Mitigation:
- **Anti-aliasing filters**: Apply a low-pass filter before digitization
- **Oversampling**: Sample at a higher rate than strictly required
- **Analytical verification**: Check if results make physical sense

### 2. Spectral Leakage

Spectral leakage occurs when the signal is not periodic within the sampling window, causing energy to "leak" into adjacent frequency bins.

#### Mitigation:
- **Windowing**: Apply tapered windows to reduce discontinuities
- **Appropriate record length**: Try to capture an integer number of periods
- **Zero-padding**: Increase apparent frequency resolution for peak identification

### 3. Picket Fence Effect

The picket fence effect refers to the limited frequency resolution in the DFT, where the true peak of a frequency component may fall between discrete frequency bins.

#### Mitigation:
- **Zero-padding**: Add zeros to increase the number of frequency points
- **Interpolation**: Estimate the true peak location using interpolation
- **Longer data record**: Increase the fundamental frequency resolution

## Computing Considerations

### Performance Optimization:
1. **Use power-of-2 lengths** for most efficient FFT computation
2. **Pre-allocate arrays** to avoid memory reallocation during computation
3. **Consider batch processing** for multiple transforms of the same size
4. **Use in-place FFT algorithms** for large datasets to reduce memory usage

### Numerical Precision:
1. **Normalize properly** to minimize floating-point errors
2. **Consider using double precision** for sensitive applications
3. **Monitor for accumulation of roundoff errors** in iterative algorithms

## Files in this Repository

- `fourier_series.py`: Implementation and visualization of Fourier series
- `dft_implementation.py`: Direct implementation of the Discrete Fourier Transform
- `fft_implementation.py`: Fast Fourier Transform implementation and analysis
- `signal_processing_examples.py`: Real-world applications of Fourier analysis

Each file includes detailed comments and examples to help you understand the implementation details.

## References

1. Newman, M. (2013). Computational Physics. CreateSpace Independent Publishing Platform.
2. Feigelson, E. D., & Babu, G. J. (2012). Modern Statistical Methods for Astronomy. Cambridge University Press.