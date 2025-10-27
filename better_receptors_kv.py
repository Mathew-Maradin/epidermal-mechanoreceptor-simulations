from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter, laplace

# Wave-Like Ridges
def generate_wave(shape=(256, 256), scale=50.0, ridge_freq=15.0) -> NDArray[np.float16]:
    height, width = shape
    noise_array = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            # Basic Perlin noise
            nx = x / scale
            ny = y / scale
            noise_val = pnoise2(nx, ny, octaves=4, persistence=0.5, lacunarity=2.0)
            
            # Add sine-based ridge modulation (like fingerprint lines)
            sine_mod = np.sin(2 * np.pi * ridge_freq * (x / width + 0.3 * np.sin(y / 30.0)))
            
            # Combine Perlin noise and sine ridge pattern
            noise_array[y, x] = noise_val + 0.5 * sine_mod

    # Normalize to 0–1
    noise_array = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min())
    return noise_array

# Spiral Ridges
def generate_spiral(shape=(256, 256), ridge_freq=25.0, noise_scale=40.0, noise_strength=0.5) -> NDArray[np.float16]:
    height, width = shape
    cx, cy = width // 2, height // 2

    spiral_pattern = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)

            # Spiral ridge pattern based on radius and angle
            spiral_ridge = np.sin(ridge_freq * r / width + theta)

            # Add Perlin noise distortion
            nx = x / noise_scale
            ny = y / noise_scale
            noise_val = pnoise2(nx, ny, octaves=4, persistence=0.5, lacunarity=2.0)

            # Combine spiral and noise
            value = spiral_ridge + noise_strength * noise_val
            spiral_pattern[y, x] = value

    # Normalize to 0–1
    spiral_pattern = (spiral_pattern - spiral_pattern.min()) / (spiral_pattern.max() - spiral_pattern.min())
    return spiral_pattern

# Parameters
shape = (256, 256)
E = 10.0  # Elastic modulus
eta = 5.0  # Viscosity
dt = 0.01  # Time step
timesteps = 100

# Example: 2D fingerprint ridge height map
ridge_profile = generate_spiral(shape=shape)

# Initial state
strain = np.zeros_like(ridge_profile)
strain_history = []

# Simulate a pressure distribution (Gaussian blob as contact)
def apply_pressure(ridge_profile, center, sigma=5.0, peak_pressure=1.0):
    X, Y = np.meshgrid(np.arange(ridge_profile.shape[1]),
                       np.arange(ridge_profile.shape[0]))
    distance_squared = (X - center[0])**2 + (Y - center[1])**2
    pressure_map = peak_pressure * np.exp(-distance_squared / (2 * sigma**2))
    return pressure_map

# Define pressure input over time (could be a dynamic contact)
def pressure_sequence(t, center=(127, 127), sigma=50, peak=1.0):
    # Example: oscillating Gaussian contact
    pressure = apply_pressure(ridge_profile, center, sigma, peak * np.sin(2 * np.pi * t * 0.1))
    return pressure

# Time-stepping simulation using Kelvin–Voigt model
for t in range(timesteps):
    pressure = pressure_sequence(t)
    dstrain_dt = (pressure - E * strain) / eta
    strain += dt * dstrain_dt
    strain_history.append(strain.copy())

# Convert to array: shape = (time, x, y)
strain_history = np.stack(strain_history)

# Example: compute receptor responses at final time step
sa1_response = laplace(strain_history[-1])  # Spatial detail
sa2_response = gaussian_filter(strain_history[-1], sigma=10)  # Broad spatial stretch

# FA1: First derivative over time
fa1_response = strain_history[1:] - strain_history[:-1]

# FA2: Second derivative over time (vibration sensitivity)
fa2_response = strain_history[2:] - 2 * strain_history[1:-1] + strain_history[:-2]

for t in range(timesteps - 2):
    plt.clf()
    plt.suptitle(f'Time step {t}')
    plt.subplot(2, 2, 1)
    plt.title('SA1')
    plt.imshow(laplace(strain_history[t]), cmap='plasma')

    plt.subplot(2, 2, 2)
    plt.title('SA2')
    plt.imshow(gaussian_filter(strain_history[t], sigma=10), cmap='plasma')

    plt.subplot(2, 2, 3)
    plt.title('FA1')
    plt.imshow(fa1_response[t], cmap='viridis')

    plt.subplot(2, 2, 4)
    plt.title('FA2')
    plt.imshow(fa2_response[t], cmap='viridis')

    plt.pause(0.05)