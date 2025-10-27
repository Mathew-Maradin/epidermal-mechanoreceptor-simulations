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

# Example: 2D fingerprint ridge height map
ridge_profile = generate_spiral()  # Replace with real data

# Simulate a pressure distribution (Gaussian blob as contact)
def apply_pressure(ridge_profile, center, sigma=5.0, peak_pressure=1.0):
    X, Y = np.meshgrid(np.arange(ridge_profile.shape[1]),
                       np.arange(ridge_profile.shape[0]))
    distance_squared = (X - center[0])**2 + (Y - center[1])**2
    pressure_map = peak_pressure * np.exp(-distance_squared / (2 * sigma**2))
    return pressure_map

pressure = apply_pressure(ridge_profile, center=(127, 127), sigma=50)

# Strain or deformation response (simplified)
# Assume deformation is inversely proportional to ridge height
strain = pressure / (ridge_profile + 1e-3)

# SA1: Local spatial detail (laplacian of strain)
sa1_response = laplace(strain)

# SA2: Broad low-frequency deformation field
sa2_response = gaussian_filter(strain, sigma=10)

# FA1: Transient response - temporal derivative (simplified)
# Simulate a time-varying sequence of pressures
strain_t1 = strain
strain_t2 = strain_t1 * 0.8  # Pressure reduced in next timestep
fa1_response = strain_t2 - strain_t1  # Temporal gradient

# FA2: High-frequency vibration – second derivative (acceleration)
strain_t3 = strain_t2 * 1.2  # Sudden increase
acceleration = strain_t3 - 2 * strain_t2 + strain_t1
fa2_response = gaussian_filter(acceleration, sigma=1)

plt.figure(figsize=(10, 8))
for i, (title, data) in enumerate(zip(
    ['SA1', 'SA2', 'FA1', 'FA2'],
    [sa1_response, sa2_response, fa1_response, fa2_response])):
    plt.subplot(2, 2, i + 1)
    plt.title(title)
    plt.imshow(data, cmap='viridis')
    plt.colorbar()
plt.tight_layout()
plt.show()
