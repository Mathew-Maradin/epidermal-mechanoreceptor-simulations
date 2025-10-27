from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

def generate_fingerprint_ridge_pattern(shape=(256, 256), scale=50.0, ridge_freq=10.0):
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

def generate_spiral_fingerprint(shape=(256, 256), ridge_freq=15.0, noise_scale=40.0, noise_strength=0.5):
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

def generate_flat(shape=(256, 256)):
    return np.ones(shape=shape) * 0.5

# Generate and display
fingerprint_pattern = generate_fingerprint_ridge_pattern(ridge_freq=25)
spiral_fp = generate_spiral_fingerprint(ridge_freq=100)

# Apply Stress to patterns
def apply_deformation_x(ridges : NDArray[np.float16]):
    # New 
    alt_ridges = ridges.copy()

    # Loop thru each row
    for (index, x_row) in enumerate(ridges):
        # Add buffer for end
        x_row_pad = np.roll(x_row, -1)
        x_row_pad[-1] = 0
        x_row_pad_2 = np.roll(x_row_pad, -1)
        x_row_pad_2[-1] = 0

        # Apply shift
        alt_ridges[index, :] += -0.8*x_row + 0.4*x_row_pad + 0.2*x_row_pad_2

    return alt_ridges

fingerprint_stress = apply_deformation_x(fingerprint_pattern)
spiral_stress = apply_deformation_x(spiral_fp)

def display_data(ridges : NDArray[np.float16], ridges_stressed : NDArray[np.float16], x, y, max_res=25, slice_index=127):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.contour3D(x, y, ridges, max_res)
    ax1.set_zlim(0, 1)
    ax1.set_title('Original')

    ax2.contour3D(x, y, ridges_stressed, max_res)
    ax2.set_zlim(0, 1)
    ax2.set_title('Stressed')

    ax3.plot(x, ridges[slice_index], 1)
    ax3.plot(x, ridges_stressed[slice_index], 1)
    ax3.set_title('Slice')

    ax4.plot(x, ridges[slice_index] - ridges_stressed[slice_index], 1)
    ax4.set_title('Difference')

    plt.show()

# set up the figure and Axes
x = np.arange(0, 256)
y = np.arange(0, 256)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

max_res = 25

ax1.contour3D(x, y, fingerprint_pattern, max_res)
ax1.set_zlim(0, 1)
ax1.set_title('Waves')

ax3.contour3D(x, y, spiral_fp, max_res)
ax3.set_zlim(0, 1)
ax3.set_title('Spiral')

ax2.contour3D(x, y, fingerprint_stress, max_res)
ax2.set_zlim(0, 1)
ax2.set_title('Waves (Stressed)')

ax4.contour3D(x, y, spiral_stress, max_res)
ax4.set_zlim(0, 1)
ax4.set_title('Spiral (Stressed)')

plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

slice_index = 100

ax1.plot(x, fingerprint_pattern[slice_index], 1)
ax1.set_title('Waves')

ax3.plot(x, spiral_fp[slice_index], 1)
ax3.set_title('Spiral')

ax2.plot(x, fingerprint_stress[slice_index], 1)
ax2.set_title('Waves (Stressed)')

ax4.plot(x, spiral_stress[slice_index], 1)
ax4.set_title('Spiral (Stressed)')

plt.show()

def sa1_response(signal):
    return np.mean(signal)  # Sustained pressure (integrated over time)

def sa2_response(signal):
    return np.std(signal)  # Response to stretch or shear (variance)

def fa1_response(signal):
    return np.max(np.abs(np.diff(signal)))  # Rapid pressure changes (high frequency)

def fa2_response(signal):
    return np.max(np.abs(np.diff(signal)))  # Maximum rate of change in vibration

print(sa1_response(fingerprint_pattern - fingerprint_stress))
print(sa2_response(fingerprint_pattern - fingerprint_stress))
print(fa1_response(fingerprint_pattern - fingerprint_stress))
print(fa2_response(fingerprint_pattern - fingerprint_stress))

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(fingerprint_pattern - fingerprint_stress)
ax1.set_title('Waves (Diff)')

ax2.imshow(spiral_fp - spiral_stress)
ax2.set_title('Spiral (Diff)')

plt.show()

# Flat comparison
flat = generate_flat()
flat_stressed = apply_deformation_x(flat)
display_data(flat, flat_stressed, x, y)

print(sa1_response(flat - flat_stressed))
print(sa2_response(flat - flat_stressed))
print(fa1_response(flat - flat_stressed))
print(fa2_response(flat - flat_stressed))