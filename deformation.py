from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from noise import pnoise2

# Parameters
E = 0.42  # Elastic modulus (stiffness)
eta = 0.16  # Viscosity coefficient
dt = 0.01  # Time step
time = np.arange(0, 10, dt)  # Time array
nx, ny = 100, 100  # Grid size
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

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

# Initial surface profile
surface = generate_spiral(shape=(nx, ny))

def create_pressure_map(shape, pattern_type='circular', center=None, radius=None, intensity=1.0):
    """
    Create different pressure patterns.

    Parameters:

    shape: Shape of the heightmap (height, width)
    pattern_type: 'circular', 'linear', or 'point'
    center: Center point of pressure (x, y)
    radius: Radius for circular pattern or width for linear
    intensity: Maximum pressure intensity (0-1)

    Returns:
    pressure_map: A 2D array representing pressure distribution
    """
    height, width = shape
    pressure_map = np.zeros(shape)

    if center is None:
        center = (width // 2, height // 2)

    if radius is None:
        radius = min(width, height) // 4

    if pattern_type == 'circular':
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    # Pressure decreases from center
                    pressure_map[y, x] = intensity * (1 - dist/radius)

    elif pattern_type == 'point':
        # Single point pressure
        x, y = center
        pressure_map[y, x] = intensity
        pressure_map = gaussian_filter(pressure_map, sigma=radius/4)

    elif pattern_type == 'linear':
        # Linear pressure across width
        for y in range(height):
            for x in range(width):
                if abs(y - center[1]) < radius:
                    pressure_map[y, x] = intensity * (1 - abs(y - center[1])/radius)

    return pressure_map

# Applied stress 
sigma = create_pressure_map((nx, ny), pattern_type="linear")

# Initialize strain and its derivative
deformation = np.zeros((nx, ny, len(time)))
curvature = np.zeros((nx, ny, len(time)))
epsilon = np.zeros((nx, ny, len(time)))
d_epsilon_dt = np.zeros((nx, ny, len(time)))
d2_epsilon_dt2 = np.zeros((nx, ny, len(time)))

# Solve Kelvin-Voigt model using finite differences
for i in range(1, len(time)):
    d_epsilon_dt[:, :, i] = (sigma - E * epsilon[:, :, i-1]) / eta
    epsilon[:, :, i] = epsilon[:, :, i-1] + d_epsilon_dt[:, :, i] * dt
    deformation[:, :, i] = surface - epsilon[:, :, i]
    curvature[:, :, i] = np.gradient(np.gradient(deformation[:, :, i], axis=0), axis=0) + np.gradient(np.gradient(deformation[:, :, i], axis=1), axis=1)

# Compute second time derivative (acceleration)
d2_epsilon_dt2[:, :, 1:-1] = (d_epsilon_dt[:, :, 2:] - d_epsilon_dt[:, :, :-2]) / (2 * dt)

# Compute receptor responses (modulated by surface curvature)
def sa1_response(epsilon, curvature):
    # SA1 responds to static deformation, modulated by curvature
    return epsilon * (1 + curvature)

def sa2_response(epsilon, curvature):
    # SA2 responds to sustained deformation, modulated by curvature
    return np.cumsum(epsilon, axis=2) * dt * (1 + curvature)

def fa1_response(d_epsilon_dt, curvature):
    # FA1 responds to rate of deformation, modulated by curvature
    return d_epsilon_dt * (1 + curvature)

def fa2_response(d2_epsilon_dt2, curvature):
    # FA2 responds to acceleration of deformation, modulated by curvature
    return d2_epsilon_dt2 * (1 + curvature)

# Compute receptor responses
sa1 = sa1_response(epsilon, curvature)
sa2 = sa2_response(epsilon, curvature)
fa1 = fa1_response(d_epsilon_dt, curvature)
fa2 = fa2_response(d2_epsilon_dt2, curvature)

# 3D Visualization
def plot_3d_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# Plot initial surface profile
plot_3d_surface(X, Y, surface, 'Initial Spiral Ridge Surface')

# Final time step
t_idx = -1  

# Plot deformed surface
plot_3d_surface(X, Y, deformation[:, :, t_idx], 'Deformed Spiral Ridge Surface')

# Plot receptor responses at a specific time step
# plot_3d_surface(X, Y, sa1[:, :, t_idx], 'SA1 Response at t=10')
# plot_3d_surface(X, Y, sa2[:, :, t_idx], 'SA2 Response at t=10')
# plot_3d_surface(X, Y, fa1[:, :, t_idx], 'FA1 Response at t=10')
# plot_3d_surface(X, Y, fa2[:, :, t_idx], 'FA2 Response at t=10')

# Plot Deformation thru time
# Set up the figure and axis for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the surface plot
def init():
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Response')
    ax.set_zlim(-1, 1)
    ax.set_title('Receptor Response Animation')
    return fig,

# Update function for animation
def update(frame):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Response')
    ax.set_zlim(-1, 1)
    ax.set_title(f'Receptor Response at t={time[frame]:.2f}')
    
    # Plot the receptor response at the current frame
    response = fa2[:, :, frame]  # Change to sa2, fa1, or fa2 for other receptors
    ax.plot_surface(X, Y, response, cmap=cm.viridis, linewidth=0, antialiased=False)
    return fig,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=False, interval=5)

# Save the animation as a video file
# ani.save('fa2_response_animation.mp4', writer='ffmpeg', fps=30)

# Display the animation interactively
plt.show()

# Total Deformation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Calculate Max Deformation
def_avg = [epsilon[:, :, i].mean() for i in range(len(time))]
def_max = [epsilon[:, :, i].max() for i in range(len(time))]

ax.plot(time, def_avg, label="Avg Displacement")
ax.plot(time, def_max, label="Max Displacement")

plt.title("Displacement vs. Time")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.legend()
plt.show()