import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameters
E = 1.0  # Elastic modulus (stiffness)
eta = 0.5  # Viscosity coefficient
dt = 0.01  # Time step
time = np.arange(0, 10, dt)  # Time array
nx, ny = 256, 256  # Grid size
dx, dy = 1.0, 1.0  # Spatial resolution

# Create a 2D grid for the surface
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# Define the surface profile (wave-like or spiral)
def surface_profile(X, Y, pattern="wave"):
    if pattern == "wave":
        return np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # Wave-like pattern
    elif pattern == "spiral":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        return np.sin(R + theta)  # Spiral pattern
    else:
        return np.zeros_like(X)  # Flat surface

# Initial surface profile
surface = surface_profile(X, Y, pattern="spiral")

# Applied stress (e.g., a Gaussian pressure at the center)
sigma = np.zeros((nx, ny))
sigma[nx//2, ny//2] = 1.0  # Point stress at the center
sigma = gaussian_filter(sigma, sigma=50)  # Smooth the stress distribution

# Initialize strain and its derivative
epsilon = np.zeros((nx, ny, len(time)))
d_epsilon_dt = np.zeros((nx, ny, len(time)))

# Solve Kelvin-Voigt model using finite differences
for i in range(1, len(time)):
    d_epsilon_dt[:, :, i] = (sigma - E * epsilon[:, :, i-1]) / eta
    epsilon[:, :, i] = epsilon[:, :, i-1] + d_epsilon_dt[:, :, i] * dt

# Simulate receptor responses
def sa1_response(epsilon):
    # SA1 responds to static deformation
    return epsilon

def sa2_response(epsilon):
    # SA2 responds to sustained deformation
    return np.cumsum(epsilon, axis=2) * dt

def fa1_response(d_epsilon_dt):
    # FA1 responds to rapid changes (velocity)
    return d_epsilon_dt

def fa2_response(d_epsilon_dt):
    # FA2 responds to high-frequency changes (acceleration)
    return np.gradient(d_epsilon_dt, dt, axis=2)

# Compute receptor responses
sa1 = sa1_response(epsilon)
sa2 = sa2_response(epsilon)
fa1 = fa1_response(d_epsilon_dt)
fa2 = fa2_response(d_epsilon_dt)

# 3D Visualization
def plot_3d_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Deformation')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# Plot initial surface profile
plot_3d_surface(X, Y, surface, 'Initial Surface Profile')

# Plot deformation at a specific time step
plot_3d_surface(X, Y, epsilon[:, :, -1], 'Deformation at t=10')

# Plot receptor responses at a specific time step
plot_3d_surface(X, Y, sa1[:, :, -1], 'SA1 Response at t=10')
plot_3d_surface(X, Y, sa2[:, :, -1], 'SA2 Response at t=10')
plot_3d_surface(X, Y, fa1[:, :, -1], 'FA1 Response at t=10')
plot_3d_surface(X, Y, fa2[:, :, -1], 'FA2 Response at t=10')