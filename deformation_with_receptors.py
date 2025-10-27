from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from noise import pnoise2
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                dist = np.sqrt(np.abs((x - center[0])**2 + (y - center[1])**2))
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

def plot_3d_surface(X, Y, Z, title, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if ax is None:
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
    
    return surf

def run_skin_deformation_simulation(surface_type='spiral', pressure_pattern='circular'):
    # Initial surface profile
    if surface_type == 'spiral':
        surface = generate_spiral(shape=(nx, ny))
    else:
        surface = generate_wave(shape=(nx, ny))
    
    # Applied pressure
    sigma = create_pressure_map((nx, ny), pattern_type=pressure_pattern, intensity=1.0)
    
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
    
    # Compute receptor responses
    sa1 = sa1_response(epsilon, curvature)
    sa2 = sa2_response(epsilon, curvature)
    fa1 = fa1_response(d_epsilon_dt, curvature)
    fa2 = fa2_response(d2_epsilon_dt2, curvature)
    
    return {
        'surface': surface,
        'deformation': deformation,
        'epsilon': epsilon, 
        'curvature': curvature,
        'sa1': sa1,
        'sa2': sa2,
        'fa1': fa1,
        'fa2': fa2,
        'time': time
    }

def analyze_specific_points(results, points_of_interest):
    """
    Analyze mechanoreceptor responses at specific points of interest over time
    """
    time = results['time']
    response_data = {}
    
    for idx, (x, y) in enumerate(points_of_interest):
        point_name = f'Point {idx+1} ({x},{y})'
        response_data[point_name] = {
            'SA1': results['sa1'][y, x, :],
            'SA2': results['sa2'][y, x, :],
            'FA1': results['fa1'][y, x, :],
            'FA2': results['fa2'][y, x, :]
        }
        
    return response_data, time

def plot_receptor_responses_over_time(response_data, time):
    """
    Plot mechanoreceptor responses over time for specific points
    """
    receptor_types = ['SA1', 'SA2', 'FA1', 'FA2']
    num_points = len(response_data)
    
    fig, axs = plt.subplots(len(receptor_types), 1, figsize=(12, 12), sharex=True)
    
    for i, receptor in enumerate(receptor_types):
        for point, data in response_data.items():
            axs[i].plot(time, data[receptor], label=point)
        
        axs[i].set_ylabel(f'{receptor} Response')
        axs[i].set_title(f'{receptor} Mechanoreceptor Response Over Time')
        axs[i].legend()
    
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    return fig

def create_receptor_animation(results, receptor_type='sa1', interval=100):
    """
    Create animation of receptor response over time
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = results[receptor_type]
    vmin = np.min(data)
    vmax = np.max(data)
    
    # Initial plot
    im = ax.imshow(data[:, :, 0], cmap='viridis', vmin=vmin, vmax=vmax, 
                  extent=[x.min(), x.max(), y.min(), y.max()])
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    title = ax.set_title(f'{receptor_type.upper()} Response at t=0')
    
    def update(frame):
        im.set_array(data[:, :, frame])
        title.set_text(f'{receptor_type.upper()} Response at t={results["time"][frame]:.2f}s')
        return [im, title]
    
    ani = FuncAnimation(fig, update, frames=range(0, len(results['time']), interval), 
                        blit=True, interval=50)
    
    plt.tight_layout()
    return ani

def create_comparison_plots(results):
    """
    Create side-by-side comparison plots of initial and deformed surfaces
    along with receptor responses at specific time points
    """
    time_points = [0, len(time)//5, len(time)//2, -1]  # Start, 20%, 50%, End
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot initial surface
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_3d_surface(X, Y, results['surface'], 'Initial Surface', ax=ax1)
    
    # Plot final deformation
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_3d_surface(X, Y, results['deformation'][:, :, -1], 'Final Deformed Surface', ax=ax2)
    
    # Plot receptor responses
    receptors = ['sa1', 'sa2', 'fa1', 'fa2']
    
    for i, receptor in enumerate(receptors):
        ax = fig.add_subplot(2, 3, i+3, projection='3d')
        plot_3d_surface(X, Y, results[receptor][:, :, -1], f'Final {receptor.upper()} Response', ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def export_data_to_csv(results, points_of_interest):
    """
    Export mechanoreceptor response data to CSV for further analysis
    """
    response_data, time_array = analyze_specific_points(results, points_of_interest)
    
    # Create DataFrame
    data_frames = []
    
    for point_name, responses in response_data.items():
        for receptor, values in responses.items():
            df = pd.DataFrame({
                'time': time_array,
                'point': [point_name] * len(time_array),
                'receptor': [receptor] * len(time_array),
                'response': values
            })
            data_frames.append(df)
    
    # Combine all data
    result_df = pd.concat(data_frames)
    
    # Export to CSV
    result_df.to_csv('mechanoreceptor_responses.csv', index=False)
    
    # Also create a pivot table for easier analysis
    pivot_df = result_df.pivot_table(
        index=['time'], 
        columns=['point', 'receptor'], 
        values='response'
    )
    
    pivot_df.to_csv('mechanoreceptor_responses_pivot.csv')
    
    return result_df, pivot_df

# Main analysis function
def analyze_skin_mechanoreceptors(surface_type='spiral', pressure_type='circular'):
    # Run simulation
    results = run_skin_deformation_simulation(surface_type, pressure_type)
    
    # Define points of interest (example - could be specific mechanoreceptor locations)
    # Points at center, near center, and periphery
    points_of_interest = [
        (nx//2, ny//2),  # Center
        (nx//2 + 10, ny//2 + 10),  # Near center
        (nx//2 - 15, ny//2 + 5),  # Another near center point
        (nx//4, ny//4),  # Periphery
        (3*nx//4, 3*ny//4)  # Opposite periphery
    ]
    
    # Analyze and plot time-based responses
    response_data, time_array = analyze_specific_points(results, points_of_interest)
    time_plot = plot_receptor_responses_over_time(response_data, time_array)
    
    # Create comparison plots
    comparison_plot = create_comparison_plots(results)
    
    # Create animations
    sa1_animation = create_receptor_animation(results, 'sa1')
    fa1_animation = create_receptor_animation(results, 'fa1')
    
    # Export data
    result_df, pivot_df = export_data_to_csv(results, points_of_interest)
    
    return {
        'results': results,
        'response_data': response_data,
        'time_plot': time_plot,
        'comparison_plot': comparison_plot,
        'sa1_animation': sa1_animation,
        'fa1_animation': fa1_animation,
        'dataframe': result_df,
        'pivot_dataframe': pivot_df
    }

if __name__ == "__main__":
    # Run analysis with default parameters
    analysis = analyze_skin_mechanoreceptors(surface_type='spiral', pressure_type='circular')
    
    # Display plots
    plt.show()
    
    # To save animations:
    # analysis['sa1_animation'].save('sa1_response.mp4', writer='ffmpeg', dpi=100)
    # analysis['fa1_animation'].save('fa1_response.mp4', writer='ffmpeg', dpi=100)
    
    print("Analysis complete. Data exported to 'mechanoreceptor_responses.csv'")