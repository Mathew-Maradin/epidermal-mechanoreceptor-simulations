import numpy as np
import matplotlib.pyplot as plt

# ---------- PARAMETERS ----------
E = 1.0     # Elastic modulus
eta = 0.5   # Viscosity
dt = 0.1    # Time step
T = 30      # Number of time steps

# ---------- 1. Generate synthetic fingerprint ridge pattern ----------
def generate_ridge_pattern(shape=(100, 100), ridge_spacing=5):
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    ridge_pattern = 0.5 * (np.sin(2 * np.pi * y / ridge_spacing) + 1)
    return ridge_pattern

ridge_base = generate_ridge_pattern()

# ---------- 2. External stress field (finger pressing center) ----------
def generate_stress_field(shape=(100, 100), center=None, max_stress=1.0, falloff=0.005):
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    r2 = (x - center[1])**2 + (y - center[0])**2
    stress = max_stress * np.exp(-falloff * r2)
    return stress

stress_field = generate_stress_field(ridge_base.shape)

# ---------- 3. Simulate strain evolution ----------
strain_history = []
strain = np.zeros_like(ridge_base)

for t in range(T):
    dstrain_dt = (stress_field - E * strain) / eta
    strain = strain + dstrain_dt * dt
    strain_history.append(strain.copy())

strain_history = np.stack(strain_history, axis=0)  # shape: (T, H, W)

# ---------- 4. Deform ridge pattern over time ----------
deformed_ridges = ridge_base[None, :, :] * (1 + strain_history)

# ---------- 5. Visualize as animation ----------
import matplotlib.animation as animation

fig, ax = plt.subplots()
im = ax.imshow(deformed_ridges[0], cmap='gray', vmin=0, vmax=2)
ax.set_title("Kelvin–Voigt Ridge Deformation")
ax.axis('off')

def update(frame):
    im.set_data(deformed_ridges[frame])
    ax.set_title(f"Time Step {frame}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=T, interval=200)
plt.show()