"""
ER3BP UNIFIED LABORATORY (Jupiter System)
=========================================
Dynamics Analysis & Initial Condition Generator for N-Body Simulations.

Description:
    This script solves the Elliptic Restricted Three-Body Problem (ER3BP) for 
    the Jupiter-Europa and Jupiter-Ganymede systems. It allows for:
    1. Stability analysis via Poincaré Maps.
    2. Visualization of trajectories in rotating frames (2D/3D).
    3. Transformation of stable orbits from the Rotating/Pulsating frame 
       to the Inertial frame for export to N-Body propagators.

Author: Juan Sebastián Victoriono - Juan Pablo Cruz Gutiérrez, Universidad Nacional de Colombia.
System: ER3BP (Hamiltonian Formulation)
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1. SYSTEM CONFIGURATION & CONSTANTS
# =============================================================================

# Global Configuration Holder
ACTIVE_CONFIG = {
    'name': None,       # System Name (e.g., 'EUROPA')
    'mu': 0.0,          # Mass Parameter
    'epsilon': 0.0,     # Orbital Eccentricity
    'L_dist': 1.0,      # Characteristic Length (m)
    'R_jup_dim': 0.0,   # Dimensionless Jupiter Radius
    'R_moon_dim': 0.0,  # Dimensionless Moon Radius
    'target_body': None # Dictionary containing Moon's physical data
}

# --- CELESTIAL BODY DATA (JPL HORIZONS SNAPSHOT) ---

DATA_JUPITER = {
    'r': np.array([-9.575324576133315e4, -6.150456912510757e4, -3.072242063097825e3]),
    'v': np.array([4.898609412767204e-1, -9.537003839258596e-1, -3.028640371889238e-2]),
    'm': 1.89818e27,
    'radius_km': 71492.0
}

DATA_EUROPA = {
    'r': np.array([3.904990291276731e8, -5.392340224863521e8, -9.037579559937003e6]),
    'v': np.array([1.116678537182261e4, 8.174989252496797e3, 5.127064307826501e2]),
    'm': 4.79870e22,
    'epsilon': 0.0094,
    'radius_km': 1560.8
}

DATA_GANYMEDE = {
    'r': np.array([8.626133793413050e8, -6.313676115858576e8, -1.178962605631183e4]),
    'v': np.array([6.431976727831559e3, 8.781079262403239e3, 4.266350959168634e2]),
    'm': 1.48157e23,
    'epsilon': 0.0013,
    'radius_km': 2634.1
}

# Fixed background body for N-Body export
STR_IO = """# Initial condition for Io
-1.169290164457216e7  4.206724877968931e8  1.469934463084565e7 -1.735159563649886e4 -5.367255748605908e2 -2.666674111749237e2 4.79872e22"""

# =============================================================================
# 2. SYSTEM SETUP & INITIALIZATION
# =============================================================================

def setup_system(choice: str):
    """
    Initializes the physics engine with the selected moon's parameters.
    Calculates dimensionless quantities and Lagrange points.
    """
    global X_L4, Y_L4
    
    if choice == '1':
        target_data = DATA_EUROPA
        name = "EUROPA"
    elif choice == '2':
        target_data = DATA_GANYMEDE
        name = "GANYMEDE"
    else:
        print(" [!] Invalid choice. Defaulting to EUROPA.")
        target_data = DATA_EUROPA
        name = "EUROPA"

    # 1. Characteristic Length (Distance between primaries)
    r_rel = target_data['r'] - DATA_JUPITER['r']
    dist_L = np.linalg.norm(r_rel)

    # 2. Mass Parameter (mu = m2 / (m1 + m2))
    mu_val = target_data['m'] / (DATA_JUPITER['m'] + target_data['m'])

    # 3. Update Configuration
    ACTIVE_CONFIG.update({
        'name': name,
        'mu': mu_val,
        'epsilon': target_data['epsilon'],
        'L_dist': dist_L,
        'target_body': target_data,
        # Dimensionless radii for collision detection
        'R_jup_dim': (DATA_JUPITER['radius_km'] * 1000) / dist_L,
        'R_moon_dim': (target_data['radius_km'] * 1000) / dist_L
    })

    # 4. Calculate Lagrange Point L4 (Equilateral Triangle)
    X_L4 = 0.5 - mu_val
    Y_L4 = np.sqrt(3) / 2

    # 5. Print Dashboard
    print(f"\n✅ SYSTEM INITIALIZED: JUPITER-{name}")
    print(f"   {'─'*30}")
    print(f"   ├── Mass Param (μ):   {mu_val:.6e}")
    print(f"   ├── Eccentricity (e): {ACTIVE_CONFIG['epsilon']}")
    print(f"   ├── Char. Length (L): {dist_L/1000:,.1f} km")
    print(f"   ├── R_Jupiter (dim):  {ACTIVE_CONFIG['R_jup_dim']:.4f}")
    print(f"   └── R_Moon (dim):     {ACTIVE_CONFIG['R_moon_dim']:.4f}")
    print(f"   {'─'*30}")

# =============================================================================
# 3. VISUALIZATION STYLING
# =============================================================================

plt.rcParams.update({
    'figure.facecolor': '#000000', 
    'axes.facecolor': '#000000', 
    'savefig.facecolor': '#000000',
    'text.color': '#E0E0E0', 
    'axes.labelcolor': '#E0E0E0', 
    'xtick.color': '#E0E0E0', 
    'ytick.color': '#E0E0E0',
    'font.family': 'sans-serif', 
    'font.size': 10, 
    'axes.grid': True, 
    'grid.color': '#444444', 
    'grid.linestyle': ':', 
    'grid.alpha': 0.5,
    'lines.linewidth': 0.8, 
    'legend.facecolor': '#1A1A1A', 
    'legend.edgecolor': '#E0E0E0'
})

colors = {
    'jupiter': '#F4A460', 
    'secondary': '#A6D96A', 
    'spacecraft': '#FDFEFE', 
    'l4_marker': '#FF3333'
}

# =============================================================================
# 4. PHYSICS ENGINE (ER3BP HAMILTONIAN)
# =============================================================================

def solve_kepler(M: float, e: float, tol: float = 1e-10) -> float:
    """Newton-Raphson solver for Kepler's Equation: M = E - e*sin(E)."""
    E = M
    for _ in range(20):
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - delta
        if abs(delta) < tol: break
    return E

def get_n_t(t: float, eps: float) -> float:
    """Calculates the instantaneous mean motion of the system."""
    E = solve_kepler(t, eps)
    rho = 1 - eps * np.cos(E)
    return np.sqrt(1 - eps**2) / (rho**2)

def equations_of_motion_hamiltonian(t, state, mu, eps):
    """
    Differential equations for the ER3BP in canonical pulsating coordinates.
    
    State vector: [x, y, px, py]
    Returns: [dx/dt, dy/dt, dpx/dt, dpy/dt]
    """
    x, y, px, py = state
    E = solve_kepler(t, eps)
    rho = 1 - eps * np.cos(E)
    n_t = np.sqrt(1 - eps**2) / (rho**2)
    
    # Positions relative to primaries
    dx1 = x + mu * rho
    dx2 = x - (1 - mu) * rho
    
    # Distances (with softening to avoid singularities)
    r1 = np.sqrt(dx1**2 + y**2)
    r2 = np.sqrt(dx2**2 + y**2)
    r1, r2 = max(r1, 1e-4), max(r2, 1e-4)
    
    # Equations of Motion
    dx_dt = px + y * n_t
    dy_dt = py - x * n_t
    dpx_dt = n_t * py - ((1 - mu) * dx1) / r1**3 - (mu * dx2) / r2**3
    dpy_dt = -n_t * px - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    
    return [dx_dt, dy_dt, dpx_dt, dpy_dt]

def velocity_to_momentum(x, y, vx, vy, t, eps):
    """Converts physical velocity (vx, vy) to canonical momentum (px, py)."""
    n_t = get_n_t(t, eps)
    return vx - n_t * y, vy + n_t * x

# --- EVENT HANDLERS (COLLISION & ESCAPE) ---

def crossing_L4_plane(t, state, mu, eps):
    """Event: Spacecraft crosses the x-plane of L4."""
    return state[0] - X_L4
crossing_L4_plane.direction = 0

def crash_jupiter(t, state, mu, eps):
    """Event: Collision with Jupiter."""
    return np.sqrt((state[0] + mu)**2 + state[1]**2) - ACTIVE_CONFIG['R_jup_dim']
crash_jupiter.terminal = True

def crash_moon(t, state, mu, eps):
    """Event: Collision with the Secondary Body."""
    return np.sqrt((state[0] - (1 - mu))**2 + state[1]**2) - ACTIVE_CONFIG['R_moon_dim']
crash_moon.terminal = True

def escape_system(t, state, mu, eps):
    """Event: Spacecraft escapes the system (Radius > 3.5 AU_dim)."""
    return 3.5 - np.sqrt(state[0]**2 + state[1]**2)
escape_system.terminal = True

# =============================================================================
# 5. UTILITIES & VISUALIZATION
# =============================================================================

class ProgressBar:
    """Simple ASCII progress bar for console feedback."""
    def __init__(self, total, prefix='Computing', length=40):
        self.total = total
        self.prefix = prefix
        self.length = length
        
    def update(self, current):
        if self.total == 0: return
        percent = float(current) * 100 / self.total
        filled = int(self.length * current // self.total)
        bar = '█' * filled + '░' * (self.length - filled)
        sys.stdout.write(f'\r   {self.prefix} |{bar}| {percent:.1f}%')
        sys.stdout.flush()
        
    def finish(self):
        sys.stdout.write('\n')

def run_single_trajectory_vis():
    """Integrates and plots a single trajectory starting near L4."""
    print(f"\n[Graph] Running Visualization for {ACTIVE_CONFIG['name']}...")
    mu = ACTIVE_CONFIG['mu']
    eps = ACTIVE_CONFIG['epsilon']
    
    # Initial Conditions
    y_start = Y_L4 + 0.035 
    vx_start = 0.046939 
    px, py = velocity_to_momentum(X_L4, y_start, vx_start, 0.0, 0.0, eps)
    
    print("   >> Integrating equations of motion...")
    sol = solve_ivp(equations_of_motion_hamiltonian, (0, 8000 * np.pi), [X_L4, y_start, px, py], 
                    args=(mu, eps), method='DOP853', rtol=1e-10, atol=1e-12, max_step=0.05)
    
    x, y, px_arr, t = sol.y[0], sol.y[1], sol.y[2], sol.t
    vx = [px_arr[i] + get_n_t(t[i], eps) * y[i] for i in range(len(t))]

    # Plotting
    fig = plt.figure(figsize=(14, 6))
    
    # 2D Plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, y, color=colors['spacecraft'], linewidth=0.6, alpha=0.8)
    ax1.plot([X_L4], [Y_L4], '+', color=colors['l4_marker'], markersize=12, label='L4')
    ax1.plot([-mu], [0], 'o', color=colors['jupiter'], markersize=10, label='Jupiter')
    ax1.plot([1-mu], [0], 'o', color=colors['secondary'], markersize=6, label=ACTIVE_CONFIG['name'])
    ax1.set_title(f"2D Trajectory ({ACTIVE_CONFIG['name']})")
    ax1.set_xlabel("x (rotating)"); ax1.set_ylabel("y (rotating)")
    ax1.axis('equal')
    ax1.legend()

    # 3D Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    p = ax2.scatter(x, y, vx, c=t, cmap='plasma', s=0.5, alpha=0.6)
    ax2.set_title("3D Phase Space Evolution")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("Vx")
    fig.colorbar(p, ax=ax2, label='Time (dimless)')
    
    plt.show()

def run_poincare_map():
    """Generates a Poincaré Section (Y vs Vy) at the L4 crossing."""
    print(f"\n[Map] Generating Poincaré Map for {ACTIVE_CONFIG['name']}...")
    mu = ACTIVE_CONFIG['mu']
    eps = ACTIVE_CONFIG['epsilon']
    
    y_values = np.linspace(Y_L4 - 0.15, Y_L4 + 0.15, 40)
    vx_values = np.linspace(-0.015, 0.015, 5)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Poincaré Surface of Section ({ACTIVE_CONFIG['name']})\nPlane x = x_L4")
    ax.set_xlabel("y coordinate"); ax.set_ylabel("Vy velocity")
    
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(vx_values)))
    progress = ProgressBar(len(y_values)*len(vx_values))
    count = 0
    
    for j, vx in enumerate(vx_values):
        c = colors_map[j]
        for y_start in y_values:
            count += 1
            px, py = velocity_to_momentum(X_L4, y_start, vx, 0.0, 0.0, eps)
            try:
                sol = solve_ivp(
                    equations_of_motion_hamiltonian, (0, 500*np.pi), [X_L4, y_start, px, py], 
                    args=(mu, eps), method='DOP853', 
                    events=[crossing_L4_plane, crash_jupiter, crash_moon, escape_system],
                    rtol=1e-9, atol=1e-9, max_step=0.1
                )
                
                # Extract crossing events
                if len(sol.t_events[0]) > 0: 
                    y_ev = sol.y_events[0]
                    t_ev = sol.t_events[0]
                    # Transform momentum back to velocity for plotting
                    map_y = y_ev[:, 1]
                    map_vy = [y_ev[k, 3] - get_n_t(t_ev[k], eps) * y_ev[k, 0] for k in range(len(t_ev))]
                    ax.scatter(map_y, map_vy, s=0.8, color=c, alpha=0.6)
            except Exception:
                pass
            progress.update(count)
    
    progress.finish()
    plt.show()

# =============================================================================
# 6. DATA EXPORT (ROTATING -> INERTIAL FRAME)
# =============================================================================

def analyze_snapshot(body1, body2):
    """Derives orbital elements and basis vectors from inertial snapshots."""
    r_rel = body2['r'] - body1['r']
    v_rel = body2['v'] - body1['v']
    dist = np.linalg.norm(r_rel)
    M_tot = body1['m'] + body2['m']
    
    # Barycenter and Basis Vectors
    r_bc = (body1['m']*body1['r'] + body2['m']*body2['r']) / M_tot
    v_bc = (body1['m']*body1['v'] + body2['m']*body2['v']) / M_tot
    
    u_hat = r_rel / dist
    h_vec = np.cross(r_rel, v_rel)
    h_hat = h_vec / np.linalg.norm(h_vec)
    v_hat = np.cross(h_hat, u_hat)
    
    r_dot = np.dot(v_rel, u_hat)
    Omega = np.dot(v_rel, v_hat) / dist
    n_mean = np.sqrt(6.674e-11 * M_tot / dist**3)
    
    return {'dist': dist, 'r_bc': r_bc, 'v_bc': v_bc, 'u_hat': u_hat, 'v_hat': v_hat, 
            'Omega': Omega, 'r_dot': r_dot, 'n_mean': n_mean}

def transform_to_inertial(state_dimless, sys_params):
    """Transforms state from ER3BP Rotating Frame to Inertial Cartesian Frame."""
    x, y, vx_puls, vy_puls = state_dimless['x'], state_dimless['y'], state_dimless['vx'], state_dimless['vy']
    L, r_dot, Omega = sys_params['dist'], sys_params['r_dot'], sys_params['Omega']
    u, v = sys_params['u_hat'], sys_params['v_hat']
    
    # Position Transformation
    pos_vec = sys_params['r_bc'] + L * (x * u + y * v)
    
    # Velocity Transformation (Expansion + Peculiar + Rotation)
    v_exp = r_dot * (x * u + y * v)
    v_peculiar = L * sys_params['n_mean'] * (vx_puls * u + vy_puls * v)
    v_rot = L * Omega * (x * v - y * u)
    vel_vec = sys_params['v_bc'] + v_exp + v_peculiar + v_rot
    
    return pos_vec, vel_vec

def run_file_generator():
    """Generates the N-Body initial condition file."""
    moon_data = ACTIVE_CONFIG['target_body']
    name = ACTIVE_CONFIG['name']
    
    print(f"\n[Export] Generating N-Body Data for: JUPITER-{name}")
    print("   >> Calculating Transformation Matrix (Snapshot Analysis)...")
    sys_params = analyze_snapshot(DATA_JUPITER, moon_data)
    
    # Define Grid of Candidates (L4 Vicinity + Tadpoles)
    candidates = []
    candidates.append({'x': X_L4, 'y': Y_L4, 'vx': 0.0, 'vy': 0.0, 'label': 'L4_Center'})
    for dy in [0.05, -0.05]:
        candidates.append({'x': X_L4, 'y': Y_L4 + dy, 'vx': 0.0, 'vy': 0.0, 'label': f'Libration_{dy}'})
    
    output_lines = []
    output_lines.append(f"# Initial data generated for {name} System")
    output_lines.append("# Columns: x y z vx vy vz mass")
    output_lines.append("# Units: SI (Meters, m/s, kg)\n")
    
    # 1. Write Primary Bodies
    print("   >> Writing Primary Bodies...")
    for n, body in [("Jupiter", DATA_JUPITER), (name, moon_data)]:
        r, v, m = body['r'], body['v'], body['m']
        output_lines.append(f"# {n}\n{r[0]:.16e} {r[1]:.16e} {r[2]:.16e} {v[0]:.16e} {v[1]:.16e} {v[2]:.16e} {m:.5e}\n")
    
    # 2. Write Background Bodies
    output_lines.append(STR_IO + "\n")
    
    # 3. Transform and Write Spacecraft
    print(f"   >> Transforming {len(candidates)} Spacecraft states...")
    count = 0
    for cand in candidates:
        count += 1
        r_sc, v_sc = transform_to_inertial(cand, sys_params)
        output_lines.append(f"# SC_{count}: {cand['label']}")
        output_lines.append(f"{r_sc[0]:.16e} {r_sc[1]:.16e} {r_sc[2]:.16e} {v_sc[0]:.16e} {v_sc[1]:.16e} {v_sc[2]:.16e} 1000.0\n")
        
    filename = f'CondInicSim_{name}.txt'
    with open(filename, 'w') as f:
        f.writelines(output_lines)
    
    print(f"✅ SUCCESS: File saved as '{filename}'")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("      ER3BP UNIFIED LABORATORY")
    print("      Dynamical Systems Analysis Tool")
    print("="*50)
    print("\nSELECT TARGET SYSTEM:")
    print("  1. Jupiter - Europa")
    print("  2. Jupiter - Ganymede")
    
    sys_choice = input("\n>> Select System (1-2): ")
    setup_system(sys_choice)
    
    while True:
        print(f"\n[{ACTIVE_CONFIG['name']}] AVAILABLE TOOLS:")
        print("  1. Poincaré Map (Analyze Stability)")
        print("  2. Visual Trajectory 2D/3D (Single Orbit)")
        print("  3. GENERATE N-BODY FILE (Export Data)")
        print("  0. Exit")
        
        choice = input("\n>> Select Tool: ")
        
        if choice == '1': 
            run_poincare_map()
        elif choice == '2': 
            run_single_trajectory_vis()
        elif choice == '3': 
            run_file_generator()
        elif choice == '0': 
            print("\nExiting... Goodbye.")
            break
        else:
            print(" [!] Invalid selection.")