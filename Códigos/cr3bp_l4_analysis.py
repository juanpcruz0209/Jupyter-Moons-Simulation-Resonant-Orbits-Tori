import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# =============================================================================
# 1. VISUAL STYLE & CONFIGURATION
# =============================================================================

# Define the custom "Jupiter Mission" style
jupiter_style = {
    # Backgrounds
    'figure.facecolor': '#000000',  # Pitch black background
    'axes.facecolor': '#000000',    # Pitch black axes
    'savefig.facecolor': '#000000', # Save with black background
    
    # Text and Fonts
    'text.color': '#E0E0E0',        # Off-white text
    'axes.labelcolor': '#E0E0E0',
    'xtick.color': '#E0E0E0',
    'ytick.color': '#E0E0E0',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    
    # Grid and Lines
    'axes.grid': True,
    'grid.color': '#444444',        # Subtle gray grid
    'grid.linestyle': ':',          # Dotted lines
    'grid.alpha': 0.5,              # Transparency
    'lines.linewidth': 0.8,         # Thin, precise lines
    
    # Legend
    'legend.facecolor': '#1A1A1A',  # Dark gray legend box
    'legend.edgecolor': '#E0E0E0',
    'legend.framealpha': 0.8,
    
    # Spines (The box around the plot)
    'axes.spines.top': False,       # Clean look
    'axes.spines.right': False,
    'axes.edgecolor': '#666666',
}

# Apply the style
plt.rcParams.update(jupiter_style)

# Custom Color Palette (Based on your slides)
colors = {
    'jupiter': '#F4A460',    # Sandy Orange (Jupiter)
    'europa': '#E67E22',     # Darker Orange/Red (Europa)
    'ganymede': '#A6D96A',   # Olive/Greenish (Ganymede)
    'spacecraft': '#FDFEFE', # White/Cream (Spacecraft)
    'orbit_trace': '#555555', # Faint gray for orbit trails
    'l4_marker': '#FF3333'   # Red for L4 marker
}

# =============================================================================
# 2. PHYSICAL CONSTANTS
# =============================================================================

# Mass Parameter for Jupiter-Europa
# mu = M2 / (M1 + M2)
MU = 2.526e-5  

# Theoretical Location of Lagrangian Point L4
# Forms an equilateral triangle with the primaries.
X_L4 = 0.5 - MU
Y_L4 = np.sqrt(3) / 2

# Integrator Tolerances
RTOL = 1e-9
ATOL = 1e-11
MAX_STEP = 0.1 

# =============================================================================
# 3. UTILITIES (PROGRESS BAR)
# =============================================================================

class ProgressBar:
    """
    A simple, self-contained progress bar for console output.
    """
    def __init__(self, total, prefix='Progress', length=40):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.start_time = time.time()

    def update(self, current):
        """Updates the progress bar visuals."""
        elapsed_time = time.time() - self.start_time
        percent = float(current) * 100 / self.total
        filled_length = int(self.length * current // self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        # Estimate time remaining
        if current > 0:
            time_per_item = elapsed_time / current
            remaining_items = self.total - current
            eta = remaining_items * time_per_item
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "?"

        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:.1f}% Complete (ETA: {eta_str})')
        sys.stdout.flush()

    def finish(self):
        """Cleans up the progress bar line."""
        sys.stdout.write('\n')

# =============================================================================
# 4. PHYSICS KERNEL
# =============================================================================

def equations_of_motion(t, state, mu):
    """
    Hamiltonian equations of motion for the CR3BP in the synodic (rotating) frame.
    State vector Y: [x, y, px, py] where (px, py) are canonical momenta.
    """
    x, y, px, py = state
    
    r1_sq = (x + mu)**2 + y**2
    r2_sq = (x - (1 - mu))**2 + y**2
    
    r1_sq = max(r1_sq, 1e-9)
    r2_sq = max(r2_sq, 1e-9)

    r1_cubed = r1_sq * np.sqrt(r1_sq)
    r2_cubed = r2_sq * np.sqrt(r2_sq)
    
    dx_dt = px + y
    dy_dt = py - x
    
    dpx_dt = py - ((1 - mu) * (x + mu)) / r1_cubed - (mu * (x - (1 - mu))) / r2_cubed
    dpy_dt = -px - ((1 - mu) * y) / r1_cubed - (mu * y) / r2_cubed
    
    return [dx_dt, dy_dt, dpx_dt, dpy_dt]


def get_vx_for_energy(x, y, C, mu):
    """
    Inverse Jacobi calculation. Returns required Vx for energy C.
    """
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    
    omega = 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2
    v_sq_target = 2 * omega - C
    
    if v_sq_target < 0:
        return None 
    
    return np.sqrt(v_sq_target)

# =============================================================================
# 5. EVENT DETECTORS
# =============================================================================

def crossing_L4_plane(t, y, mu):
    return y[0] - X_L4
crossing_L4_plane.direction = 0 

def crash_jupiter(t, y, mu):
    r = np.sqrt((y[0] + mu)**2 + y[1]**2)
    return r - 0.04 
crash_jupiter.terminal = True

def crash_europa(t, y, mu):
    r = np.sqrt((y[0] - (1 - mu))**2 + y[1]**2)
    return r - 0.005
crash_europa.terminal = True

def escape_system(t, y, mu):
    r = np.sqrt(y[0]**2 + y[1]**2)
    return 3.0 - r
escape_system.terminal = True

# =============================================================================
# 6. HELPER: DRAW CELESTIAL BODIES
# =============================================================================

def draw_system_context_2d(ax):
    """Adds Jupiter, Europa, and Ganymede (schematic) to a 2D plot."""
    # Jupiter at (-mu, 0)
    ax.plot([-MU], [0], 'o', color=colors['jupiter'], markersize=12, label='Jupiter')
    
    # Europa at (1-mu, 0)
    ax.plot([1-MU], [0], 'o', color=colors['europa'], markersize=6, label='Europa')
    
    # Europa Orbit (Circle radius ~1 around Jupiter)
    # Note: In rotating frame, primaries are fixed, but we draw orbit traces for context
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Europa Orbit Circle (approx r=1) centered on Jupiter
    x_eu = -MU + 1.0 * np.cos(theta)
    y_eu = 1.0 * np.sin(theta)
    ax.plot(x_eu, y_eu, '--', color=colors['europa'], linewidth=0.5, alpha=0.5)
    
    # Ganymede (approx r=1.6 relative to Europa)
    # Ganymede is NOT fixed in this frame, but we show its orbital path
    r_gan = 1.59  # Semi-major axis ratio Ganymede/Europa
    x_gan = -MU + r_gan * np.cos(theta)
    y_gan = r_gan * np.sin(theta)
    ax.plot(x_gan, y_gan, ':', color=colors['ganymede'], linewidth=0.5, alpha=0.5, label='Ganymede Orbit')


def draw_system_context_3d(ax):
    """Adds Jupiter, Europa, and Ganymede to a 3D plot with correct styling."""
    # 3D Panes cleanup for "Deep Space" look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    
    # Jupiter
    ax.scatter([-MU], [0], [0], s=200, c=colors['jupiter'], label='Jupiter', edgecolors='none')
    
    # Europa
    ax.scatter([1-MU], [0], [0], s=50, c=colors['europa'], label='Europa', edgecolors='none')
    
    # Orbital Paths
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Europa Path
    x_eu = -MU + 1.0 * np.cos(theta)
    y_eu = 1.0 * np.sin(theta)
    z_eu = np.zeros_like(theta)
    ax.plot(x_eu, y_eu, z_eu, '--', c=colors['europa'], linewidth=0.5, alpha=0.4)
    
    # Ganymede Path
    r_gan = 1.59
    x_gan = -MU + r_gan * np.cos(theta)
    y_gan = r_gan * np.sin(theta)
    ax.plot(x_gan, y_gan, z_eu, ':', c=colors['ganymede'], linewidth=0.5, alpha=0.4, label='Ganymede Orbit')

# =============================================================================
# 7. ANALYSIS MODES
# =============================================================================

def run_diagnostic_map():
    print("\n" + "="*60)
    print("RUNNING: DIAGNOSTIC POINCARE MAP (VARIABLE ENERGY)")
    print("="*60)
    
    # --- Configuration ---
    y_values = np.linspace(Y_L4 - 0.15, Y_L4 + 0.15, 40) 
    vx_values = np.linspace(-0.015, 0.015, 5)
    
    NUM_PERIODS = 1200
    t_max = NUM_PERIODS * 2 * np.pi
    
    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Diagnostic Poincaré Map @ L4 (Jupiter-Europa)\nVariable Energy Scan")
    ax.set_xlabel("Y Position (Rotating Frame)")
    ax.set_ylabel("Vertical Velocity (Vy)")
    
    # Plot L4
    ax.plot([Y_L4], [0], '+', color=colors['l4_marker'], markersize=15, markeredgewidth=2, label='L4 Equilibrium')
    
    # Note: Poincare maps are abstract phase space plots (Y vs Vy). 
    # Drawing physical bodies (Jupiter/Europa) here implies projecting them onto the section plane,
    # which usually isn't done for Poincare maps, but we can keep L4 marked clearly.
    
    colors_map = plt.cm.turbo(np.linspace(0, 1, len(y_values)))
    total_sims = len(y_values) * len(vx_values)
    
    # --- Simulation Loop ---
    progress = ProgressBar(total_sims, prefix='Simulating Orbits')
    sim_count = 0
    total_points = 0
    
    for i, y_start in enumerate(y_values):
        c = colors_map[i]
        for vx_start in vx_values:
            sim_count += 1
            
            curr_x = X_L4 + 1e-5
            curr_y = y_start
            px = vx_start - curr_y
            py = 0.0 + curr_x
            state = [curr_x, curr_y, px, py]
            
            try:
                sol = solve_ivp(
                    equations_of_motion, (0, t_max), state, args=(MU,),
                    method='DOP853', 
                    events=[crossing_L4_plane, crash_jupiter, crash_europa, escape_system],
                    rtol=RTOL, atol=ATOL, max_step=MAX_STEP
                )
                
                if len(sol.y_events[0]) > 0:
                    crossings = sol.y_events[0]
                    map_y = crossings[:, 1]
                    map_vy = crossings[:, 3] - crossings[:, 0]
                    ax.scatter(map_y, map_vy, s=1.0, color=c, alpha=0.6)
                    total_points += len(map_y)
                    
            except Exception:
                pass
            progress.update(sim_count)

    progress.finish()
    
    if total_points == 0:
        ax.text(Y_L4, 0, "NO STABLE ORBITS FOUND", ha='center', color='red')

    # No physical context drawn here because this is Phase Space (Y vs Vy), not Physical Space (X vs Y)
    ax.legend()
    plt.show()


def run_isoenergetic_map(target_C=3.0005):
    print("\n" + "="*60)
    print(f"RUNNING: ISO-ENERGETIC POINCARE MAP (C = {target_C})")
    print("="*60)
    
    # --- Configuration ---
    y_values = np.linspace(Y_L4 - 0.15, Y_L4 + 0.15, 40)
    NUM_PERIODS = 1200
    t_max = NUM_PERIODS * 2 * np.pi
    
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(f"Iso-Energetic Poincaré Map\nJacobi Constant C = {target_C}")
    ax.set_xlabel("Y Position")
    ax.set_ylabel("Vertical Velocity (Vy)")
    ax.plot([Y_L4], [0], '+', color=colors['l4_marker'], markersize=15, markeredgewidth=2, label='L4 Center')
    
    colors_map = plt.cm.plasma(np.linspace(0, 1, len(y_values)))
    total_sims = len(y_values) * 2 
    
    progress = ProgressBar(total_sims, prefix='Computing Layers')
    sim_count = 0
    
    for i, y_start in enumerate(y_values):
        vx_mag = get_vx_for_energy(X_L4, y_start, target_C, MU)
        
        if vx_mag is None:
            sim_count += 2
            progress.update(sim_count)
            continue 
        
        for direction in [1, -1]:
            sim_count += 1
            vx_initial = vx_mag * direction
            px = vx_initial - y_start
            py = 0.0 + (X_L4 + 1e-5)
            state = [X_L4 + 1e-5, y_start, px, py]
            
            sol = solve_ivp(
                equations_of_motion, (0, t_max), state, args=(MU,),
                method='DOP853', 
                events=[crossing_L4_plane, crash_jupiter, crash_europa, escape_system],
                rtol=RTOL, atol=ATOL, max_step=MAX_STEP
            )
            
            if len(sol.y_events[0]) > 0:
                crossings = sol.y_events[0]
                map_y = crossings[:, 1]
                map_vy = crossings[:, 3] - crossings[:, 0]
                ax.scatter(map_y, map_vy, s=0.8, color=colors_map[i], alpha=0.6)
                
            progress.update(sim_count)

    progress.finish()
    ax.legend()
    plt.show()


def run_3d_torus_visualization(target_C=3.0005):
    print("\n" + "="*60)
    print(f"RUNNING: 3D INVARIANT TORUS VISUALIZATION (C = {target_C})")
    print("="*60)
    
    # Use a stable initial condition
    y_start = Y_L4 + 0.035
    vx_start = get_vx_for_energy(X_L4, y_start, target_C, MU)
    
    if vx_start is None:
        print("[Error] Selected condition is energetically forbidden.")
        return

    print(f"Integrating trajectory from Y={y_start:.4f}...")
    
    px = vx_start - y_start
    py = 0.0 + X_L4
    state = [X_L4, y_start, px, py]
    
    # 3D Visualization requires long integration for visual impact
    t_span = (0, 600 * np.pi) 
    
    # Progress bar for single integration
    # Since solve_ivp is monolithic, we simulate "progress" by printing status
    print("Integration started (this may take a moment)...")
    
    start_t = time.time()
    sol = solve_ivp(
        equations_of_motion, t_span, state, args=(MU,),
        method='DOP853', rtol=RTOL, atol=ATOL, max_step=0.1
    )
    elapsed = time.time() - start_t
    print(f"Integration finished in {elapsed:.2f}s.")
    
    # Data Extraction
    x, y = sol.y[0], sol.y[1]
    px_arr = sol.y[2]
    vx = px_arr + y 
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 7))
    
    # Subplot 1: 2D Physical Space (left)
    # This is where we draw the bodies!
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("2D Physical Trajectory (X-Y Frame)")
    
    # DRAW CONTEXT BODIES
    draw_system_context_2d(ax1)
    
    # Draw Trajectory
    ax1.plot(x, y, linewidth=0.8, color=colors['spacecraft'], alpha=0.9, label='Particle Track')
    ax1.plot([X_L4], [Y_L4], '+', color=colors['l4_marker'], markersize=10, label='L4')
    
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.axis('equal')
    
    # Ensure we see L4 and the bodies
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    
    ax1.legend(loc='lower right', fontsize=8)

    # Subplot 2: 3D Phase Space (right)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("3D Phase Space Topology (X-Y-Vx)")
    
    # DRAW 3D CONTEXT
    draw_system_context_3d(ax2)
    
    # Scatter plot data (colored by time)
    p = ax2.scatter(x, y, vx, c=sol.t, cmap='plasma', s=0.3, alpha=0.4)
    
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Vx")
    
    # Draw Poincare Plane (transparent red)
    ys_plane = np.linspace(np.min(y), np.max(y), 10)
    zs_plane = np.linspace(np.min(vx), np.max(vx), 10)
    Y_plane, Z_plane = np.meshgrid(ys_plane, zs_plane)
    X_plane = np.full_like(Y_plane, X_L4)
    ax2.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='red')
    
    # Colorbar Fix
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(p, cax=cax, label='Time Evolution')
    
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print("   CR3BP DYNAMICS LABORATORY")
        print("   System: Jupiter-Europa (L4)")
        print("="*40)
        print("1. Diagnostic Map")
        print("2. Iso-Energetic Map")
        print("3. Invariant Torus Visualization (3D)")
        print("0. Exit")
        
        choice = input("\nSelect an option [0-3]: ")
        
        if choice == '1':
            run_diagnostic_map()
        elif choice == '2':
            run_isoenergetic_map()
        elif choice == '3':
            run_3d_torus_visualization()
        elif choice == '0':
            break
        else:
            print("Invalid selection.")