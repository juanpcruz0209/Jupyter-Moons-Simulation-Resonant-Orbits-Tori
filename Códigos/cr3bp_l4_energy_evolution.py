import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib as mpl

# =============================================================================
# 1. VISUAL STYLE & CONFIGURATION
# =============================================================================

jupiter_style = {
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
    'savefig.facecolor': '#000000',
    'text.color': '#E0E0E0',
    'axes.labelcolor': '#E0E0E0',
    'xtick.color': '#E0E0E0',
    'ytick.color': '#E0E0E0',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.grid': True,
    'grid.color': '#444444',
    'grid.linestyle': ':',
    'grid.alpha': 0.5,
    'lines.linewidth': 0.8,
    'legend.facecolor': '#1A1A1A',
    'legend.edgecolor': '#E0E0E0',
    'legend.framealpha': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#666666',
}
plt.rcParams.update(jupiter_style)

colors = {
    'jupiter': '#F4A460',
    'europa': '#E67E22',
    'ganymede': '#A6D96A',
    'spacecraft': '#FDFEFE',
    'orbit_trace': '#555555',
    'l4_marker': '#FF3333'
}

# =============================================================================
# 2. PHYSICAL CONSTANTS
# =============================================================================

MU = 2.526e-5
X_L4 = 0.5 - MU
Y_L4 = np.sqrt(3) / 2

# Energy Range (C)
C_START = 2.998
C_END = 3.002
NUM_LEVELS = 10 

# Integration
NUM_PERIODS = 1200 
T_MAX = NUM_PERIODS * 2 * np.pi
RTOL, ATOL = 1e-9, 1e-11
MAX_STEP = 0.5

# =============================================================================
# 3. UTILITIES
# =============================================================================

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

# =============================================================================
# 4. PHYSICS KERNEL
# =============================================================================

def equations_of_motion(t, state, mu):
    x, y, px, py = state
    r1_sq = max((x + mu)**2 + y**2, 1e-9)
    r2_sq = max((x - (1 - mu))**2 + y**2, 1e-9)
    r1_c = r1_sq * np.sqrt(r1_sq); r2_c = r2_sq * np.sqrt(r2_sq)
    
    dx = px + y
    dy = py - x
    dpx = py - ((1-mu)*(x+mu))/r1_c - (mu*(x-(1-mu)))/r2_c
    dpy = -px - ((1-mu)*y)/r1_c - (mu*y)/r2_c
    return [dx, dy, dpx, dpy]

def get_vx_for_energy(x, y, C, mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    omega = 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2
    v_sq = 2 * omega - C
    return np.sqrt(v_sq) if v_sq >= 0 else None

def crossing_L4_plane(t, y, mu): return y[0] - X_L4
crossing_L4_plane.direction = 0

def terminators(t, y, mu):
    r_e = np.sqrt((y[0]-(1-mu))**2 + y[1]**2)
    r_j = np.sqrt((y[0]+mu)**2 + y[1]**2)
    return min(r_e-0.005, r_j-0.04, 3.0-np.sqrt(y[0]**2+y[1]**2))
terminators.terminal = True

# =============================================================================
# 5. DRAWING HELPERS (SYSTEM CONTEXT)
# =============================================================================

def draw_system_context_2d(ax):
    ax.axvline(0, color=colors['jupiter'], linestyle='-', alpha=0.3, linewidth=0.5)
    ax.plot([Y_L4], [0], '+', color=colors['l4_marker'], markersize=15, markeredgewidth=2, label='L4')

def draw_system_context_3d(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    z_line = np.linspace(C_START, C_END, 100)
    y_line = np.full_like(z_line, Y_L4)
    x_line = np.zeros_like(z_line) 
    
    ax.plot(y_line, x_line, z_line, '--', color=colors['l4_marker'], alpha=0.5, label='L4 Center Line')

    y_jup = np.zeros_like(z_line)
    ax.plot(y_jup, x_line, z_line, '-', color=colors['jupiter'], alpha=0.3, linewidth=2, label='Jupiter Plane (Y=0)')

# =============================================================================
# 6. DATA COMPUTATION
# =============================================================================

def compute_energy_stack():
    c_levels = np.linspace(C_START, C_END, NUM_LEVELS) 
    stack_data = []
    y_scan = np.linspace(Y_L4 - 0.15, Y_L4 + 0.15, 40) 
    
    total_ops = len(c_levels) * len(y_scan)
    current_op = 0
    
    print(f"\n--- Computing {NUM_LEVELS} Energy Slices ---")
    print_progress_bar(0, total_ops, prefix='Progress:', suffix='Complete', length=40)
    
    for idx, C_val in enumerate(c_levels):
        slice_y = []
        slice_vy = []
        
        for y_s in y_scan:
            current_op += 1
            print_progress_bar(current_op, total_ops, prefix='Progress:', suffix=f'(C={C_val:.4f})', length=40)
            
            vx_geo = get_vx_for_energy(X_L4, y_s, C_val, MU)
            if vx_geo is None: continue
            
            for direction in [1, -1]:
                vx = vx_geo * direction
                state = [X_L4 + 1e-5, y_s, vx - y_s, 0.0 + X_L4]
                
                try:
                    sol = solve_ivp(
                        equations_of_motion, (0, T_MAX), state, args=(MU,),
                        method='DOP853', events=[crossing_L4_plane, terminators],
                        rtol=RTOL, atol=ATOL, max_step=MAX_STEP
                    )
                    if len(sol.y_events[0]) > 0:
                        crossings = sol.y_events[0]
                        slice_y.extend(crossings[:, 1])
                        slice_vy.extend(crossings[:, 3] - crossings[:, 0])
                except: continue

        stack_data.append({'C': C_val, 'y': np.array(slice_y), 'vy': np.array(slice_vy)})
        
    return stack_data

def compute_3d_trajectory(C_val=3.0005):
    print(f"\n--- Computing 3D Trajectory (C={C_val}) ---")
    y_start = Y_L4 + 0.035
    vx_start = get_vx_for_energy(X_L4, y_start, C_val, MU)
    
    if vx_start is None:
        print("Error: Forbidden region.")
        return None

    px = vx_start - y_start
    py = 0.0 + X_L4
    state = [X_L4, y_start, px, py]
    
    t_span = (0, 1000 * np.pi)
    t_eval = np.linspace(0, t_span[1], 100000)

    sol = solve_ivp(
        equations_of_motion, t_span, state, args=(MU,),
        method='DOP853', rtol=RTOL, atol=ATOL, max_step=0.1, t_eval=t_eval
    )
    
    return sol

def compute_3d_trajectory_two_particles(C_val=3.0005, perturbation=1e-6):
    """Calcula dos trayectorias idénticas con una pequeña perturbación inicial."""
    print(f"\n--- Computing TWO Trajectories (C={C_val}, Perturbation={perturbation:.2e}) ---")
    
    y_start_base = Y_L4 + 0.035
    
    # --- Partícula 1 (Referencia) ---
    y_start1 = y_start_base
    vx_start1 = get_vx_for_energy(X_L4, y_start1, C_val, MU)
    if vx_start1 is None: return None, None
    state1 = [X_L4, y_start1, vx_start1 - y_start1, 0.0 + X_L4]

    # --- Partícula 2 (Perturbada) ---
    y_start2 = y_start_base + perturbation # Pequeña diferencia en Y
    vx_start2 = get_vx_for_energy(X_L4, y_start2, C_val, MU)
    if vx_start2 is None: return None, None
    state2 = [X_L4, y_start2, vx_start2 - y_start2, 0.0 + X_L4]
    
    t_span = (0, 1000 * np.pi) 
    t_eval = np.linspace(0, t_span[1], 100000)

    sol1 = solve_ivp(
        equations_of_motion, t_span, state1, args=(MU,),
        method='DOP853', rtol=RTOL, atol=ATOL, max_step=0.1, t_eval=t_eval
    )
    sol2 = solve_ivp(
        equations_of_motion, t_span, state2, args=(MU,),
        method='DOP853', rtol=RTOL, atol=ATOL, max_step=0.1, t_eval=t_eval
    )
    
    return sol1, sol2

# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def plot_3d_stack(data):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(right=0.85) 
    draw_system_context_3d(ax)
    
    ax.set_title("Energy Evolution of L4 Invariant Tori\n(Vertical Axis = Jacobi Constant C)")
    ax.set_xlabel("Y Position")
    ax.set_ylabel("Vertical Velocity (Vy)")
    ax.set_zlabel("Energy (C)", labelpad=15)
    
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    for i, slice_d in enumerate(data):
        c_val = slice_d['C']
        ys = slice_d['y']
        vys = slice_d['vy']
        cs = np.full_like(ys, c_val)
        if len(ys) > 0:
            ax.scatter(ys, vys, cs, s=2, color=colors_map[i], alpha=0.6)
            
    plt.legend(loc='upper left')
    plt.show()

def animate_energy_sweep(data, save_to_file=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    
    all_ys = np.concatenate([d['y'] for d in data if len(d['y']) > 0])
    all_vys = np.concatenate([d['vy'] for d in data if len(d['vy']) > 0])

    if len(all_ys) > 0:
        y_min, y_max = np.min(all_ys), np.max(all_ys)
        vy_min, vy_max = np.min(all_vys), np.max(all_vys)
        pad_y = (y_max - y_min) * 0.1
        pad_vy = (vy_max - vy_min) * 0.1
        ax.set_xlim(y_min - pad_y, y_max + pad_y)
        ax.set_ylim(vy_min - pad_vy, vy_max + pad_vy)
    else:
        ax.set_xlim(Y_L4 - 0.2, Y_L4 + 0.2)
        ax.set_ylim(-0.2, 0.2)

    ax.set_xlabel("Y Position")
    ax.set_ylabel("Vertical Velocity (Vy)")
    ax.grid(True, alpha=0.2)
    
    draw_system_context_2d(ax)

    scatter = ax.scatter([], [], s=3, color=colors['spacecraft'])
    title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", fontsize=12, color='white')
    
    def update(frame_idx):
        slice_d = data[frame_idx]
        c_val = slice_d['C']
        if len(slice_d['y']) > 0:
            points = np.column_stack((slice_d['y'], slice_d['vy']))
            scatter.set_offsets(points)
            cols = plt.cm.plasma((slice_d['vy'] + 0.1)/0.2) 
            scatter.set_color(cols)
        else:
            scatter.set_offsets(np.empty((0, 2)))
            
        title_text.set_text(f"Energy Level C = {c_val:.5f}")
        return scatter, title_text

    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=10, blit=False)
    
    if save_to_file:
        print(f"\nGuardando animación como GIF en: {save_to_file}...")
        # FPS bajo porque son pocas capas de energía
        ani.save(save_to_file, writer='pillow', fps=2, savefig_kwargs={'facecolor': '#000000'})
        print("¡Guardado completado!")
    else:
        print("Mostrando animación... (Cierre la ventana para salir)")
        plt.show()

def animate_3d_trajectory(sol, save_to_file=None):
    if sol is None: return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # Plot del sistema (Jupiter, Europa, L4, etc.)
    ax.scatter([-MU], [0], [0], s=200, c=colors['jupiter'], label='Jupiter', edgecolors='none')
    ax.scatter([1-MU], [0], [0], s=50, c=colors['europa'], label='Europa', edgecolors='none')
    theta = np.linspace(0, 2*np.pi, 100)
    z_ring = np.zeros_like(theta)
    x_eu = -MU + 1.0 * np.cos(theta); y_eu = 1.0 * np.sin(theta)
    ax.plot(x_eu, y_eu, z_ring, '--', c=colors['europa'], linewidth=0.5, alpha=0.3)
    r_gan = 1.59
    x_gan = -MU + r_gan * np.cos(theta); y_gan = r_gan * np.sin(theta)
    ax.plot(x_gan, y_gan, z_ring, ':', c=colors['ganymede'], linewidth=0.5, alpha=0.4, label='Ganymede Orbit')
    ax.scatter([X_L4], [Y_L4], [0], marker='+', color=colors['l4_marker'], s=100, label='L4')

    # Datos de la trayectoria
    x = sol.y[0]
    y = sol.y[1]
    z = np.zeros_like(x) 

    # Objetos a animar: la línea de la trayectoria y el punto actual
    line, = ax.plot([], [], [], color=colors['spacecraft'], linewidth=1.5, label='Spacecraft')
    point, = ax.plot([], [], [], 'o', color=colors['spacecraft'], markersize=4)

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-0.1, 0.1) 
    ax.set_xlabel("X Position"); ax.set_ylabel("Y Position"); ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectory Animation (Physical Space)")
    ax.legend()
    
    # ----------------------------------------------------
    # --> FUNCIÓN UPDATE FALTANTE <--
    def update(frame):
        # La línea acumula la trayectoria hasta el frame actual
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        # El punto solo marca la posición actual del satélite
        point.set_data(x[frame:frame+1], y[frame:frame+1])
        point.set_3d_properties(z[frame:frame+1])
        return line, point
    # ----------------------------------------------------

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0,len(x),100), interval=10, blit=False)
    
    if save_to_file:
        print(f"\nGuardando animación de 1 partícula como GIF en: {save_to_file}...")
        ani.save(save_to_file, writer='pillow', fps=30, savefig_kwargs={'facecolor': '#000000'})
        print("¡Guardado completado!")
    else:
        print("Mostrando animación 3D... (Cierre la ventana para salir)")
        plt.show()


def animate_3d_trajectory_two_particles(sol1, sol2, save_to_file=None):
    if sol1 is None or sol2 is None: return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ... (Configuración de paneles y planetas - igual que en animate_3d_trajectory) ...
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.scatter([-MU], [0], [0], s=200, c=colors['jupiter'], label='Jupiter', edgecolors='none')
    ax.scatter([1-MU], [0], [0], s=50, c=colors['europa'], label='Europa', edgecolors='none')

    theta = np.linspace(0, 2*np.pi, 100); z_ring = np.zeros_like(theta)
    x_eu = -MU + 1.0 * np.cos(theta); y_eu = 1.0 * np.sin(theta)
    ax.plot(x_eu, y_eu, z_ring, '--', c=colors['europa'], linewidth=0.5, alpha=0.3)
    ax.scatter([X_L4], [Y_L4], [0], marker='+', color=colors['l4_marker'], s=100, label='L4')

    # Data for Particle 1 (Reference)
    x1, y1 = sol1.y[0], sol1.y[1]
    z1 = np.zeros_like(x1) 
    line1, = ax.plot([], [], [], color=colors['spacecraft'], linewidth=1.0, alpha=0.8, label='P1 (Reference)')
    point1, = ax.plot([], [], [], 'o', color=colors['spacecraft'], markersize=4)

    # Data for Particle 2 (Perturbed)
    x2, y2 = sol2.y[0], sol2.y[1]
    z2 = np.zeros_like(x2) 
    # Usar un color distinto, por ejemplo, el de Ganímedes, para contraste
    line2, = ax.plot([], [], [], color=colors['ganymede'], linestyle=':', linewidth=1.0, alpha=0.8, label='P2 (Perturbed)')
    point2, = ax.plot([], [], [], '^', color=colors['ganymede'], markersize=4)


    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-0.1, 0.1) 
    ax.set_xlabel("X Position"); ax.set_ylabel("Y Position"); ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectory Divergence (Two Particles)")
    ax.legend()

    def update(frame):
        # Particle 1
        line1.set_data(x1[:frame], y1[:frame]); line1.set_3d_properties(z1[:frame])
        point1.set_data(x1[frame:frame+1], y1[frame:frame+1]); point1.set_3d_properties(z1[frame:frame+1])
        # Particle 2
        line2.set_data(x2[:frame], y2[:frame]); line2.set_3d_properties(z2[:frame])
        point2.set_data(x2[frame:frame+1], y2[frame:frame+1]); point2.set_3d_properties(z2[frame:frame+1])
        
        return line1, point1, line2, point2

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, len(x1), 100), interval=20, blit=False)
    
    if save_to_file:
        print(f"\nGuardando animación doble como GIF en: {save_to_file}...")
        ani.save(save_to_file, writer='pillow', fps=30, savefig_kwargs={'facecolor': '#000000'})
        print("¡Guardado completado!")
    else:
        print("Mostrando animación 3D de divergencia... (Cierre la ventana para salir)")
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    stack_data_cache = None 

    while True:
        print("\n--- MENU PRINCIPAL ---")
        print("1. Plot Estático 3D (Evolución de Energía)")
        print("2. Ver Animación 2D (Barrido de Energía)")
        print("3. GUARDAR Animación 2D como GIF")
        print("4. Ver Animación 3D (Trayectoria - 1 Partícula)")
        print("5. GUARDAR Animación 3D como GIF (1 Partícula)")
        print("6. Ver Animación 3D (Trayectoria - 2 Partículas)") # NUEVA OPCIÓN
        print("7. GUARDAR Animación 3D como GIF (2 Partículas)") # NUEVA OPCIÓN
        print("0. Salir")
        
        choice = input("Selecciona [0-7]: ")
        
        if choice == '1':
            if stack_data_cache is None: stack_data_cache = compute_energy_stack()
            plot_3d_stack(stack_data_cache)
        elif choice == '2':
            if stack_data_cache is None: stack_data_cache = compute_energy_stack()
            animate_energy_sweep(stack_data_cache, save_to_file=None)
        elif choice == '3':
            if stack_data_cache is None: stack_data_cache = compute_energy_stack()
            filename = f"energy_sweep_{int(time.time())}.gif"
            animate_energy_sweep(stack_data_cache, save_to_file=filename)
        elif choice == '4':
            sol = compute_3d_trajectory()
            animate_3d_trajectory(sol, save_to_file=None)
        elif choice == '5':
            sol = compute_3d_trajectory()
            filename = f"trajectory_3d_1P_{int(time.time())}.gif"
            animate_3d_trajectory(sol, save_to_file=filename)
        elif choice == '6': # NUEVA OPCIÓN
            sol1, sol2 = compute_3d_trajectory_two_particles()
            animate_3d_trajectory_two_particles(sol1, sol2, save_to_file=None)
        elif choice == '7': # NUEVA OPCIÓN
            sol1, sol2 = compute_3d_trajectory_two_particles()
            filename = f"trajectory_3d_2P_Divergence_{int(time.time())}.gif"
            animate_3d_trajectory_two_particles(sol1, sol2, save_to_file=filename)
        elif choice == '0':
            print("Saliendo.")
            break
        else:
            print("Selección inválida.")