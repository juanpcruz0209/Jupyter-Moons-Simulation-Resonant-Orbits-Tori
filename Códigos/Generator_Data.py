import sys
import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# 1. CONFIGURACIÓN GLOBAL Y DATOS
# =============================================================================

# Diccionario que guardará la configuración activa (se llena al elegir sistema)
ACTIVE_CONFIG = {
    'name': 'UNKNOWN',
    'mu': 0,
    'epsilon': 0,
    'L_dist': 1,      # Distancia característica (m)
    'R_jup_dim': 0.04, # Radio de colisión Júpiter (adimensional)
    'R_moon_dim': 0.01 # Radio de colisión Luna (adimensional)
}

# --- JUPITER (Cuerpo Central Común) ---
DATA_JUPITER = {
    'r': np.array([-9.575324576133315e4, -6.150456912510757e4, -3.072242063097825e3]),
    'v': np.array([4.898609412767204e-1, -9.537003839258596e-1, -3.028640371889238e-2]),
    'm': 1.89818e27,
    'radius_km': 71492.0
}

# --- EUROPA ---
DATA_EUROPA = {
    'r': np.array([3.904990291276731e8, -5.392340224863521e8, -9.037579559937003e6]),
    'v': np.array([1.116678537182261e4, 8.174989252496797e3, 5.127064307826501e2]),
    'm': 4.79870e22,
    'epsilon': 0.0094,
    'radius_km': 1560.8
}

# --- GANYMEDE
DATA_GANYMEDE = {
    'r': np.array([8.626133793413050e8, -6.313676115858576e8, -1.178962605631183e4]),
    'v': np.array([6.431976727831559e3, 8.781079262403239e3, 4.266350959168634e2]),
    'm': 1.48157e23,
    'epsilon': 0.0015, # Excentricidad aprox de Ganímedes
    'radius_km': 2634.1
}

# Cadenas de texto para cuerpos de fondo (se usarán según corresponda)
STR_IO = """# Initial condition for Io
-1.169290164457216e7  4.206724877968931e8  1.469934463084565e7 -1.735159563649886e4 -5.367255748605908e2 -2.666674111749237e2 4.79872e22"""

# Estas funciones formatean los datos para escribirlos como "fondo" si no son el cuerpo principal
def format_body_string(name, data):
    r, v, m = data['r'], data['v'], data['m']
    return f"# Initial condition for {name}\n{r[0]:.16e} {r[1]:.16e} {r[2]:.16e} {v[0]:.16e} {v[1]:.16e} {v[2]:.16e} {m:.5e}"

# =============================================================================
# 2. LÓGICA DE SELECCIÓN DE SISTEMA
# =============================================================================
def setup_system(choice):
    """Configura las constantes globales basadas en la elección del usuario."""
    global DATA_SECONDARY, UNUSED_BODIES_STR
    
    if choice == '1':
        DATA_SECONDARY = DATA_EUROPA
        name = "EUROPA"
        # Si usamos Europa, Ganímedes pasa al fondo
        UNUSED_BODIES_STR = [STR_IO, format_body_string("Ganymede", DATA_GANYMEDE)]
    elif choice == '2':
        DATA_SECONDARY = DATA_GANYMEDE
        name = "GANYMEDE"
        # Si usamos Ganímedes, Europa pasa al fondo
        UNUSED_BODIES_STR = [STR_IO, format_body_string("Europa", DATA_EUROPA)]
    else:
        print("Opción inválida. Usando EUROPA por defecto.")
        return setup_system('1')

    # 1. Calcular Distancia Característica (L)
    r_rel = DATA_SECONDARY['r'] - DATA_JUPITER['r']
    dist_L = np.linalg.norm(r_rel)

    # 2. Calcular Parámetros Adimensionales
    mu_val = DATA_SECONDARY['m'] / (DATA_JUPITER['m'] + DATA_SECONDARY['m'])
    
    # 3. Radios de Colisión (Dinámicos)
    # Radio Dim = (Radio_km * 1000) / L_metros
    # Se agrega un pequeño buffer de seguridad
    r_jup_dim = (DATA_JUPITER['radius_km'] * 1000) / dist_L 
    r_moon_dim = (DATA_SECONDARY['radius_km'] * 1000) / dist_L 

    # 4. Actualizar Configuración Activa
    ACTIVE_CONFIG['name'] = name
    ACTIVE_CONFIG['mu'] = mu_val
    ACTIVE_CONFIG['epsilon'] = DATA_SECONDARY['epsilon']
    ACTIVE_CONFIG['L_dist'] = dist_L
    ACTIVE_CONFIG['R_jup_dim'] = r_jup_dim
    ACTIVE_CONFIG['R_moon_dim'] = r_moon_dim

    # 5. Definir Puntos de Lagrange globales para este sistema
    global X_L4, Y_L4
    X_L4 = 0.5 - mu_val
    Y_L4 = np.sqrt(3) / 2

    print(f"\n✅ SISTEMA CONFIGURADO: JÚPITER - {name}")
    print(f"   ├── Mu: {mu_val:.6e}")
    print(f"   ├── Epsilon: {ACTIVE_CONFIG['epsilon']}")
    print(f"   ├── Radio Colisión Júpiter (dim): {r_jup_dim:.4f}")
    print(f"   └── Radio Colisión Luna (dim):    {r_moon_dim:.4f}")

# =============================================================================
# 3. MOTOR DE FÍSICA (ER3BP & KEPLER)
# =============================================================================
def solve_kepler(M, e, tol=1e-10):
    E = M
    for _ in range(20):
        val = E - e * np.sin(E) - M
        der = 1 - e * np.cos(E)
        E = E - val / der
        if abs(val/der) < tol: break
    return E

def get_n_t(t, eps):
    E = solve_kepler(t, eps)
    rho = 1 - eps * np.cos(E)
    return np.sqrt(1 - eps**2) / (rho**2)

def equations_of_motion_hamiltonian(t, state, mu, eps):
    x, y, px, py = state
    E = solve_kepler(t, eps)
    rho = 1 - eps * np.cos(E)
    n_t = np.sqrt(1 - eps**2) / (rho**2)
    
    dx1, dx2 = x + mu*rho, x - (1-mu)*rho
    r1 = np.sqrt(dx1**2 + y**2)
    r2 = np.sqrt(dx2**2 + y**2)
    r1, r2 = max(r1, 1e-4), max(r2, 1e-4)
    
    dx_dt = px + y*n_t
    dy_dt = py - x*n_t
    dpx_dt = n_t*py - (1-mu)*dx1/r1**3 - mu*dx2/r2**3
    dpy_dt = -n_t*px - (1-mu)*y/r1**3 - mu*y/r2**3
    return [dx_dt, dy_dt, dpx_dt, dpy_dt]

def velocity_to_momentum(x, y, vx, vy, t, eps):
    n_t = get_n_t(t, eps)
    return vx - n_t*y, vy + n_t*x

# =============================================================================
# 4. EVENTOS 
# =============================================================================
def crash_jupiter(t, s, m, e): 
    # Usa el radio calculado dinámicamente en setup_system
    r_jup = ACTIVE_CONFIG['R_jup_dim']
    return np.sqrt((s[0]+m)**2 + s[1]**2) - r_jup
crash_jupiter.terminal = True
crash_jupiter.direction = 0

def crash_moon(t, s, m, e): 
    # Usa el radio calculado dinámicamente en setup_system
    r_moon = ACTIVE_CONFIG['R_moon_dim']
    return np.sqrt((s[0]-(1-m))**2 + s[1]**2) - r_moon
crash_moon.terminal = True
crash_moon.direction = 0

def escape_system(t, s, m, e): 
    return 3.5 - np.sqrt(s[0]**2 + s[1]**2)
escape_system.terminal = True
escape_system.direction = 0

EVENTS_LIST = [crash_jupiter, crash_moon, escape_system]

# =============================================================================
# 5. TRANSFORMACIÓN DE COORDENADAS
# =============================================================================
def analyze_snapshot(body1, body2):
    r_rel = body2['r'] - body1['r']
    v_rel = body2['v'] - body1['v']
    dist = np.linalg.norm(r_rel)
    M_tot = body1['m'] + body2['m']
    
    r_bc = (body1['m']*body1['r'] + body2['m']*body2['r']) / M_tot
    v_bc = (body1['m']*body1['v'] + body2['m']*body2['v']) / M_tot
    
    u_hat = r_rel / dist
    h_vec = np.cross(r_rel, v_rel); h_hat = h_vec / np.linalg.norm(h_vec)
    v_hat = np.cross(h_hat, u_hat)
    
    r_dot = np.dot(v_rel, u_hat)
    Omega = np.dot(v_rel, v_hat) / dist
    n_mean = np.sqrt(6.674e-11 * M_tot / dist**3)
    
    return {'dist': dist, 'r_bc': r_bc, 'v_bc': v_bc, 'u_hat': u_hat, 'v_hat': v_hat, 
            'Omega': Omega, 'r_dot': r_dot, 'n_mean': n_mean}

def transform_to_inertial(x, y, vx_puls, vy_puls, sys_params):
    L, r_dot, Omega = sys_params['dist'], sys_params['r_dot'], sys_params['Omega']
    u, v = sys_params['u_hat'], sys_params['v_hat']
    
    pos_vec = sys_params['r_bc'] + L * (x * u + y * v)
    v_exp = r_dot * (x * u + y * v)
    v_peculiar = L * sys_params['n_mean'] * (vx_puls * u + vy_puls * v)
    v_rot = L * Omega * (x * v - y * u)
    vel_vec = sys_params['v_bc'] + v_exp + v_peculiar + v_rot
    
    return pos_vec, vel_vec

# =============================================================================
# 6. GENERADOR PRINCIPAL
# =============================================================================
def run_integrated_file_generator():
    # Recuperar constantes activas
    mu = ACTIVE_CONFIG['mu']
    eps = ACTIVE_CONFIG['epsilon']
    sys_name = ACTIVE_CONFIG['name']
    
    print("\n" + "="*60)
    print(f"GENERANDO CONDICIONES ESTABLES PARA: {sys_name}")
    print("Escaneando cuadrícula alrededor de L4...")
    print("="*60)
    
    # Configuración de búsqueda
    y_search = np.linspace(Y_L4 - 0.15, Y_L4 + 0.15, 40) 
    vx_search = np.linspace(-0.015, 0.015, 5)
    TEST_DURATION = 500 * 2 * np.pi 
    
    # Obtener parámetros de transformación para el sistema actual
    sys_params = analyze_snapshot(DATA_JUPITER, DATA_SECONDARY)
    stable_candidates = []
    
    count = 0
    total_sims = len(y_search) * len(vx_search)
    
    for vx in vx_search:
        for y in y_search:
            count += 1
            # Integración
            px, py = velocity_to_momentum(X_L4, y, vx, 0.0, 0.0, eps)
            state = [X_L4, y, px, py]
            sim_id = f"Sim {count}/{total_sims} | Y={y:.3f}, Vx={vx:.3f}"
            
            try:
                sol = solve_ivp(
                    equations_of_motion_hamiltonian, (0, TEST_DURATION), state,
                    args=(mu, eps), method='DOP853',
                    events=EVENTS_LIST,
                    rtol=1e-6, atol=1e-8
                )
                
                # Análisis de supervivencia
                if len(sol.t_events[0]) > 0: status = "❌ CRASH: JUPITER"
                elif len(sol.t_events[1]) > 0: status = f"❌ CRASH: {sys_name}"
                elif len(sol.t_events[2]) > 0: status = "⚠️ ESCAPED"
                else:
                    status = "✅ STABLE"
                    stable_candidates.append({
                        'x': X_L4, 'y': y, 'vx': vx, 'vy': 0.0,
                        'label': f'Stable_Y{y:.3f}_Vx{vx:.3f}'
                    })
                
                print(f"{sim_id} -> {status}")
                    
            except Exception as e:
                print(f"{sim_id} -> ERROR: {e}")

    print("-" * 60)
    print(f"Análisis completo. Encontradas {len(stable_candidates)} órbitas estables.")
    
    if not stable_candidates:
        return

    # --- GENERACIÓN DE ARCHIVO ---
    output_lines = []
    output_lines.append(f"# Initial data from ER3BP Scan ({sys_name} System)")
    output_lines.append("# 1:x 2:y 3:z 4:vx 5:vy 6:vz 7:mass\n")
    
    # 1. Escribir Cuerpos Principales (Júpiter + Luna Activa)
    for name, body in [("Jupiter", DATA_JUPITER), (sys_name, DATA_SECONDARY)]:
        r, v, m = body['r'], body['v'], body['m']
        output_lines.append(f"# {name}\n{r[0]:.16e} {r[1]:.16e} {r[2]:.16e} {v[0]:.16e} {v[1]:.16e} {v[2]:.16e} {m:.5e}\n")
    
    # 2. Escribir Cuerpos de Fondo (Io + La luna que NO se eligió)
    for body_str in UNUSED_BODIES_STR:
        output_lines.append(body_str + "\n")
    
    # 3. Escribir Naves
    for i, cand in enumerate(stable_candidates, 1):
        r_sc, v_sc = transform_to_inertial(cand['x'], cand['y'], cand['vx'], cand['vy'], sys_params)
        output_lines.append(f"# SC_{i}: {cand['label']}")
        output_lines.append(f"{r_sc[0]:.16e} {r_sc[1]:.16e} {r_sc[2]:.16e} {v_sc[0]:.16e} {v_sc[1]:.16e} {v_sc[2]:.16e} 1000.0\n")
        
    filename = f'CondInicSim_{sys_name}.txt'
    with open(filename, 'w') as f:
        f.writelines(output_lines)
        
    print(f"Archivo generado: {filename}")

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*40)
    print("   ER3BP STABILITY ANALYZER")
    print("="*40)
    print("SELECCIONE SISTEMA:")
    print("1. Jupiter - Europa")
    print("2. Jupiter - Ganymede")
    
    sys_choice = input("\nOpción: ")
    setup_system(sys_choice) # Configura mu, epsilon y radios

    while True:
        print(f"\n--- SISTEMA ACTIVO: {ACTIVE_CONFIG['name']} ---")
        print("1. GENERATE FILE (Integrate & Report)")
        print("0. Exit")
        
        choice = input("\nSelect: ")
        if choice == '1': 
            run_integrated_file_generator()
        elif choice == '0': 

            break
