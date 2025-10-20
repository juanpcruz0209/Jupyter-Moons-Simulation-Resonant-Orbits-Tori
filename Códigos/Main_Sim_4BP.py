#!/usr/bin/env python
# coding: utf-8

# # Simulación de N-Cuerpos - Algoritmo de Verlet 
# ## 🔧 Configuración e Importaciones

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.constants import M_jup, G
import os
from typing import List, Tuple

# ## 📊 Constantes y Unidades

# Global Constants
JMU = M_jup # Jupyter Mass in Kg
IRU = 4.22e8  # Aproximated value of semi-major axis of Io's Orbit
JTU = np.sqrt(np.power(IRU,3)/(JMU * G))  # Exactly time to 

# Normalized Gravitational Constant
G_Normal = G * JMU * np.power(JTU, 2) * (1 / np.power(IRU, 3))

# ## 🏗️ Definición de Clases

class Cuerpo:
    """
    Represents a celestial body with mass, position, and velocity.

    Attributes:
        m (float): Mass of the body.
        r (NDArray[np.float64]): Position vector [x, y, z] as a NumPy array.
        V (NDArray[np.float64]): Velocity vector [Vx, Vy, Vz] as a NumPy array.
    """
    def __init__(self, x0: float, y0: float, z0: float, 
                 Vx0: float, Vy0: float, Vz0: float, m0: float):
        """
        Initializes the Cuerpo object.

        Args:
            x0 (float): Initial x-coordinate.
            y0 (float): Initial y-coordinate.
            z0 (float): Initial z-coordinate.
            Vx0 (float): Initial x-velocity component.
            Vy0 (float): Initial y-velocity component.
            Vz0 (float): Initial z-velocity component.
            m0 (float): Mass of the body.
        """
        self.m: float = m0
        self.r: NDArray[np.float64] = np.array([x0, y0, z0])
        self.V: NDArray[np.float64] = np.array([Vx0, Vy0, Vz0])
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Cuerpo object.
        """
        # Improved formatting for representation
        pos_str = np.array2string(self.r, precision=3, separator=', ', suppress_small=True)
        vel_str = np.array2string(self.V, precision=3, separator=', ', suppress_small=True)
        return f"Cuerpo(m={self.m:.3e}, r={pos_str}, V={vel_str})"

class DynamicManager:
    """
    Manages the calculation of gravitational forces and potential energy 
    for a system of celestial bodies.
    
    Attributes:
        EpTotal (float): Stores the total potential energy calculated in the last 
                         call to calculateAllForces. Note: This gets reset on each call.
    """
    def __init__(self):
        """Initializes the DynamicManager."""
        self.EpTotal: float = 0.0

    def calculateAllForces(self, Planetas: list[Cuerpo], positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculates the total gravitational force on each body and the total potential energy.
        
        Uses vectorized NumPy operations for efficiency. Avoids calculating force
        of a body on itself and prevents division by zero. Calculates the total
        potential energy U = -G * sum(mi*mj / rij) for i < j.

        Args:
            Planetas (list[Cuerpo]): A list of Cuerpo objects in the system.
            positions (NDArray[np.float64]): A NumPy array of shape (N, 3) containing the 
                                             current positions of all N bodies.

        Returns:
            NDArray[np.float64]: A NumPy array of shape (N, 3) containing the total 
                                 gravitational force vector acting on each body.
                                 Updates self.EpTotal with the calculated total potential energy.
        """
        self.EpTotal = 0.0  # Reset potential energy for this calculation step

        N = len(Planetas)
        if N < 2:
            return np.zeros((N, 3)) # No forces if less than 2 bodies

        m = np.array([c.m for c in Planetas])
        # Reshape masses for broadcasting (N,) -> (N, 1)
        masses_col = m.reshape(-1, 1) 
        
        # Calculate pairwise position differences: shape (N, N, 3)
        # r_i - r_j
        pos_differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        
        # Calculate pairwise distances: shape (N, N)
        # ||r_i - r_j||
        distances = np.linalg.norm(pos_differences, axis=2)
        
        # Add identity matrix to distance^3 to avoid division by zero on the diagonal (i=j)
        # We handle the diagonal elements later by setting them to zero force.
        # Using np.fill_diagonal avoids creating a potentially large identity matrix for large N
        distances_cube = distances**3
        np.fill_diagonal(distances_cube, 1.0) # Avoid division by zero, diag force will be zero anyway

        # Calculate force contributions F_ij = G * m_i * m_j * (r_j - r_i) / ||r_i - r_j||^3
        # Note: pos_differences[i, j, :] = r_i - r_j. We need -(r_j - r_i) = r_i - r_j
        # Force calculation using broadcasting:
        # masses_col (N, 1) * pos_differences (N, N, 3) / distances_cube (N, N, 1)
        # Result shape: (N, N, 3), where result[i, j, :] is the force ON i DUE TO j
        force_contributions = (masses_col * pos_differences) / distances_cube[..., np.newaxis]

        # Sum forces acting ON each body 'i' (sum over axis=1, which is the 'j' index)
        # The result includes the G constant and the minus sign F_i = sum_j(F_ij)
        total_forces = -G_Normal.value * np.sum(force_contributions, axis=1)
        
        # --- Potential Energy Calculation ---
        # U = -G * sum_{i<j} (m_i * m_j / r_ij)
        # Create matrix of mass products m_i * m_j: shape (N, N)
        mass_product_matrix = np.outer(m, m)
        
        # Set diagonal distances to infinity to avoid self-energy and division by zero
        np.fill_diagonal(distances, np.inf) 
        
        # Calculate potential energy matrix terms (m_i * m_j / r_ij)
        potential_terms = mass_product_matrix / distances
        
        # Summing all elements
        U = -G_Normal.value * np.sum(potential_terms) / 2.0
        self.EpTotal = U
        
        return total_forces

# ## 📥 Función para Cargar Datos

def load_and_normalize_bodies(file_name: str, 
                              dist_unit: float, 
                              time_unit: float, 
                              mass_unit: float) -> List[Cuerpo]:
    """
    Loads celestial body initial conditions (pos, vel, mass) from a text file 
    IN SI UNITS (e.g., m, m/s, kg) and normalizes them using the provided units.

    Assumes the file format has 7 columns per line, separated by whitespace:
    x, y, z, vx, vy, vz, m
    Lines starting with '#' are ignored as comments.

    Args:
        file_name (str): The path to the input data file.
        dist_unit (float): The characteristic distance unit for normalization (e.g., IRU in meters).
        time_unit (float): The characteristic time unit for normalization (e.g., JTU in seconds).
        mass_unit (float): The characteristic mass unit for normalization (e.g., JMU in kg).

    Returns:
        List[Cuerpo]: A list of 'Cuerpo' objects initialized with NORMALIZED data.
                      Returns an empty list if the file is not found or contains errors.
                      
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If a data line does not contain exactly 7 numeric values.
        Exception: For other potential errors during file reading or processing.
    """
    bodies_list = []
    try:
        data = np.loadtxt(file_name, comments='#', dtype=float)

        # Validate data shape
        if data.ndim == 1: 
            if data.shape[0] == 7:
                data = data.reshape(1, 7)
            else:
                 raise ValueError(f"Data line in '{file_name}' needs 7 columns, found {data.shape[0]}.")
        elif data.shape[1] != 7:
            raise ValueError(f"Data lines in '{file_name}' need 7 columns, found {data.shape[1]}.")

        # --- Calculate velocity unit ---
        vel_unit = dist_unit / time_unit # Characteristic velocity (m/s if inputs are SI)

        # --- Iterate, Normalize, and Create Cuerpo objects ---
        for row in data:
            x_si, y_si, z_si, vx_si, vy_si, vz_si, m_si = row
            
            # Normalize position (divide SI value by distance unit)
            x_norm = x_si / dist_unit
            y_norm = y_si / dist_unit
            z_norm = z_si / dist_unit
            
            # Normalize velocity (divide SI value by velocity unit)
            vx_norm = vx_si / vel_unit
            vy_norm = vy_si / vel_unit
            vz_norm = vz_si / vel_unit
            
            # Normalize mass (divide SI value by mass unit)
            m_norm = m_si / mass_unit
            
            # Append the Cuerpo object with NORMALIZED values
            bodies_list.append(Cuerpo(x_norm, y_norm, z_norm, vx_norm, vy_norm, vz_norm, m_norm))
            
        print(f"✅ Successfully loaded and normalized {len(bodies_list)} bodies from {file_name}.")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_name}' not found.")
        raise 
    except ValueError as ve:
        print(f"❌ Error parsing file '{file_name}': {ve}")
        return [] 
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading '{file_name}': {e}")
        return []

    return bodies_list

# ## 🔄 Algoritmo de Verlet

def verletAlgorithm(finalTime: float, dt: float, Planetas: list[Cuerpo]
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Implements the position Verlet integration algorithm for an N-body system.

    This method calculates the trajectory of N bodies, presumably under mutual
    gravitational attraction. It is a symplectic and time-reversible integrator,
    known for its good long-term stability in energy conservation.

    The position Verlet algorithm is:
    r(t + dt) = 2*r(t) - r(t - dt) + a(t) * dt^2

    A "bootstrap" step is required to estimate r(-dt) to start the iteration.
    This implementation uses a first-order approximation:
    r(-dt) ≈ r(0) - v(0) * dt

    Note on implementation:
    Velocities are calculated using the centered difference formula:
    v(t) = (r(t + dt) - r(t - dt)) / (2 * dt)
    As a result, the velocity at the final time step (n-1) is not
    calculated, and the last row of `velocityBodies` will be all zeros.
    Similarly, energy is tied to the velocity calculation, so the energy
    at the final time step is also not computed.

    Parameters:
    ----------
    finalTime : float
        Total simulation time (in normalized units).
    dt : float
        Time step (in normalized units).
    Planetas : list[Cuerpo]
        A list of 'Cuerpo' (Body) objects that make up the system.
        Each object must have .r, .V, and .m attributes.

    Returns:
    -------
    tuple
        A tuple containing four numpy arrays:
        - positionsBodies (np.array[n, N, 3]): 
            History of positions for N bodies at n time steps.
        - velocityBodies (np.array[n, N, 3]): 
            History of velocities. Note: The last entry (index [n-1]) 
            is not calculated and will be [0, 0, 0].
        - energy (np.array[n-1, 2]): 
            History of [Potential, Kinetic] energy. Contains n-1 entries,
            from t=0 to t=(n-2)*dt.
        - t_array (np.array[n]): 
            Array of the n time steps from 0 to finalTime.
    """
    
    Newton = DynamicManager()
    t_array = np.arange(0, finalTime + dt, dt)
    n = len(t_array)  # Number of time steps
    N = len(Planetas) # Number of bodies
    
    # Storage arrays
    positionsBodies = np.zeros((n, N, 3))
    velocityBodies = np.zeros((n, N, 3))
    # Energy array has n-1 rows, storing from t=0 to t=(n-2)*dt
    energy = np.zeros((n-1, 2)) 
    
    # Initial conditions from the list of objects
    initialPosition = np.array([cuerpo.r for cuerpo in Planetas]) # (N, 3)
    initialVelocity = np.array([cuerpo.V for cuerpo in Planetas]) # (N, 3)
    masses = np.array([cuerpo.m for cuerpo in Planetas])     # (N,)
    massesN1 = masses.reshape(-1, 1) # (N, 1) for broadcasting
    
    # --- Bootstrap Step (t=0) ---
    
    # 1. Estimate r(-dt) using r(-dt) ≈ r(0) - v(0)*dt
    position_verletm1 = initialPosition - initialVelocity * dt 
    
    # 2. Calculate initial acceleration a(0)
    a_0 = Newton.calculateAllForces(Planetas, initialPosition)
    
    # 3. Calculate r(dt) using the Verlet formula
    # r(1) = 2*r(0) - r(-1) + a(0)*dt^2
    position_verlet_step_np1 = 2*initialPosition - position_verletm1 + a_0*(dt**2)
    
    # 4. Store initial (t=0) and first (t=1) steps
    positionsBodies[0] = initialPosition
    positionsBodies[1] = position_verlet_step_np1
    velocityBodies[0] = initialVelocity # Store known initial velocity
    
    # 5. Store initial energy (t=0)
    energy[0][0] = Newton.EpTotal # Potential Energy U(0)
    energy[0][1] = (1/2) * np.sum(np.sum(initialVelocity**2, axis=1) * masses) # Kinetic K(0)
    
    # --- Main integration loop ---
    # Starts from i=2 (calculating r(2*dt)) up to i=n-1 (calculating r((n-1)*dt))
    for i in range(2, n):
        # r(i-2)
        position_verlet_step_nm1 = positionsBodies[i-2]
        # r(i-1)
        position_verlet_step_n = positionsBodies[i-1]
        
        # 1. Calculate acceleration a(i-1) at the current step
        a_n = Newton.calculateAllForces(Planetas, position_verlet_step_n)
        
        # 2. Calculate next position r(i) using Verlet
        # r(i) = 2*r(i-1) - r(i-2) + a(i-1)*dt^2
        position_verlet_step_np1 = (2*position_verlet_step_n - 
                                  position_verlet_step_nm1 + a_n*(dt**2))
        
        # Store position r(i)
        positionsBodies[i] = position_verlet_step_np1
        
        # 3. Calculate velocity v(i-1) using centered difference
        # v(i-1) = (r(i) - r(i-2)) / (2*dt)
        velocity = (position_verlet_step_np1 - position_verlet_step_nm1) / (2*dt)
        velocityBodies[i-1] = velocity
        
        # 4. Calculate energies at step t=(i-1)*dt
        # U(i-1)
        energy[i-1][0] = Newton.EpTotal 
        # K(i-1)
        v_squared = np.sum(velocity**2, axis=1, keepdims=True)
        mv2 = massesN1 * v_squared
        energy[i-1][1] = (1/2) * np.sum(mv2)
        
        if i % 100 == 0:
            print(f'📊 Progress: {i/n*100:.1f}%', end='\r')
    
    print('✅ Simulation completed: 100.0%')
    
    # Note: velocityBodies[n-1] and energy[n-1] are not calculated.
    return positionsBodies, velocityBodies, energy, t_array

# ## 📈 Visualización de Resultados

def plot_3d_tracks(positions_history: np.ndarray,
                   bodies,
                   title: str = "N-Body System Trajectories"):
    """
    Visualizes the 3D trajectories of an N-body simulation.

    Parameters:
    ----------
    positions_history : np.ndarray
        A 3D NumPy array containing the position history of all bodies.
        The expected shape is (n_steps, n_bodies, 3), where the last
        dimension corresponds to [x, y, z].

    bodies : list, optional
        A list of 'Cuerpo' (Body) objects or any list where each item
        has a '.name' attribute. This is used to label the trajectories
        in the legend. If not provided, bodies will be labeled
        "Body 1", "Body 2", etc.

    title : str, optional
        The title for the plot.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The Figure object for the plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D Axes object, allowing for further customization
        (e.g., saving the figure, changing limits).
        
    Assumes:
    ------
    - `positions_history[:, i, 0]` is the X-coordinate.
    - `positions_history[:, i, 1]` is the Y-coordinate.
    - `positions_history[:, i, 2]` is the Z-coordinate.
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the number of bodies directly from the positions array
    num_bodies = positions_history.shape[1]

    for i in range(num_bodies):
        # Extract the full trajectory for body 'i'
        x = positions_history[:, i, 0]
        y = positions_history[:, i, 1] 
        z = positions_history[:, i, 2]
        
        # --- Determine the label for the legend ---
        label = f'Body {i+1}' # Default label
        if bodies and i < len(bodies):
            try:
                # Try to get a name attribute from the object
                label = bodies[i].name
            except AttributeError:
                # If it has no .name, use the default
                pass 
        
        # --- Plot the trajectory ---
        # By not specifying a 'color', matplotlib automatically cycles
        # through different colors for each 'plot' call.
        ax.plot(x, y, z, label=label, linewidth=2)
        
        # --- Plot the starting point (t=0) ---
        # The scatter plot will automatically use the *same color*
        # as the line, which is better than hardcoding 'red'.
        ax.scatter(x[0], y[0], z[0], marker='o', s=50) 
    
    # --- Set labels and title in English ---
    # Note: I changed 'X [rad]' to 'X [AU]' as 'rad' (radians)
    # is an unusual unit for a Cartesian axis.
    ax.set_xlabel('X [IRU]')
    ax.set_ylabel('Y [IRU]')
    ax.set_zlabel('Z [IRU]')
    ax.set_title(title)
    
    ax.legend()
    
    # Return the figure and axes objects instead of calling plt.show()
    # This gives the user more control (e.g., to save the figure)
    return fig, ax

# # Exportación de datos

def export_data_simulation(filename, t_array, positions, velocities, energy, planetas):
    """
    Saves simulation data arrays and body masses to a compressed .npz file.

    This function packages all major simulation outputs (time, position,
    velocity, energy) and the corresponding body masses into a single,
    compressed NumPy file (`.npz`). This allows for easy loading and
    post-processing of the simulation results.

    The saved file will contain the following data keys:
    - 'tiempo': The time steps array.
    - 'posiciones': The positions history.
    - 'velocidades': The velocities history.
    - 'energia': The energy (Potential, Kinetic) history.
    - 'masas': The masses of the simulated bodies.

    Parameters:
    ----------
    nombre_archivo : str
        The name of the output file (e.g., "simulation_data.npz").
    t_array : np.ndarray
        Array of time steps, shape (n_steps,).
    positions : np.ndarray
        Array of positions, shape (n_steps, n_bodies, 3).
    velocities : np.ndarray
        Array of velocities, shape (n_steps, n_bodies, 3).
    energy : np.ndarray
        Array of energies, shape (n_steps, 2).
    planetas : list[Cuerpo]
        List of 'Cuerpo' (Body) objects from which to extract masses.

    Prints:
    ------
    str
        A success message indicating the file path upon successful save,
        or an error message if the save operation fails.
    """
    try:
        # Extract masses to save them as well
        masses = np.array([c.m for c in planetas])
        
        np.savez_compressed(
            filename,
            tiempo=t_array,
            posiciones=positions,
            velocidades=velocities,
            energia=energy,
            masas=masses
        )
        print(f"✅ Simulation data successfully exported to: {filename}")
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

# ## 🎯 Ejecución Completa de la Simulación

if __name__ == "__main__":
    finalTime = 15      # 6.76 hours per unit
    dt = 0.0001          # Time Step
    Relative_Path = os.path.join('..', 'JPLData', 'CondInicSim.txt')
    
    # Cargar datos
    Bodies = load_and_normalize_bodies(Relative_Path, IRU, JTU.value, JMU.value)
    
    if Bodies:
         # Ejecutar simulación
         positionsBodies, velocityBodies, energy, t_array = verletAlgorithm(
             finalTime, dt, Bodies
         )
        
         # 1. Definir la carpeta y el nombre del archivo
         output_folder = os.path.join('..', 'SimulatedData')
         file_name = "simulacion_jupiter_Europa_Example.npz"
         full_export_path = os.path.join(output_folder, file_name)

         # 2. Asegurarse de que la carpeta exista (la crea si no existe)
         os.makedirs(output_folder, exist_ok=True)

         # 3. Exportar datos usando la ruta completa
         export_data_simulation(
             full_export_path, # Usar la nueva variable con la ruta completa
             t_array,
             positionsBodies,
             velocityBodies, 
             energy,
             Bodies  
         )

         # Visualizar
         fig, ax = plot_3d_tracks(positionsBodies, Bodies)
         plt.show() # <-- Añadido para mostrar el gráfico
        
         # Mostrar estadísticas
         print(f"\n📊 Simulation Statistics:")
         print(f"   • Time simulated: {finalTime * JTU.value / 3600 :.2e} hours")
         print(f"   • Time Step: {len(t_array)}")
         print(f"   • Number of Bodies: {len(Bodies)}")
         print(f"   • Initial Total Energy: {energy[0,0] + energy[0,1]:.6e}")
         print(f"   • Final Total Energy: {energy[-1,0] + energy[-1,1]:.6e}")