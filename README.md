# Jupiter Moons Simulation & Resonant Orbits Laboratory

## Overview

This project is a comprehensive Python framework designed to analyze the orbital dynamics of the Jovian system (specifically Jupiter, Europa, and Ganymede). It combines analytical tools from the **Elliptic Restricted Three-Body Problem (ER3BP)** with a **N-Body numerical integrator** to study stable orbits, resonant tori, and Lagrangian point (L4) stability.

The suite allows users to generate precise initial conditions based on stability maps (Poincaré sections), simulate the trajectories using a symplectic integrator, and visualize the results in high-quality 3D animations.

## Project Structure

The project is organized into three main modules:

### 1. Dynamics Analysis & Generation (`Códigos/`)
* **`ER3BP UNIFIED LABORATORY.py`**: The core analytical tool. It solves the ER3BP Hamiltonian formulation to:
    * Analyze stability via Poincaré Maps.
    * Visualize trajectories in the Rotating/Pulsating frame.
    * **Export Data**: Transforms stable orbits from the rotating frame to the inertial frame to be used as initial conditions for the N-Body simulation.
* **`cr3bp_l4_analysis.py`**: Specialized tool for the Circular Restricted Three-Body Problem, focusing on L4 invariant tori visualization and iso-energetic maps.
* **`er3bp_l4_analysis.py`**: Extended analysis for the Elliptical case.

### 2. N-Body Simulation (`Códigos/`)
* **`Main_Sim_4BP.py`**: A high-precision N-Body simulator using the **Velocity Verlet algorithm** (symplectic and time-reversible).
    * Loads normalized initial conditions.
    * Integrates the equations of motion for Jupiter, its moons, and multiple spacecraft/test particles.
    * Calculates and tracks total energy conservation to ensure simulation accuracy.
    * Exports simulation history to compressed NumPy files (`.npz`).

### 3. Visualization (`Códigos/`)
* **`GraphicsAnimation.py`**: Generates 3D GIFs of the simulation results. It creates a galactic background and traces the paths of the moons and spacecraft relative to Jupiter.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/juanpcruz0209/Jupyter-Moons-Simulation-Resonant-Orbits-Tori.git

Install the required dependencies:

Bash

pip install -r Requirements.txt

Usage Workflow
To run a complete simulation experiment, follow this sequence:

Step 1: Generate Initial Conditions
Run the Unified Laboratory to find a stable orbit around L4 or within the system.

Bash

python "Códigos/ER3BP UNIFIED LABORATORY.py"
Select the system (e.g., Jupiter-Ganymede).

Use option 3 ("GENERATE N-Body FILE") to export the calculated positions and velocities to the JPLData/ folder (e.g., CondInicSim_GANYMEDE.txt).

Step 2: Run the N-Body Simulation
Execute the main simulation script. Ensure the script points to the file generated in Step 1.

Bash

python "Códigos/Main_Sim_4BP.py"
This will generate a binary data file (e.g., Elliptical_Ganymede_Ten_Spacecrafts.npz) in the SimulatedData/ folder.

Note: Simulation parameters (dt, total time) can be adjusted inside the __main__ block of the script.

Step 3: Generate Animation
Visualize the resulting data.

Bash

python "Códigos/GraphicsAnimation.py"
This script reads the .npz file and saves a GIF animation in the AnimatedGraphics/ folder.

Theoretical Background
ER3BP (Elliptic Restricted Three-Body Problem): Used to model the dynamics of a spacecraft of negligible mass under the influence of two massive primaries (Jupiter and a Moon) moving in elliptical orbits.

Verlet Integration: A symplectic integration scheme used for the N-Body simulation. It offers excellent conservation of energy over long time steps, making it ideal for orbital mechanics.

Coordinate Systems: The project handles complex transformations between the Synodic (Rotating-Pulsating) frame used for stability analysis and the Inertial (Cartesian) frame used for N-Body propagation.

Authors
Juan Sebastián Victorino

Juan Pablo Cruz Gutiérrez

Universidad Nacional de Colombia
