#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python script to animate the Jupiter-Europa-Ganymede system.
Optimized version: Uses loops and structures instead of manual repetition.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

def main():
    # --- 1. Load Data ---
    filename = 'Elliptical_Ganymede_Ten_Spacecrafts.npz'
    path = os.path.join('..', 'SimulatedData', filename)
    
    print(f"Loading data from: {os.path.abspath(path)}")
    
    if not os.path.exists(path):
        print(f" ✗ Error! File not found at {os.path.abspath(path)}")
        return

    try:
        with np.load(path) as data:
            # Check keys safely
            if 'positions' not in data and 'posiciones' in data:
                positions = data['posiciones']
            elif 'positions' in data:
                positions = data['positions']
            else:
                print(f" ✗ Error! Keys found: {list(data.keys())}")
                return
    except Exception as e:
        print(f" ✗ Error loading data: {e}")
        return

    print(" ✓ Data loaded successfully.")
    
    # shape of positions: (frames, bodies, 3)
    num_frames, num_bodies, _ = positions.shape
    print(f"   Frames: {num_frames}, Bodies: {num_bodies}")

    # --- Configuration for Bodies ---
    # Define specific styles for the first 4 bodies (Planets/Moons)
    # and a generic logic or list for the spacecrafts.
    
    # Colors for spacecrafts (SC_1 to SC_10) based on your original hex codes
    sc_colors = [
        '#440154', '#482878', '#3e4989', '#31688e', '#26828e', 
        '#1f9e89', '#35b779', '#6dcd59', '#b4de2c', '#fde725'
    ]

    # Build a configuration list for all bodies
    bodies_config = []
    
    # Main Bodies (Indices 0-3)
    bodies_config.append({'label': 'Jupiter',  'color': 'sandybrown', 'size': 10}) # 0
    #bodies_config.append({'label': 'Europa',   'color': 'coral',      'size': 6})  # 1
    #bodies_config.append({'label': 'Io',       'color': 'beige',      'size': 6})  # 2
    bodies_config.append({'label': 'Ganymede', 'color': 'olive',      'size': 6})  # 3

    # Spacecrafts (Indices 4+)
    for i in range(4, num_bodies):
        sc_idx = i - 4
        color = sc_colors[sc_idx % len(sc_colors)] # Cycle colors if we have more SCs than colors
        bodies_config.append({
            'label': f'SC_{sc_idx + 1}',
            'color': color,
            'size': 5
        })

    # --- 2. Configure Figure and 3D Axes ---
    print("Configuring figure and galactic background...")
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Styling
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # --- Calculate View Limits ---
    # Vectorized min/max calculation over all bodies and frames
    all_pos = positions.reshape(-1, 3)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    margins = (maxs - mins) * 0.1

    # Handle 2D case (flat simulation)
    margins[margins == 0] = 1.0

    view_lims = list(zip(mins - margins, maxs + margins))
    ax.set_xlim(view_lims[0])
    ax.set_ylim(view_lims[1])
    ax.set_zlim(view_lims[2])

    # --- Starfield Generation ---
    star_ext = 1.5
    n_stars = 1000
    for i, (vmin, vmax) in enumerate(view_lims):
        # Simple star generation logic per axis
        center = (vmax + vmin) / 2
        span = (vmax - vmin) * star_ext
        # We generate stars generically for x, y, z
        # Note: reusing variable names strictly for plotting logic
        if i == 0: stars_x = np.random.uniform(center - span/2, center + span/2, n_stars)
        if i == 1: stars_y = np.random.uniform(center - span/2, center + span/2, n_stars)
        if i == 2: stars_z = np.random.uniform(center - span/2, center + span/2, n_stars)

    star_sizes = np.random.uniform(0.1, 1.0, n_stars)
    ax.scatter(stars_x, stars_y, stars_z, s=star_sizes, c='snow', alpha=0.25)

    # --- 3. Draw Initial Artists ---
    print("Initializing graphics...")
    
    # Draw trails for Europa (1) and Io (2) - optional logic
    # You can loop this if you want trails for everyone
    for i in [1, 2]:
        ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2],
                color='lightgrey', linewidth=0.5, linestyle=':')

    # Initialize points list to store the "Artist" objects
    points = []
    
    for i, cfg in enumerate(bodies_config):
        # Initial position: Frame 0, Body i
        p, = ax.plot([positions[0, i, 0]], 
                     [positions[0, i, 1]], 
                     [positions[0, i, 2]], 
                     'o', 
                     markersize=cfg['size'], 
                     color=cfg['color'], 
                     label=cfg['label'])
        points.append(p)

    # Labels and Style
    ax.set_xlabel('IRU', color='white')
    ax.set_ylabel('IRU', color='white')
    ax.set_zlabel('IRU', color='white')
    ax.set_title('Jupiter System Dynamics', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Legend (optional, can get crowded)
    # legend = ax.legend(loc='upper right', fontsize='small')
    # plt.setp(legend.get_texts(), color='white')

    # --- 5. Animation Functions ---
    
    def update_point(frame_idx, body_idx, point_artist):
        """Helper to update a single point."""
        # Extract x, y, z for the specific frame and body
        x = positions[frame_idx, body_idx, 0]
        y = positions[frame_idx, body_idx, 1]
        z = positions[frame_idx, body_idx, 2]
        point_artist.set_data([x], [y])
        point_artist.set_3d_properties([z])

    def init():
        """Init function for FuncAnimation."""
        for i, p in enumerate(points):
            update_point(0, i, p)
        return points

    def animate(i):
        """Update function for each frame."""
        for body_idx, p in enumerate(points):
            update_point(i, body_idx, p)
        return points

    # --- 6. Create, Save, and Show ---
    print("Creating animation...")
    
    # Step calculation to limit frames for smoother/faster GIF generation
    target_frames = 300
    step = max(1, num_frames // target_frames)
    frames_indices = range(0, num_frames, step)
    
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=frames_indices, interval=30, blit=True)

    output_dir = os.path.join('..', 'AnimatedGraphics')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '10Bodies_Elliptical_Optimized_Ganymede.gif')

    print(f"Saving animation to: {save_path}")
    # Added extra_args to ensure clear background handling in some backends
    anim.save(save_path, writer='pillow', fps=20, dpi=100) 
    print("✓ Animation saved!")

    print("Showing plot window...")
    plt.show()

if __name__ == "__main__":
    main()