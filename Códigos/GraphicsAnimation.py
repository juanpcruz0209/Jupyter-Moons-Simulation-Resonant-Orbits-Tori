#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python script to animate the Jupiter-Europa-Ganymede system.
Migrated from a Jupyter Notebook.

This script loads simulation data from a .npz file and generates
a 3D animation of the orbits with a galactic background.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
import os

def main():
    """Main function to encapsulate the script's logic."""
    
    # --- 1. Load Data ---
    # Define the relative path to the simulation data file.
    path = os.path.join('..', 'SimulatedData', 'DataIoEuropeGanimedeJupyter688H.npz')
    
    print(f"Loading data from: {os.path.abspath(path)}")
    try:
        data = np.load(path)
        # Load the 'positions' array. Assumes 'posiciones' is the key.
        positions = data['positions'] 
    except FileNotFoundError:
        print(f"  ✗ Error! File not found at {os.path.abspath(path)}")
        print("  - Please check that the path and filename are correct.")
        return
    except KeyError:
        print("  ✗ Error! The .npz file does not contain the key 'posiciones'.")
        print(f"  - Keys found in file: {list(data.keys())}")
        return

    print("  ✓ Data loaded successfully.")

    # Extract position data for each body.
    # Based on the labels later, 0=Jupyter, 1=Europe, 2=Spacecraft
    x_body0 = positions[:, 0, 0]
    y_body0 = positions[:, 0, 1]
    z_body0 = positions[:, 0, 2]

    x_body1 = positions[:, 1, 0]
    y_body1 = positions[:, 1, 1]
    z_body1 = positions[:, 1, 2]

    x_body2 = positions[:, 2, 0]
    y_body2 = positions[:, 2, 1]
    z_body2 = positions[:, 2, 2]

    x_body3 = positions[:, 3, 0]
    y_body3 = positions[:, 3, 1]
    z_body3 = positions[:, 3, 2]

    # Get the total number of time steps (frames) from the data.
    num_frames = len(x_body0)

    # --- 2. Configure Figure and 3D Axes ---
    print("Configuring figure and galactic background...")
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- Galactic Background Setup ---
    # Set the figure and axes background color to black.
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Disable the 3D axis panes (the grey "walls") to make them transparent.
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Disable the grid lines.
    ax.grid(False)

    # --- Dynamic Axis Limits Calculation ---
    # Find the min/max values across all bodies and axes to set the view.
    x_min, x_max = positions[:, :, 0].min(), positions[:, :, 0].max()
    y_min, y_max = positions[:, :, 1].min(), positions[:, :, 1].max()
    z_min, z_max = positions[:, :, 2].min(), positions[:, :, 2].max()

    # Add a 10% margin to each axis limit for better visibility.
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    z_margin = (z_max - z_min) * 0.1

    # Define the final view limits.
    view_xlim = (x_min - x_margin, x_max + x_margin)
    view_ylim = (y_min - y_margin, y_max + y_margin)
    view_zlim = (z_min - z_margin, z_max + z_margin)
    
    # Handle flat (2D) simulations: add a small buffer if margin is 0.
    if x_margin == 0: view_xlim = (x_min - 1, x_max + 1)
    if y_margin == 0: view_ylim = (y_min - 1, y_max + 1)
    if z_margin == 0: view_zlim = (z_min - 1, z_max + 1)

    # --- Starfield Generation ---
    # Define a volume for stars 1.5x larger than the view, to give depth.
    star_extension = 1.5
    star_xlim = (view_xlim[0] * star_extension, view_xlim[1] * star_extension)
    star_ylim = (view_ylim[0] * star_extension, view_ylim[1] * star_extension)
    star_zlim = (view_zlim[0] * star_extension, view_zlim[1] * star_extension)

    # Generate 1000 random stars.
    n_stars = 1000
    stars_x = np.random.uniform(*star_xlim, n_stars)
    stars_y = np.random.uniform(*star_ylim, n_stars)
    stars_z = np.random.uniform(*star_zlim, n_stars)
    star_sizes = np.random.uniform(0.1, 1.0, n_stars)
    
    # Plot the stars as a faint, white scatter plot.
    ax.scatter(stars_x, stars_y, stars_z, s=star_sizes, c='snow', alpha=0.25)
    # --- End of background section ---

    # --- 3. Draw Orbits and Initial Points ---
    print("Drawing orbits and initial points...")
    # Plot the full trajectory (trace) for Europe and Spacecraft as faint lines.
    ax.plot(x_body1, y_body1, z_body1,  color='lightgrey', linewidth=0.5, linestyle=':')
    ax.plot(x_body2, y_body2, z_body2,  color='lightgrey', linewidth=0.5, linestyle=':')
    ax.plot(x_body3, y_body3, z_body3,  color='lightgrey', linewidth=0.5, linestyle=':')

    # Create the animated points (artists) for each body with labels.
    point_body0, = ax.plot([x_body0[0]], [y_body0[0]], [z_body0[0]], 'o', markersize=10, color='sandybrown', label='Jupyter')
    point_body1, = ax.plot([x_body1[0]], [y_body1[0]], [z_body1[0]], 'o', markersize=6, color='coral', label='Ío')
    point_body2, = ax.plot([x_body2[0]], [y_body2[0]], [z_body2[0]], 'o', markersize=6, color='beige', label='Europe')
    point_body3, = ax.plot([x_body3[0]], [y_body3[0]], [z_body3[0]], 'o', markersize=6, color='olive', label='Ganymede')
    # --- 4. Final Plot Configuration ---
    # Set labels and title with white text to be visible on the black background.
    ax.set_xlabel('IRU', color='white')
    ax.set_ylabel('IRU', color='white')
    ax.set_zlabel('IRU', color='white')
    ax.set_title('Jupyter-Ío-Europe-Ganymede System, 688 hours', color='white')

    # Set the color of the axis tick marks and labels to white.
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Create the legend and set its text color to white.
    legend = ax.legend(loc='upper right')
    plt.setp(legend.get_texts(), color='white')

    # Apply the calculated "crop" to the axes.
    ax.set_xlim(view_xlim)
    ax.set_ylim(view_ylim)
    ax.set_zlim(view_zlim)

    # --- 5. Animation Functions ---
    def init():
        """Initializes the animation by setting the starting positions."""
        point_body0.set_data_3d([x_body0[0]], [y_body0[0]], [z_body0[0]])
        point_body1.set_data_3d([x_body1[0]], [y_body1[0]], [z_body1[0]])
        point_body2.set_data_3d([x_body2[0]], [y_body2[0]], [z_body2[0]])
        point_body3.set_data_3d([x_body3[0]], [y_body3[0]], [z_body3[0]])
        # Return a tuple of the artists to be animated.
        return (point_body0, point_body1, point_body2,point_body3)

    def animate(i):
        """
Setting `blit=True` means `animate` must return an iterable of artists.
        """
        # `i` is the frame index provided by FuncAnimation.
        point_body0.set_data_3d([x_body0[i]], [y_body0[i]], [z_body0[i]])
        point_body1.set_data_3d([x_body1[i]], [y_body1[i]], [z_body1[i]])
        point_body2.set_data_3d([x_body2[i]], [y_body2[i]], [z_body2[i]])
        point_body3.set_data_3d([x_body3[i]], [y_body3[i]], [z_body3[i]])
        # Return the tuple of artists that have been updated.
        return (point_body0, point_body1, point_body2,point_body3)

    # --- 6. Create, Save, and Show the Animation ---
    print("Creating animation (FuncAnimation)...")
    
    # Calculate a step size to target ~300 frames for the animation.
    # This prevents the animation from being too long if num_frames is large.
    step = max(1, int(num_frames / 300)) # Ensure step is at least 1.
    frames_to_render = range(0, num_frames, step)

    # Create the animation object.
    anim = FuncAnimation(fig, animate, init_func=init,
                           frames=frames_to_render, interval=30, blit=True)
    
    # --- Save the animation as a GIF ---
    
    # Define the output path.
    output_directory = os.path.join('..', 'AnimatedGraphics')
    file_name = 'DataEuropeJupyterTest8_270H.gif'
    full_save_path = os.path.join(output_directory, file_name)

    # Create the output directory if it doesn't already exist.
    os.makedirs(output_directory, exist_ok=True)

    # Save the animation.
    print(f"Saving animation to {os.path.abspath(full_save_path)}...")
    print("This may take several minutes.")
    # `writer='pillow'` is used for saving GIFs. `dpi` controls resolution.
    anim.save(full_save_path, writer='pillow', fps=20, dpi=100)
    print("Animation saved successfully!")

    # --- Show the animation ---
    
    # `plt.show()` opens an interactive window to display the animation.
    # This is used for .py scripts, unlike `HTML(anim.to_jshtml())` in notebooks.
    print("Showing animation in a new window. Close the window to end the script.")
    plt.show()

# Standard boilerplate to run the `main` function when the script is executed.
if __name__ == "__main__":
    main()