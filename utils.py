import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # This is the older way
# For modern Matplotlib, 3D axes are created as: fig.add_subplot(projection='3d')
import random

def calculate_space_utilization(placed_items_volumes, container_volume):
    """
    Calculates the space utilization ratio.

    Args:
        placed_items_volumes: A list or array of the volumes of items successfully placed.
        container_volume: The total volume of the pallet/container.

    Returns:
        The space utilization ratio (float), or 0 if container_volume is 0.
    """
    if container_volume == 0:
        return 0.0
    
    total_placed_volume = sum(placed_items_volumes)
    utilization = total_placed_volume / container_volume
    return utilization

def visualize_pallet_matplotlib(space_grid, container_dims, item_colors=None, 
                                voxel_edge_color='k', voxel_linewidth=0.5):
    """
    Visualizes the pallet space_grid using a 3D voxel plot with Matplotlib.

    Args:
        space_grid: The 3D NumPy array representing the pallet space.
                    Non-zero entries indicate occupied cells by items (values are item IDs).
        container_dims: Tuple (length, width, height) of the container/grid.
                        Used for setting plot limits. space_grid.shape can also be used.
        item_colors: Optional. A dictionary mapping item IDs (values in space_grid) to colors.
                     If None, random colors are generated for different item IDs.
        voxel_edge_color: Color of the voxel edges.
        voxel_linewidth: Line width of the voxel edges.
    """
    if not isinstance(space_grid, np.ndarray) or space_grid.ndim != 3:
        print("Error: space_grid must be a 3D NumPy array.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get unique item IDs present in the grid (excluding 0 which is empty space)
    unique_item_ids = np.unique(space_grid[space_grid > 0])

    if item_colors is None:
        item_colors = {}
    
    # Generate random colors for item IDs not in item_colors
    for item_id in unique_item_ids:
        if item_id not in item_colors:
            item_colors[item_id] = (random.random(), random.random(), random.random(), 0.8) # RGBA, alpha for some transparency

    # Create a boolean array for occupied cells to pass to ax.voxels
    # ax.voxels expects a 3D boolean array where True indicates a voxel to draw.
    # We need to plot each item ID with its specific color, so we iterate
    # and call ax.voxels for each item_id's mask.
    
    # The `voxels` function in matplotlib can take a `facecolors` argument
    # that is an array of the same shape as `space_grid` (plus a 4th dim for RGBA).
    # This is more efficient than calling `ax.voxels` multiple times.

    # Prepare facecolors array
    facecolors = np.zeros(space_grid.shape + (4,), dtype=float) # RGBA
    filled = np.zeros(space_grid.shape, dtype=bool)

    for item_id in unique_item_ids:
        mask = (space_grid == item_id)
        filled = filled | mask # Combine masks for all items to tell `voxels` where to draw
        color = item_colors.get(item_id, (0.5, 0.5, 0.5, 0.1)) # Default grey if somehow not in item_colors
        facecolors[mask] = color
        
    # Draw all voxels at once if any items are present
    if filled.any():
        ax.voxels(filled, facecolors=facecolors, edgecolor=voxel_edge_color, linewidth=voxel_linewidth)
    else:
        print("No items to visualize in the space grid.")


    ax.set_xlabel('Length (X-axis / Grid Index)')
    ax.set_ylabel('Width (Y-axis / Grid Index)')
    ax.set_zlabel('Height (Z-axis / Grid Index)')

    # Set limits based on the shape of the space_grid
    # These are grid indices, not necessarily physical dimensions if scaling was used.
    ax.set_xlim([0, space_grid.shape[0]])
    ax.set_ylim([0, space_grid.shape[1]])
    ax.set_zlim([0, space_grid.shape[2]])
    
    ax.set_title('3D Pallet Visualization')
    
    # Optional: Create a custom legend for item colors
    # This is a bit tricky with voxels as they don't have direct legend handles like scatter plots.
    # One way is to create proxy artists.
    if item_colors and unique_item_ids.size > 0:
        patches = [plt.Rectangle((0, 0), 1, 1, fc=item_colors[item_id]) for item_id in unique_item_ids if item_id in item_colors]
        legend_labels = [f'Item ID {item_id}' for item_id in unique_item_ids if item_id in item_colors]
        if patches: # Only show legend if there are items
             ax.legend(patches, legend_labels, loc='best', title='Items')


    # Improve layout
    plt.tight_layout()
    # plt.show() # Caller should call plt.show() if needed in a script.
                 # For interactive environments, it might show automatically.

if __name__ == '__main__':
    print("--- Testing calculate_space_utilization ---")
    placed_vols = [10, 20, 30] # e.g., m^3 or any consistent unit
    container_vol = 100       # Same unit as placed_vols
    utilization = calculate_space_utilization(placed_vols, container_vol)
    print(f"Space Utilization (volumes: {placed_vols}, container: {container_vol}): {utilization*100:.2f}%")

    placed_vols_empty = []
    utilization_empty = calculate_space_utilization(placed_vols_empty, container_vol)
    print(f"Space Utilization (volumes: {placed_vols_empty}, container: {container_vol}): {utilization_empty*100:.2f}%")

    utilization_zero_container = calculate_space_utilization(placed_vols, 0)
    print(f"Space Utilization (volumes: {placed_vols}, container: 0): {utilization_zero_container*100:.2f}%")

    print("\n--- Testing visualize_pallet_matplotlib ---")
    # Create a dummy space grid (e.g., 10x10x5)
    # These dimensions are L, W, H for the grid itself.
    grid_l, grid_w, grid_h = 10, 8, 5 
    dummy_space_grid = np.zeros((grid_l, grid_w, grid_h), dtype=int)

    # Place some dummy items (item IDs: 1, 2, 3)
    # Item 1: A flat box at the bottom
    dummy_space_grid[1:4, 1:3, 0:1] = 1  # Occupies x from 1-3, y from 1-2, z from 0-0

    # Item 2: A taller box next to item 1
    dummy_space_grid[1:3, 4:6, 0:3] = 2  # Occupies x from 1-2, y from 4-5, z from 0-2
    
    # Item 3: On top of a part of item 1
    dummy_space_grid[1:3, 1:3, 1:2] = 3  # Occupies x from 1-2, y from 1-2, z from 1-1

    # Custom colors for specific items
    custom_item_colors = {
        1: 'blue',  # Item ID 1 will be blue
        3: (0.1, 0.9, 0.1, 0.7) # Item ID 3 will be greenish (RGBA)
        # Item ID 2 will get a random color
    }

    print(f"Visualizing dummy_space_grid with shape: {dummy_space_grid.shape}")
    print(f"Unique item IDs in grid: {np.unique(dummy_space_grid[dummy_space_grid > 0])}")
    
    # The container_dims for visualization should match the grid's shape.
    visualize_pallet_matplotlib(dummy_space_grid, 
                                container_dims=(grid_l, grid_w, grid_h), 
                                item_colors=custom_item_colors)
    
    # Test with an empty grid
    empty_grid = np.zeros((5,5,5), dtype=int)
    print("\nVisualizing an empty grid...")
    visualize_pallet_matplotlib(empty_grid, (5,5,5))

    plt.show() # Display all plots created

    print("\nUtils script finished.")

```
