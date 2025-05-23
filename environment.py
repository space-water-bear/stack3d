import numpy as np

class PalletEnvironment:
    def __init__(self, container_dims, cargo_items):
        """
        Initializes the pallet loading environment.

        Args:
            container_dims: A tuple (length, width, height) e.g., (1200, 1000, 1000).
            cargo_items: A list of dictionaries, where each dictionary represents a cargo item.
        """
        self.length, self.width, self.height = container_dims
        self.cargo_items = cargo_items
        
        # Initialize self.space with direct units. This might be large.
        # If smallest unit is e.g. 1mm, then 1200x1000x1000 is 1.2 billion cells.
        # Consider scaling if this is too memory intensive. For now, direct.
        # Max dimension from example data is 410. Max container dim is 1200.
        # Let's assume for now that the dimensions are in consistent units (e.g., mm).
        try:
            self.space = np.zeros((self.length, self.width, self.height), dtype=np.int32)
        except MemoryError:
            print("Error: Direct unit dimensions for self.space are too large, leading to MemoryError.")
            print("Consider scaling down the dimensions (e.g., by a common factor like 10 or smallest item dimension).")
            # As a fallback for now, let's try scaling by 10 if direct allocation fails.
            # This should be a parameter or a more sophisticated auto-scaling in a real scenario.
            self.scaling_factor = 10 
            print(f"Attempting to scale dimensions by a factor of {self.scaling_factor}.")
            self.scaled_length = self.length // self.scaling_factor
            self.scaled_width = self.width // self.scaling_factor
            self.scaled_height = self.height // self.scaling_factor
            self.space = np.zeros((self.scaled_length, self.scaled_width, self.scaled_height), dtype=np.int32)
            print(f"Using scaled space: ({self.scaled_length}, {self.scaled_width}, {self.scaled_height})")
            # We'll need to scale item dimensions and positions too if this path is taken.
            # For now, the rest of the code will assume unscaled, but this highlights the issue.
        else:
            self.scaling_factor = 1 # No scaling applied initially
            self.scaled_length = self.length
            self.scaled_width = self.width
            self.scaled_height = self.height


        self.current_item_index = 0

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        self.space = np.zeros((self.scaled_length, self.scaled_width, self.scaled_height), dtype=np.int32)
        self.current_item_index = 0
        # Potentially shuffle self.cargo_items here if desired for training variability
        return self.get_state()

    def get_state(self):
        """
        Gets the current state of the environment.

        Returns:
            A tuple: (self.space, current_item_features).
            current_item_features will be None if all items are placed.
        """
        if self.current_item_index >= len(self.cargo_items):
            current_item = None # Or some representation of "no item"
        else:
            current_item = self.cargo_items[self.current_item_index]
        
        # For DQN, item features should be numerical.
        # Let's select a few key features for now.
        # This needs to be consistent for the DQN input.
        if current_item:
            # Example: L, W, H, can_rotate (binary), stackable (binary)
            # Ensure all values are scaled if self.scaling_factor != 1
            item_l = current_item["length"]
            item_w = current_item["width"]
            item_h = current_item["height"]
            if self.scaling_factor != 1:
                item_l //= self.scaling_factor
                item_w //= self.scaling_factor
                item_h //= self.scaling_factor

            processed_item_features = np.array([
                item_l,
                item_w,
                item_h,
                1 if current_item["can_rotate"] else 0,
                1 if current_item["stackable"] else 0,
            ])
        else:
            # Provide a zero vector or similar if no item (e.g., episode end)
            # The size should match the feature vector of an actual item.
            processed_item_features = np.zeros(5) 
            
        return (self.space.copy(), processed_item_features)

    def _get_item_dims_after_rotation(self, item, orientation):
        """ Helper to get item dimensions based on orientation """
        length = item["length"]
        width = item["width"]
        height = item["height"]

        if item["can_rotate"] and orientation == 1: # 90-degree rotation (swap L and W)
            actual_length, actual_width = width, length
        else: # orientation 0 or cannot rotate
            actual_length, actual_width = length, width
        
        actual_height = height # Height is fixed for this iteration

        if self.scaling_factor != 1:
            actual_length //= self.scaling_factor
            actual_width //= self.scaling_factor
            actual_height //= self.scaling_factor
        
        return int(actual_length), int(actual_width), int(actual_height)


    def _is_valid_placement(self, item_dims, position):
        """
        Checks if placing an item with item_dims at position is valid.

        Args:
            item_dims: (length, width, height) of the item after rotation and scaling.
            position: (x, y, z) coordinates (bottom-front-left) after scaling.
        """
        item_l, item_w, item_h = item_dims
        x, y, z = position

        # 1. Check container boundaries
        if not (0 <= x < self.scaled_length and \
                0 <= y < self.scaled_width and \
                0 <= z < self.scaled_height):
            return False # Position itself is out of bounds (e.g. negative coords)

        if not (x + item_l <= self.scaled_length and \
                y + item_w <= self.scaled_width and \
                z + item_h <= self.scaled_height):
            return False # Item extends beyond container boundaries

        # 2. Check for collision with already placed items
        # The space the item would occupy must be all zeros.
        try:
            if np.any(self.space[x : x + item_l, y : y + item_w, z : z + item_h] != 0):
                return False # Collision
        except IndexError:
            # This can happen if calculations for slicing go wrong, defensive check.
            return False


        # 3. Stacking constraints (basic for now)
        if z > 0:
            # Check if the area below the item is solid (not empty space)
            # This is a simplified check: assumes any non-zero value means it's supported.
            # A more advanced check would verify full support.
            if item_l == 0 or item_w == 0 : # avoid issues with empty slice for support_area
                 return False # Cannot place item with zero dimension on an existing item
            
            support_area = self.space[x : x + item_l, y : y + item_w, z - 1]
            if np.any(support_area == 0):
                # This basic check means if any part of the base is unsupported, it's invalid.
                # More realistically, one might check percentage of base supported.
                return False 
        
        return True

    def _place_item(self, item_id_val, item_dims, position):
        """
        Places the item in self.space.

        Args:
            item_id_val: The value to mark in the space grid (e.g., index + 1).
            item_dims: (length, width, height) of the item after rotation and scaling.
            position: (x, y, z) coordinates after scaling.
        """
        item_l, item_w, item_h = item_dims
        x, y, z = position
        
        self.space[x : x + item_l, y : y + item_w, z : z + item_h] = item_id_val

    def _calculate_reward(self, item_dims, successfully_placed):
        """
        Calculates the reward.

        Args:
            item_dims: (length, width, height) of the item (original, unscaled for volume calc).
            successfully_placed: Boolean indicating if placement was successful.
        """
        if successfully_placed:
            # Reward is the volume of the item (using original dimensions for meaningful reward)
            # Note: item_dims passed here should ideally be original for this calculation.
            # If scaled_dims are passed, the reward would also be scaled.
            # Let's assume for now the caller will handle passing appropriate dims.
            # For simplicity, if item_dims are scaled, reward is scaled volume.
            return item_dims[0] * item_dims[1] * item_dims[2] 
        else:
            return -1.0 # Penalty for invalid placement


    def step(self, action):
        """
        Executes an action in the environment.

        Args:
            action: A tuple: (position_x, position_y, position_z, orientation_id).
                    It's assumed this action is for self.cargo_items[self.current_item_index].
        
        Returns:
            (next_state, reward, done, info_dict)
        """
        if self.current_item_index >= len(self.cargo_items):
            # All items have been processed (or attempted)
            next_state = self.get_state() # State with no current item
            return next_state, 0, True, {}

        current_item_obj = self.cargo_items[self.current_item_index]
        
        # Action components (assuming action space is simplified for now)
        # For a DQN, action would typically be an integer mapped to these choices.
        # Here, we assume action directly provides (pos_x, pos_y, pos_z, orientation)
        pos_x, pos_y, pos_z, orientation = action

        # Apply scaling to position from action if needed
        if self.scaling_factor != 1:
            pos_x //= self.scaling_factor
            pos_y //= self.scaling_factor
            pos_z //= self.scaling_factor
        
        # Get actual dimensions of the item based on orientation and potential scaling
        actual_item_dims_scaled = self._get_item_dims_after_rotation(current_item_obj, orientation)

        # Use original dimensions for reward calculation if placement is successful
        original_dims_for_reward = (current_item_obj["length"], current_item_obj["width"], current_item_obj["height"])


        if self._is_valid_placement(actual_item_dims_scaled, (pos_x, pos_y, pos_z)):
            # Use (self.current_item_index + 1) as a simple ID for the grid. 0 is empty.
            item_id_in_grid = self.current_item_index + 1 
            self._place_item(item_id_in_grid, actual_item_dims_scaled, (pos_x, pos_y, pos_z))
            
            # Calculate reward based on original item volume
            reward = self._calculate_reward(original_dims_for_reward, successfully_placed=True)
            self.current_item_index += 1
            successfully_placed = True
        else:
            reward = self._calculate_reward(original_dims_for_reward, successfully_placed=False)
            successfully_placed = False
            # Optional: End episode on first invalid placement, or allow agent to retry.
            # For now, we don't end the episode here, just give a penalty.

        done = self.current_item_index >= len(self.cargo_items)
        next_state = self.get_state()
        info_dict = {"placed": successfully_placed, "item_id_placed": current_item_obj["id"] if successfully_placed else None}

        return next_state, reward, done, info_dict

    def get_action_space_sample(self):
        """
        Returns a sample action. Useful for agent testing.
        This is a placeholder and needs to be more sophisticated for a real agent.
        """
        # Randomly pick an item (though step processes current_item_index)
        # item_idx = np.random.randint(len(self.cargo_items)) # Not used by step directly

        # Random position (within scaled bounds)
        rand_x = np.random.randint(0, self.scaled_length)
        rand_y = np.random.randint(0, self.scaled_width)
        rand_z = np.random.randint(0, self.scaled_height)
        
        # Random orientation (0 or 1 if item can rotate)
        current_item_obj = self.cargo_items[self.current_item_index % len(self.cargo_items)] # handle edge case if index is too high
        rand_orientation = 0
        if current_item_obj["can_rotate"]:
            rand_orientation = np.random.randint(0, 2) # 0 or 1

        # Note: The positions here are scaled if scaling is active.
        # If the agent is going to output unscaled actions, they need to be scaled in step().
        # Current step() implementation assumes action positions might need scaling.
        return (rand_x * self.scaling_factor, # Return unscaled, step will scale
                rand_y * self.scaling_factor, 
                rand_z * self.scaling_factor, 
                rand_orientation)

# Example Usage (can be moved to a test file or main script later)
if __name__ == '__main__':
    cargo_data = [
        {"id": "苏打天然水饮品", "length": 301.0, "width": 182.0, "height": 204.0, "weight": 6.15, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
        {"id": "珠江啤酒", "length": 410.0, "width": 273.0, "height": 125.0, "weight": 8.5, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
        {"id": "ItemC", "length": 200.0, "width": 150.0, "height": 100.0, "weight": 3.0, "can_rotate": False, "stackable": True, "stack_face": "width&length"},
    ]
    container_dimensions = (1200, 1000, 1000) # L, W, H in mm

    # Test with potential memory error and fallback
    # To force memory error for testing, one might use unrealistic large direct dimensions
    # For now, use the given dimensions. If they are too large, the fallback will trigger.
    # container_dimensions_huge = (120000, 100000, 100000) # This would likely cause MemoryError
    
    env = PalletEnvironment(container_dims=container_dimensions, cargo_items=cargo_data)
    
    print(f"Initialized environment with space shape: {env.space.shape}")
    print(f"Scaling factor applied: {env.scaling_factor}")

    initial_state = env.reset()
    print("Initial state retrieved.")
    print(f"Space shape: {initial_state[0].shape}, Item features: {initial_state[1]}")

    # Test a sample action
    # Action: (pos_x, pos_y, pos_z, orientation) - unscaled
    # Let's try to place the first item ("苏打天然水饮品")
    # Original dims: L=301, W=182, H=204
    # Action1: Place at (0,0,0) with no rotation (orientation 0)
    action1 = (0, 0, 0, 0) 
    print(f"\nAttempting action 1: {action1} for item '{env.cargo_items[env.current_item_index]['id']}'")
    next_state, reward, done, info = env.step(action1)
    print(f"Action 1 Result: Reward={reward}, Done={done}, Info={info}")
    print(f"Item features for next step: {next_state[1]}")

    # Check if the item was placed in the grid (example: sum of non-zero elements)
    if env.scaling_factor == 1:
        expected_volume_cells_item1 = (301 * 182 * 204)
    else:
        # Scaled volume calculation
        s = env.scaling_factor
        l, w, h = cargo_data[0]['length']//s, cargo_data[0]['width']//s, cargo_data[0]['height']//s
        expected_volume_cells_item1 = l*w*h
    
    print(f"Number of occupied cells in grid: {np.count_nonzero(env.space)}")
    # print(f"Expected cell occupancy for item 1 (approx): {expected_volume_cells_item1}") # This is volume, not count of cells if item_id is used

    item1_id_in_grid = 1 # current_item_index (0) + 1
    print(f"Number of cells marked with ID {item1_id_in_grid}: {np.sum(env.space == item1_id_in_grid)}")


    # Action2: Try to place the second item ("珠江啤酒")
    # Original dims: L=410, W=273, H=125
    # Place it next to the first one on the x-axis.
    # First item (unrotated, unscaled): L=301, W=182, H=204
    # Position for second item: x = 301 (if unscaled), y=0, z=0, orientation=0
    pos_x_item2 = int(cargo_data[0]["length"]) # x-coordinate right after the first item
    action2 = (pos_x_item2, 0, 0, 0)
    print(f"\nAttempting action 2: {action2} for item '{env.cargo_items[env.current_item_index]['id']}'")
    next_state, reward, done, info = env.step(action2)
    print(f"Action 2 Result: Reward={reward}, Done={done}, Info={info}")
    print(f"Item features for next step: {next_state[1]}")
    
    item2_id_in_grid = 2 # current_item_index (1) + 1
    print(f"Number of cells marked with ID {item2_id_in_grid}: {np.sum(env.space == item2_id_in_grid)}")
    print(f"Total occupied cells in grid: {np.count_nonzero(env.space)}")

    # Action3: Invalid placement (collision with item 1)
    action3 = (0, 0, 0, 0) # Try to place item "ItemC" at (0,0,0)
    print(f"\nAttempting action 3 (invalid): {action3} for item '{env.cargo_items[env.current_item_index]['id']}'")
    next_state, reward, done, info = env.step(action3)
    print(f"Action 3 Result: Reward={reward}, Done={done}, Info={info}")
    print(f"Total occupied cells in grid (should be same as before): {np.count_nonzero(env.space)}")
    
    # Action4: Place third item ("ItemC")
    # Original dims: L=200, W=150, H=100
    # Place it on top of the first item.
    # Z-coordinate should be height of first item: H=204
    pos_z_item3 = int(cargo_data[0]["height"])
    action4 = (0, 0, pos_z_item3, 0)
    print(f"\nAttempting action 4: {action4} for item '{env.cargo_items[env.current_item_index]['id']}'")
    next_state, reward, done, info = env.step(action4)
    print(f"Action 4 Result: Reward={reward}, Done={done}, Info={info}")
    
    item3_id_in_grid = 3 # current_item_index (2) + 1
    print(f"Number of cells marked with ID {item3_id_in_grid}: {np.sum(env.space == item3_id_in_grid)}")
    print(f"Total occupied cells in grid: {np.count_nonzero(env.space)}")
    print(f"Done flag is: {done} (should be True as all 3 items processed)")

    # Test get_action_space_sample
    env.reset()
    sample_action = env.get_action_space_sample()
    print(f"\nSample action: {sample_action}")
    # Note: The sample action's item is implicitly the first one after reset.
    # The position from get_action_space_sample is unscaled.
    print(f"This sample action is for item: {env.cargo_items[0]['id']}")

    print("\nEnvironment implementation draft complete.")
    print("Considerations:")
    print("- Memory usage for self.space if dimensions are large (e.g., mm units for large containers). Fallback scaling is basic.")
    print("- Action space definition for a DQN agent (current step() takes explicit components).")
    print("- Reward shaping for more complex goals.")
    print("- Sophistication of _is_valid_placement (e.g., full support check for stacking).")
    print("- Rotations are currently only L-W swap. Height fixed.")

```
