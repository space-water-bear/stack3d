import unittest
import numpy as np
# Assuming environment.py is in the parent directory or PYTHONPATH is set
# For robust imports, you might need to adjust sys.path if running tests directly
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import PalletEnvironment

# Define CARGO_DATA and CONTAINER_DIMS for tests (can be simplified versions)
SIMPLE_CARGO_DATA = [
    {"id": "item1", "length": 20.0, "width": 30.0, "height": 10.0, "can_rotate": False, "stackable": True, "stack_face": "width&length", "volume": 20*30*10},
    {"id": "item2_rotatable", "length": 25.0, "width": 15.0, "height": 5.0, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 25*15*5},
    {"id": "item3_stackable", "length": 10.0, "width": 10.0, "height": 8.0, "can_rotate": False, "stackable": True, "stack_face": "width&length", "volume": 10*10*8},
]
# Add volume to each item for easier access
for item in SIMPLE_CARGO_DATA:
    if 'volume' not in item: # Should be pre-calculated, but as a fallback
        item['volume'] = item['length'] * item['width'] * item['height']


DEFAULT_CONTAINER_DIMS = (100, 100, 50) # L, W, H in mm
SCALED_CONTAINER_DIMS = (12000, 10000, 10000) # These should trigger scaling
EXPECTED_SCALING_FACTOR = 10 # Based on PalletEnvironment's MemoryError fallback

class TestEnvironmentInitialization(unittest.TestCase):
    def test_basic_init(self):
        env = PalletEnvironment(DEFAULT_CONTAINER_DIMS, SIMPLE_CARGO_DATA)
        self.assertEqual(env.length, DEFAULT_CONTAINER_DIMS[0])
        self.assertEqual(env.width, DEFAULT_CONTAINER_DIMS[1])
        self.assertEqual(env.height, DEFAULT_CONTAINER_DIMS[2])
        self.assertEqual(env.scaling_factor, 1) # No scaling expected
        self.assertEqual(env.space.shape, (DEFAULT_CONTAINER_DIMS[0], DEFAULT_CONTAINER_DIMS[1], DEFAULT_CONTAINER_DIMS[2]))
        self.assertTrue(np.all(env.space == 0))
        self.assertEqual(len(env.cargo_items), len(SIMPLE_CARGO_DATA))
        self.assertEqual(env.cargo_items[0]['id'], SIMPLE_CARGO_DATA[0]['id'])
        self.assertEqual(env.current_item_index, 0)

    def test_scaled_init(self):
        env = PalletEnvironment(SCALED_CONTAINER_DIMS, SIMPLE_CARGO_DATA)
        # Check if scaling was triggered (this depends on memory available during test run)
        # For now, we assume the fallback scaling logic in PalletEnvironment works as intended
        # and that these dimensions are large enough to trigger it.
        if env.scaling_factor != 1: # If scaling actually happened
            self.assertEqual(env.scaling_factor, EXPECTED_SCALING_FACTOR)
            self.assertEqual(env.length, SCALED_CONTAINER_DIMS[0]) # Original dims stored
            self.assertEqual(env.width, SCALED_CONTAINER_DIMS[1])
            self.assertEqual(env.height, SCALED_CONTAINER_DIMS[2])
            
            expected_scaled_shape = (
                SCALED_CONTAINER_DIMS[0] // EXPECTED_SCALING_FACTOR,
                SCALED_CONTAINER_DIMS[1] // EXPECTED_SCALING_FACTOR,
                SCALED_CONTAINER_DIMS[2] // EXPECTED_SCALING_FACTOR
            )
            self.assertEqual(env.space.shape, expected_scaled_shape)
            self.assertEqual(env.scaled_length, expected_scaled_shape[0])
            self.assertEqual(env.scaled_width, expected_scaled_shape[1])
            self.assertEqual(env.scaled_height, expected_scaled_shape[2])

        else: # If scaling did not happen (e.g. machine has vast memory)
            print("\nWarning: Scaled initialization test did not trigger scaling. "
                  "The test environment might have enough memory for direct allocation "
                  "even with large dimensions. Assertions for scaling factor and scaled shape will be skipped.")
            self.assertEqual(env.space.shape, SCALED_CONTAINER_DIMS)

        self.assertTrue(np.all(env.space == 0))
        self.assertEqual(len(env.cargo_items), len(SIMPLE_CARGO_DATA))


class TestItemPlacement(unittest.TestCase):
    def setUp(self):
        # Using smaller, non-scaled dimensions for most placement tests for simplicity
        self.container_dims = (50, 40, 30) 
        self.cargo = [
            {"id": "itemA", "length": 20.0, "width": 15.0, "height": 10.0, "can_rotate": False, "stackable": True, "volume": 20*15*10},
            {"id": "itemB_rot", "length": 10.0, "width": 12.0, "height": 8.0, "can_rotate": True, "stackable": True, "volume": 10*12*8},
        ]
        for item in self.cargo: # Ensure volume is present
            if 'volume' not in item: item['volume'] = item['length']*item['width']*item['height']

        self.env = PalletEnvironment(self.container_dims, self.cargo)
        self.initial_space_sum = np.sum(self.env.space)


    def test_valid_placement(self):
        action = (0, 0, 0, 0) # Place itemA at origin, no rotation
        item_to_place = self.cargo[0]
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(reward, item_to_place['volume'])
        self.assertTrue(info['placed'])
        self.assertEqual(info['item_id_placed'], item_to_place['id'])
        self.assertEqual(self.env.current_item_index, 1)
        
        # Check space update
        # Item ID in grid is current_item_index (which was 0 before placement) + 1
        item_id_in_grid = 1 
        expected_region = self.env.space[
            0:int(item_to_place['length']), 
            0:int(item_to_place['width']), 
            0:int(item_to_place['height'])
        ]
        self.assertTrue(np.all(expected_region == item_id_in_grid))
        # Ensure other areas are still 0
        self.assertEqual(np.sum(self.env.space == item_id_in_grid), 
                         item_to_place['length'] * item_to_place['width'] * item_to_place['height'])

    def test_invalid_placement_out_of_bounds(self):
        # Action places itemA such that it goes out of bounds (e.g., x too large)
        # ItemA: L=20, W=15, H=10. Container L=50.
        # Place at x=40. 40 (pos) + 20 (len) = 60, which is > 50 (container_len)
        action = (40, 0, 0, 0) 
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(reward, -1.0) # Penalty
        self.assertFalse(info['placed'])
        self.assertIsNone(info['item_id_placed'])
        self.assertEqual(self.env.current_item_index, 0) # Should not increment for invalid physical placement
        self.assertEqual(np.sum(self.env.space), self.initial_space_sum) # Space unchanged

    def test_invalid_placement_collision(self):
        # Place itemA successfully
        action1 = (0, 0, 0, 0)
        self.env.step(action1)
        self.assertEqual(self.env.current_item_index, 1) # ItemA placed, index moves to itemB_rot

        # Attempt to place itemB_rot at the same location (collision)
        action2 = (0, 0, 0, 0) 
        item_b_original_volume = self.cargo[1]['volume'] # For reward calculation
        
        next_state, reward, done, info = self.env.step(action2)
        
        self.assertEqual(reward, -1.0) # Penalty
        self.assertFalse(info['placed'])
        self.assertEqual(self.env.current_item_index, 1) # Index still points to itemB_rot, did not advance
        
        # Space should only contain itemA (ID 1)
        item_a_id_in_grid = 1
        self.assertTrue(np.all(self.env.space[self.env.space > 0] == item_a_id_in_grid))
        # Verify itemA's volume is still there correctly
        item_a_dims = (int(self.cargo[0]['length']), int(self.cargo[0]['width']), int(self.cargo[0]['height']))
        self.assertEqual(np.sum(self.env.space == item_a_id_in_grid), item_a_dims[0]*item_a_dims[1]*item_a_dims[2])


    def test_item_rotation(self):
        # Place itemB_rot (L=10, W=12, H=8) with rotation (orientation=1)
        # Original L=10, W=12. Rotated L=12, W=10.
        action = (0, 0, 0, 1) # Place itemB_rot at origin, rotated
        
        # First, advance past itemA by placing it somewhere valid (or simulate)
        self.env.current_item_index = 1 # Manually set to test itemB_rot
        item_to_place = self.cargo[1] # This is itemB_rot
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(reward, item_to_place['volume'])
        self.assertTrue(info['placed'])
        self.assertEqual(self.env.current_item_index, 2) # Index advanced
        
        item_id_in_grid = 1 + 1 # Item index (1) + 1
        
        # Expected dimensions after rotation
        rotated_l, rotated_w, item_h = item_to_place['width'], item_to_place['length'], item_to_place['height']
        
        expected_region = self.env.space[
            0:int(rotated_l), 
            0:int(rotated_w), 
            0:int(item_h)
        ]
        self.assertTrue(np.all(expected_region == item_id_in_grid))
        self.assertEqual(np.sum(self.env.space == item_id_in_grid), 
                         rotated_l * rotated_w * item_h)

class TestReset(unittest.TestCase):
    def test_reset_environment(self):
        env = PalletEnvironment(DEFAULT_CONTAINER_DIMS, SIMPLE_CARGO_DATA)
        # Place an item
        env.step((0,0,0,0)) 
        self.assertNotEqual(np.sum(env.space), 0) # Space is occupied
        self.assertNotEqual(env.current_item_index, 0) # Index advanced
        
        initial_state_after_reset = env.reset()
        
        self.assertTrue(np.all(env.space == 0)) # Space cleared
        self.assertEqual(env.current_item_index, 0) # Index reset
        
        # Check state returned by reset
        self.assertIsInstance(initial_state_after_reset, tuple)
        self.assertEqual(initial_state_after_reset[0].shape, env.space.shape)
        self.assertTrue(np.all(initial_state_after_reset[0] == 0))
        # Check item features of the first item
        first_item = SIMPLE_CARGO_DATA[0]
        expected_features = np.array([
            first_item['length'], first_item['width'], first_item['height'], 
            0, 1 # not rotatable, stackable
        ])
        if env.scaling_factor !=1: # if scaling is active, features are scaled
            expected_features[0] //= env.scaling_factor
            expected_features[1] //= env.scaling_factor
            expected_features[2] //= env.scaling_factor

        self.assertTrue(np.array_equal(initial_state_after_reset[1], expected_features))


class TestStateRepresentation(unittest.TestCase):
    def test_get_state(self):
        env = PalletEnvironment(DEFAULT_CONTAINER_DIMS, SIMPLE_CARGO_DATA)
        state = env.get_state()

        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)
        
        # Space part of state
        self.assertIsInstance(state[0], np.ndarray)
        self.assertEqual(state[0].shape, env.space.shape)
        self.assertTrue(np.all(state[0] == env.space)) # Should be a copy

        # Item features part of state
        self.assertIsInstance(state[1], np.ndarray)
        first_item = SIMPLE_CARGO_DATA[0]
        expected_features_shape = (5,) # L,W,H,can_rotate,stackable
        self.assertEqual(state[1].shape, expected_features_shape)
        
        # Check values (assuming no scaling for this test with DEFAULT_CONTAINER_DIMS)
        self.assertEqual(env.scaling_factor, 1) # Prerequisite for direct feature check
        expected_vals_first_item = np.array([
            first_item['length'], first_item['width'], first_item['height'],
            1 if first_item['can_rotate'] else 0,
            1 if first_item['stackable'] else 0
        ])
        self.assertTrue(np.array_equal(state[1], expected_vals_first_item))

        # Place an item and check next state's item features
        env.step((0,0,0,0)) # Place first item
        next_state_info = env.get_state()
        second_item = SIMPLE_CARGO_DATA[1]
        expected_vals_second_item = np.array([
            second_item['length'], second_item['width'], second_item['height'],
            1 if second_item['can_rotate'] else 0,
            1 if second_item['stackable'] else 0
        ])
        self.assertTrue(np.array_equal(next_state_info[1], expected_vals_second_item))

        # Place all items and check "no item" state
        for _ in range(len(SIMPLE_CARGO_DATA) -1): # Already placed one
            if env.current_item_index < len(env.cargo_items) :
                 env.step((0,0,env.current_item_index * 20 ,0)) # Try to place subsequent items, avoid collision by Z
        
        final_state = env.get_state()
        self.assertTrue(np.array_equal(final_state[1], np.zeros(expected_features_shape))) # Zero vector for no item

class TestStacking(unittest.TestCase):
    def setUp(self):
        # Container: L=50, W=50, H=30. Cargo: item1 (20x30x10), item3 (10x10x8)
        self.stack_container_dims = (50, 50, 30)
        self.stack_cargo = [
            {"id": "itemA_base", "length": 20.0, "width": 30.0, "height": 10.0, "can_rotate": False, "stackable": True, "volume": 20*30*10},
            {"id": "itemC_top", "length": 10.0, "width": 10.0, "height": 8.0, "can_rotate": False, "stackable": True, "volume": 10*10*8},
        ]
        for item in self.stack_cargo: # Ensure volume
            if 'volume' not in item: item['volume'] = item['length']*item['width']*item['height']
        self.env = PalletEnvironment(self.stack_container_dims, self.stack_cargo)

    def test_valid_stacking(self):
        # Place itemA_base at (0,0,0)
        action_A = (0,0,0,0)
        item_A = self.stack_cargo[0]
        next_state_A, reward_A, done_A, info_A = self.env.step(action_A)

        self.assertEqual(reward_A, item_A['volume'])
        self.assertTrue(info_A['placed'])
        self.assertEqual(self.env.current_item_index, 1)

        # Place itemC_top on top of itemA_base
        # ItemA_base height is 10. So, z for itemC_top should be 10.
        # ItemC_top (10x10x8) can fit on itemA_base (20x30 surface at z=9)
        action_C = (0,0,10,0) # Place itemC_top at (0,0) on z=10
        item_C = self.stack_cargo[1]
        next_state_C, reward_C, done_C, info_C = self.env.step(action_C)
        
        self.assertEqual(reward_C, item_C['volume'])
        self.assertTrue(info_C['placed'])
        self.assertEqual(self.env.current_item_index, 2)

        # Check space for itemC_top (ID should be 2)
        item_C_id_in_grid = 2
        item_C_dims = (int(item_C['length']), int(item_C['width']), int(item_C['height']))
        
        expected_region_C = self.env.space[
            0:item_C_dims[0], 
            0:item_C_dims[1], 
            10 : 10 + item_C_dims[2]
        ]
        self.assertTrue(np.all(expected_region_C == item_C_id_in_grid))

    def test_invalid_stacking_no_support(self):
        # Attempt to place itemA_base (first item) at z=10 (floating)
        # This should be invalid because z>0 and nothing below it.
        action_A_floating = (0,0,10,0)
        next_state_A, reward_A, done_A, info_A = self.env.step(action_A_floating)

        self.assertEqual(reward_A, -1.0) # Penalty
        self.assertFalse(info_A['placed'])
        self.assertEqual(self.env.current_item_index, 0) # Index did not advance
        self.assertTrue(np.all(self.env.space == 0)) # Space remains empty


if __name__ == '__main__':
    unittest.main()
```
