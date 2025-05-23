import unittest
import numpy as np
from unittest.mock import patch, MagicMock, ANY # ANY is useful for arguments you don't care about

# Adjust sys.path for robust imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import calculate_space_utilization, visualize_pallet_matplotlib
import matplotlib.pyplot as plt # Import to allow patching its members

class TestCalculateSpaceUtilization(unittest.TestCase):
    def test_typical_utilization(self):
        self.assertAlmostEqual(calculate_space_utilization([10, 20, 30], 100), 0.6)

    def test_empty_placed_volumes(self):
        self.assertAlmostEqual(calculate_space_utilization([], 100), 0.0)

    def test_zero_container_volume(self):
        self.assertAlmostEqual(calculate_space_utilization([10, 20], 0), 0.0)

    def test_volume_exceeds_container(self):
        # This scenario implies an issue elsewhere (e.g. items larger than container)
        # but the utility function should still calculate the ratio correctly.
        self.assertAlmostEqual(calculate_space_utilization([100, 20], 100), 1.2)

class TestVisualizePalletMatplotlib(unittest.TestCase):
    @patch('matplotlib.pyplot.show') # Mock plt.show
    def test_runs_without_error_and_calls_show(self, mock_show):
        dummy_space_grid = np.zeros((10, 10, 5), dtype=int)
        dummy_space_grid[0:2, 0:3, 0:1] = 1 # Item 1
        container_dims = (10,10,5) # Corresponds to space_grid.shape for visualization
        
        ran_successfully = False
        try:
            visualize_pallet_matplotlib(dummy_space_grid, container_dims)
            ran_successfully = True
        except Exception as e:
            # This print is helpful for debugging if a test fails unexpectedly
            print(f"Visualization test 'test_runs_without_error_and_calls_show' failed with exception: {e}")
        
        self.assertTrue(ran_successfully, "visualize_pallet_matplotlib raised an exception.")
        mock_show.assert_called_once() # Verify plt.show() was called

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure') # Patch plt.figure to control the Figure object
    def test_voxels_called_with_items(self, mock_figure, mock_show):
        # Create a mock Figure instance
        mock_fig_instance = MagicMock()
        # Create a mock Axes3D instance that figure().add_subplot() will return
        mock_ax_instance = MagicMock()
        
        # Configure mock_figure to return our mock_fig_instance
        mock_figure.return_value = mock_fig_instance
        # Configure mock_fig_instance.add_subplot to return our mock_ax_instance
        mock_fig_instance.add_subplot.return_value = mock_ax_instance

        dummy_space_grid = np.ones((5,5,5), dtype=int) # A grid full of item ID 1
        container_dims = (5,5,5)
        
        visualize_pallet_matplotlib(dummy_space_grid, container_dims)
        
        mock_figure.assert_called_once() # Check if plt.figure() was called
        mock_fig_instance.add_subplot.assert_called_once_with(111, projection='3d')
        mock_ax_instance.voxels.assert_called_once() # Check if ax.voxels() was called
        
        # More specific check for ax.voxels arguments if needed:
        # args, kwargs = mock_ax_instance.voxels.call_args
        # self.assertTrue(np.array_equal(args[0], dummy_space_grid.astype(bool))) # Check the filled mask
        # self.assertTrue('facecolors' in kwargs)
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_voxels_not_called_for_empty_grid(self, mock_figure, mock_show):
        mock_fig_instance = MagicMock()
        mock_ax_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        mock_fig_instance.add_subplot.return_value = mock_ax_instance

        empty_space_grid = np.zeros((5,5,5), dtype=int) # An empty grid
        container_dims = (5,5,5)
        
        visualize_pallet_matplotlib(empty_space_grid, container_dims)
        
        mock_figure.assert_called_once()
        mock_fig_instance.add_subplot.assert_called_once_with(111, projection='3d')
        mock_ax_instance.voxels.assert_not_called() # Voxels should not be called if grid is empty
        
        # Legend should also not be called if no items
        mock_ax_instance.legend.assert_not_called()

        mock_show.assert_called_once() # show is still called to display the empty plot box

    @patch('matplotlib.pyplot.show')
    def test_handles_invalid_grid_gracefully(self, mock_show):
        # Test with a non-3D grid, should print an error and not raise exception
        invalid_grid = np.zeros((5,5), dtype=int) 
        container_dims = (5,5,0) # Dummy dims

        with patch('builtins.print') as mock_print:
            visualize_pallet_matplotlib(invalid_grid, container_dims)
            mock_print.assert_any_call("Error: space_grid must be a 3D NumPy array.")
        
        mock_show.assert_not_called() # Show might not be called if it exits early due to error handling

if __name__ == '__main__':
    unittest.main()
```
