import unittest
import numpy as np
from collections import deque
from unittest.mock import patch, MagicMock, ANY
import tensorflow as tf # Needed for DQNAgent import and some Keras functionalities

# Adjust sys.path for robust imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import DQNAgent

# Define default shapes and sizes for testing
DEFAULT_SPACE_SHAPE = (10, 10, 5) # L, W, H
DEFAULT_ITEM_FEATURES_SHAPE = (5,) # e.g., L, W, H, can_rotate, stackable
DEFAULT_STATE_SHAPE = (DEFAULT_SPACE_SHAPE, DEFAULT_ITEM_FEATURES_SHAPE)
DEFAULT_ACTION_SIZE = 24 # e.g., 2 orientations * 2x * 3y * 2z

def create_dummy_state(space_shape, item_features_shape):
    """Helper to create a dummy state tuple."""
    return (np.random.rand(*space_shape).astype(np.float32), 
            np.random.rand(*item_features_shape).astype(np.float32))

class TestAgentInitialization(unittest.TestCase):
    def test_agent_creation_and_hyperparameters(self):
        agent = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE,
                         learning_rate=0.01, gamma=0.9, epsilon=0.95,
                         epsilon_decay=0.99, epsilon_min=0.05,
                         replay_buffer_size=5000, batch_size=32)
        
        self.assertEqual(agent.state_space_shape, DEFAULT_SPACE_SHAPE)
        self.assertEqual(agent.state_item_features_shape, DEFAULT_ITEM_FEATURES_SHAPE)
        self.assertEqual(agent.action_size, DEFAULT_ACTION_SIZE)
        
        self.assertEqual(agent.gamma, 0.9)
        self.assertEqual(agent.epsilon, 0.95)
        self.assertEqual(agent.epsilon_decay, 0.99)
        self.assertEqual(agent.epsilon_min, 0.05)
        self.assertEqual(agent.learning_rate, 0.01) # Check if it stored it
        self.assertEqual(agent.batch_size, 32)

        self.assertIsInstance(agent.memory, deque)
        self.assertEqual(agent.memory.maxlen, 5000)
        
        self.assertIsNotNone(agent.model)
        self.assertIsInstance(agent.model, tf.keras.Model)
        self.assertIsNotNone(agent.target_model)
        self.assertIsInstance(agent.target_model, tf.keras.Model)

        # Check if target model weights are initially the same as model weights
        model_weights = agent.model.get_weights()
        target_model_weights = agent.target_model.get_weights()
        self.assertEqual(len(model_weights), len(target_model_weights))
        for i in range(len(model_weights)):
            np.testing.assert_array_equal(model_weights[i], target_model_weights[i],
                                          err_msg=f"Weight matrix {i} differs between model and target_model at init.")

class TestModelBuilding(unittest.TestCase):
    def test_model_input_output_shapes(self):
        agent = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE)
        model = agent.model

        # Check input shapes
        # model.input is a list of symbolic tensors if multiple inputs
        self.assertIsInstance(model.input, list)
        self.assertEqual(len(model.input), 2) # space_input and item_input
        
        # Input shapes include None for batch size, so we check from index 1 onwards
        # tf.TensorShape format: (None, dim1, dim2, ...)
        self.assertEqual(tuple(model.input[0].shape.as_list()[1:]), DEFAULT_SPACE_SHAPE)
        self.assertEqual(tuple(model.input[1].shape.as_list()[1:]), DEFAULT_ITEM_FEATURES_SHAPE)

        # Check output shape (None for batch size, action_size for Q-values)
        self.assertEqual(tuple(model.output.shape.as_list()), (None, DEFAULT_ACTION_SIZE))

class TestRemember(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 5
        self.agent = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE, replay_buffer_size=self.buffer_size)

    def test_remember_adds_to_memory(self):
        self.assertEqual(len(self.agent.memory), 0)
        dummy_s = create_dummy_state(*DEFAULT_STATE_SHAPE)
        self.agent.remember(dummy_s, 0, 1.0, dummy_s, False)
        self.assertEqual(len(self.agent.memory), 1)
        
        experience = self.agent.memory[0]
        self.assertEqual(experience[0], dummy_s) # state
        self.assertEqual(experience[1], 0)       # action
        self.assertEqual(experience[2], 1.0)     # reward
        self.assertEqual(experience[3], dummy_s) # next_state
        self.assertEqual(experience[4], False)   # done

    def test_memory_respects_max_size(self):
        dummy_s = create_dummy_state(*DEFAULT_STATE_SHAPE)
        for i in range(self.buffer_size + 2): # Add more than buffer size
            self.agent.remember(dummy_s, i, float(i), dummy_s, False)
        
        self.assertEqual(len(self.agent.memory), self.buffer_size)
        # Check if the oldest experiences were discarded (e.g., action of the first item)
        # The first item remembered would have action 0, second action 1.
        # After buffer_size + 2 items, the items with actions 0 and 1 should be gone.
        actions_in_memory = [exp[1] for exp in self.agent.memory]
        self.assertNotIn(0, actions_in_memory)
        self.assertNotIn(1, actions_in_memory)
        self.assertIn(2, actions_in_memory) # First item that should remain

class TestAct(unittest.TestCase):
    def setUp(self):
        # Patch _build_model to prevent actual model creation for some tests
        # If we need the model, we won't use this patch or will manage it per test.
        self.patcher = patch.object(DQNAgent, '_build_model', MagicMock(return_value=MagicMock(spec=tf.keras.Model)))
        self.mock_build_model = self.patcher.start()
        
        self.agent = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE)
        # Assign a mock model that we can control for predict
        self.agent.model = MagicMock(spec=tf.keras.Model)

    def tearDown(self):
        self.patcher.stop()

    def test_act_explore(self):
        self.agent.epsilon = 1.0 # Force exploration
        state = create_dummy_state(*DEFAULT_STATE_SHAPE)
        
        actions_chosen = [self.agent.act(state) for _ in range(50)] # Sample multiple actions
        
        self.assertTrue(all(0 <= a < DEFAULT_ACTION_SIZE for a in actions_chosen))
        # Check if actions are not all the same (probabilistic, could fail rarely for small action_size)
        if DEFAULT_ACTION_SIZE > 1:
             self.assertTrue(len(set(actions_chosen)) > 1, "Agent with epsilon=1.0 should explore diverse actions.")
        else: # If action_size is 1, it will always be the same action.
            self.assertTrue(len(set(actions_chosen)) == 1)


    def test_act_exploit(self):
        self.agent.epsilon = 0.0 # Force exploitation
        state = create_dummy_state(*DEFAULT_STATE_SHAPE)
        
        # Configure the mock model's predict method
        # Example: Q-values where action index 2 is highest for action_size > 2
        q_values = np.random.rand(1, DEFAULT_ACTION_SIZE).astype(np.float32)
        if DEFAULT_ACTION_SIZE > 2:
            expected_action = 2
            q_values[0, expected_action] = 1.0 # Max Q-value for action 2
            q_values[0, (expected_action + 1)%DEFAULT_ACTION_SIZE ] = 0.5 
        else: # Handle smaller action spaces
            expected_action = np.argmax(q_values[0])

        self.agent.model.predict.return_value = q_values
        
        action = self.agent.act(state)
        
        self.assertEqual(action, expected_action)
        # Check that predict was called with the correct state format
        self.agent.model.predict.assert_called_once()
        call_args = self.agent.model.predict.call_args[0][0] # Get the first positional argument list
        self.assertEqual(len(call_args), 2) # space_input, item_input
        np.testing.assert_array_equal(call_args[0], np.expand_dims(state[0], axis=0))
        np.testing.assert_array_equal(call_args[1], np.expand_dims(state[1], axis=0))


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.batch_s = 2
        # For replay, we need actual models to get weights, so don't mock _build_model here.
        self.agent = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE, batch_size=self.batch_s)
        
        # Pre-populate memory
        for _ in range(self.batch_s * 2): # Add more than batch_size experiences
            state = create_dummy_state(*DEFAULT_STATE_SHAPE)
            next_state = create_dummy_state(*DEFAULT_STATE_SHAPE)
            # Action, reward, done can be arbitrary for this mechanics test
            self.agent.remember(state, np.random.randint(0,DEFAULT_ACTION_SIZE), 1.0, next_state, False)

    def test_replay_does_not_run_if_buffer_too_small(self):
        agent_small_mem = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE, batch_size=5)
        # Add only one experience, less than batch_size
        agent_small_mem.remember(create_dummy_state(*DEFAULT_STATE_SHAPE), 0, 1, create_dummy_state(*DEFAULT_STATE_SHAPE), False)
        
        with patch.object(agent_small_mem.model, 'fit', wraps=agent_small_mem.model.fit) as mock_fit:
            agent_small_mem.replay()
            mock_fit.assert_not_called()

    @patch.object(tf.keras.Model, 'fit') # Mock model.fit directly on the class
    @patch.object(tf.keras.Model, 'predict') # Mock model.predict
    def test_replay_calls_fit_with_correct_args_when_buffer_sufficient(self, mock_predict, mock_fit):
        # Configure mock_predict for both model and target_model
        # Each call to predict should return a batch of Q-values.
        # (batch_size, action_size)
        dummy_q_values = np.random.rand(self.agent.batch_size, self.agent.action_size).astype(np.float32)
        mock_predict.return_value = dummy_q_values

        self.agent.replay() # Should call fit

        mock_fit.assert_called_once()
        
        # Check arguments passed to fit
        # fit_args = mock_fit.call_args[0] # Positional arguments
        fit_kwargs = mock_fit.call_args[1] # Keyword arguments
        
        # inputs to fit: {'space_input': ..., 'item_input': ...}
        # targets to fit: np.array of shape (batch_size, action_size)
        self.assertIn('x', fit_kwargs) # Keras uses 'x' for inputs dictionary
        self.assertIn('y', fit_kwargs) # Keras uses 'y' for targets
        
        inputs_dict = fit_kwargs['x']
        targets_array = fit_kwargs['y']

        self.assertIsInstance(inputs_dict, dict)
        self.assertIn('space_input', inputs_dict)
        self.assertIn('item_input', inputs_dict)
        self.assertEqual(inputs_dict['space_input'].shape, (self.agent.batch_size, *DEFAULT_SPACE_SHAPE))
        self.assertEqual(inputs_dict['item_input'].shape, (self.agent.batch_size, *DEFAULT_ITEM_FEATURES_SHAPE))
        
        self.assertIsInstance(targets_array, np.ndarray)
        self.assertEqual(targets_array.shape, (self.agent.batch_size, self.agent.action_size))
        
        self.assertEqual(fit_kwargs.get('batch_size'), self.agent.batch_size)
        self.assertEqual(fit_kwargs.get('epochs'), 1)
        self.assertEqual(fit_kwargs.get('verbose'), 0)


    def test_replay_changes_model_weights(self):
        # This is a more integration-style test for replay
        initial_weights = [w.copy() for w in self.agent.model.get_weights()]
        
        # Ensure epsilon is low enough that predictions are somewhat stable if model changes
        self.agent.epsilon = 0.01 
        
        self.agent.replay() # This should trigger learning
        
        updated_weights = self.agent.model.get_weights()
        
        weight_changed = False
        if len(initial_weights) == 0 and len(updated_weights) == 0: # Model has no weights (unlikely for DQN)
             pass # Cannot compare
        elif len(initial_weights) != len(updated_weights): # Should not happen
             weight_changed = True 
        else:
            for i in range(len(initial_weights)):
                if not np.array_equal(initial_weights[i], updated_weights[i]):
                    weight_changed = True
                    break
        
        self.assertTrue(weight_changed, "Model weights did not change after replay call. "
                                       "Check learning rate, loss calculation, or if fit was effectively called.")


class TestUpdateTargetModel(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(DEFAULT_STATE_SHAPE, DEFAULT_ACTION_SIZE)

    def test_update_target_model_copies_weights(self):
        # Modify some weights in the main model
        original_model_weights = self.agent.model.get_weights()
        if not original_model_weights: # Skip if model has no weights (e.g., not built)
            self.skipTest("Model has no weights to modify.")

        # Create new dummy weights (e.g., by adding random noise or setting to different values)
        modified_model_weights = [w + 0.1 for w in original_model_weights]
        self.agent.model.set_weights(modified_model_weights)

        # Ensure they are different before update
        current_target_weights = self.agent.target_model.get_weights()
        weights_differ = False
        if len(modified_model_weights) == len(current_target_weights):
            for i in range(len(modified_model_weights)):
                if not np.array_equal(modified_model_weights[i], current_target_weights[i]):
                    weights_differ = True
                    break
        else: # Lengths differ, so they are different
            weights_differ = True
        self.assertTrue(weights_differ, "Model and Target Model weights were already same before update call.")

        # Call update
        self.agent.update_target_model()
        updated_target_weights = self.agent.target_model.get_weights()

        self.assertEqual(len(modified_model_weights), len(updated_target_weights))
        for i in range(len(modified_model_weights)):
            np.testing.assert_array_equal(modified_model_weights[i], updated_target_weights[i],
                                          err_msg=f"Weight matrix {i} differs after target model update.")

if __name__ == '__main__':
    unittest.main()
```
