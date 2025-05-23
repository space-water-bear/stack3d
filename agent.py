import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Disable eager execution for potential performance benefits with tf.function in model training
# tf.compat.v1.disable_eager_execution() # Only if using TF1 style graphs explicitly, Keras API usually handles this.
                                      # For TF2, eager execution is default and generally preferred for debugging.
                                      # Let's stick to default TF2 behavior unless issues arise.

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 replay_buffer_size=10000, batch_size=64):
        """
        Initializes the DQN Agent.

        Args:
            state_shape: Tuple (space_shape, item_features_shape). 
                         e.g., (((120, 100, 100), (5,)))
            action_size: Integer, the number of discrete actions.
            learning_rate: Float, learning rate for the optimizer.
            gamma: Float, discount factor for future rewards.
            epsilon: Float, initial exploration rate.
            epsilon_decay: Float, factor to decay epsilon by.
            epsilon_min: Float, minimum value for epsilon.
            replay_buffer_size: Integer, maximum size of the replay memory.
            batch_size: Integer, size of the mini-batch for training.
        """
        self.state_space_shape = state_shape[0]  # e.g., (120, 100, 100)
        self.state_item_features_shape = state_shape[1] # e.g., (5,)
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate # Store for potential use, though Adam takes it directly

        self.batch_size = batch_size
        self.memory = deque(maxlen=replay_buffer_size)

        self.model = self._build_model(self.state_space_shape, self.state_item_features_shape, action_size)
        self.target_model = self._build_model(self.state_space_shape, self.state_item_features_shape, action_size)
        self.update_target_model()

    def _build_model(self, space_shape, item_features_shape, action_size):
        """
        Builds the Q-network model.

        Args:
            space_shape: Shape of the 3D space grid input.
            item_features_shape: Shape of the item features input.
            action_size: Number of output actions.

        Returns:
            A compiled Keras Model.
        """
        # Input for the 3D space grid
        space_input = Input(shape=space_shape, name='space_input')
        # Input for item features
        item_input = Input(shape=item_features_shape, name='item_input')

        # Process space input
        flattened_space = Flatten()(space_input)

        # Concatenate flattened space input with item_input
        concatenated_inputs = Concatenate()([flattened_space, item_input])

        # Dense layers for learning
        x = Dense(256, activation='relu')(concatenated_inputs)
        x = Dense(128, activation='relu')(x)
        # x = Dense(64, activation='relu')(x) # Optional extra layer

        # Output layer: Q-values for each action
        output_q_values = Dense(action_size, activation='linear', name='q_values')(x)

        # Create the model
        model = Model(inputs=[space_input, item_input], outputs=output_q_values)

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error for Q-learning

        return model

    def update_target_model(self):
        """
        Copies the weights from the main model to the target model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state: Tuple (space_array, item_features_array).
            action: Integer, the action taken.
            reward: Float, the reward received.
            next_state: Tuple (space_array, item_features_array).
            done: Boolean, whether the episode terminated.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state: Tuple (space_array, item_features_array).

        Returns:
            Integer, the action index.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action
        
        # Exploit: predict Q-values and choose the best action
        space_array, item_features_array = state
        
        # Model expects batch input: add batch dimension
        space_input_batch = np.expand_dims(space_array, axis=0)
        item_input_batch = np.expand_dims(item_features_array, axis=0)
        
        model_inputs = [space_input_batch, item_input_batch]
        # For TF2, can use model(model_inputs) directly if eager is on or inside tf.function
        q_values = self.model.predict(model_inputs, verbose=0)
        
        return np.argmax(q_values[0]) # Return action with highest Q-value

    def replay(self):
        """
        Trains the main Q-network using a mini-batch of experiences from memory.
        """
        if len(self.memory) < self.batch_size:
            return # Not enough samples to learn

        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare batches for states, next_states, and targets
        batch_current_space_inputs = []
        batch_current_item_inputs = []
        
        batch_next_space_inputs = []
        batch_next_item_inputs = []
        
        targets_for_batch_fit = [] # This will store the Q-values to be fitted

        # Collect current states and next states for batch prediction
        for state, action, reward, next_state, done in minibatch:
            current_space_array, current_item_features_array = state
            batch_current_space_inputs.append(current_space_array)
            batch_current_item_inputs.append(current_item_features_array)

            next_space_array, next_item_features_array = next_state
            batch_next_space_inputs.append(next_space_array)
            batch_next_item_inputs.append(next_item_features_array)

        # Convert lists to NumPy arrays for batch prediction
        np_batch_current_space = np.array(batch_current_space_inputs)
        np_batch_current_item = np.array(batch_current_item_inputs)
        np_batch_next_space = np.array(batch_next_space_inputs)
        np_batch_next_item = np.array(batch_next_item_inputs)

        # Predict Q-values for all current states and next states in the batch
        current_q_values_batch = self.model.predict([np_batch_current_space, np_batch_current_item], verbose=0)
        next_q_values_batch = self.target_model.predict([np_batch_next_space, np_batch_next_item], verbose=0)


        # Calculate target Q-values for each experience in the minibatch
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_q_values_batch[i]) # Max Q-value for next state from target_model
            
            # Get the Q-values for the current state from the batch prediction
            current_q_for_sample = current_q_values_batch[i].copy() # Important to copy
            current_q_for_sample[action] = target # Update the Q-value for the action taken
            targets_for_batch_fit.append(current_q_for_sample)

        # Prepare inputs for model.fit
        fit_inputs = {
            'space_input': np_batch_current_space,
            'item_input': np_batch_current_item
        }
        fit_targets = np.array(targets_for_batch_fit)

        # Train the model on the entire batch
        self.model.fit(fit_inputs, fit_targets, epochs=1, verbose=0, batch_size=self.batch_size) # Use self.batch_size here

        # Epsilon decay (typically called after each episode or a set number of steps in the training loop)
        # For now, placing it here means it decays after each replay call if replay happens frequently.
        # It's often better to call this from the main training loop.
        # self.decay_epsilon() # Moved to be an explicit call from training loop

    def decay_epsilon(self):
        """ Call this method in the training loop to decay epsilon """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Loads model weights from a file.
        """
        try:
            self.model.load_weights(name)
            self.update_target_model() # Also update target model to match loaded weights
            print(f"Model weights loaded from {name}")
        except Exception as e:
            print(f"Error loading weights from {name}: {e}")


    def save(self, name):
        """
        Saves model weights to a file.
        """
        try:
            self.model.save_weights(name)
            print(f"Model weights saved to {name}")
        except Exception as e:
            print(f"Error saving weights to {name}: {e}")


if __name__ == '__main__':
    # Example Usage (illustrative, requires an environment to fully run)
    
    # Define some dummy shapes and action size for testing the agent structure
    # Assume scaled space (e.g., by factor 10 from 1200x1000x1000)
    dummy_space_shape = (120, 100, 100) 
    dummy_item_features_shape = (5,) # L, W, H, can_rotate, stackable
    dummy_state_shape = (dummy_space_shape, dummy_item_features_shape)
    dummy_action_size = 2400 # Example: 12*10*10 (positions) * 2 (orientations)

    print(f"Initializing DQNAgent with state_shape={dummy_state_shape}, action_size={dummy_action_size}")
    agent = DQNAgent(state_shape=dummy_state_shape, action_size=dummy_action_size)
    print("DQNAgent initialized.")
    agent.model.summary()

    # Test remember and replay (requires dummy data)
    print("\nTesting remember() and replay():")
    if len(agent.memory) < agent.batch_size:
        print(f"Memory size ({len(agent.memory)}) is less than batch size ({agent.batch_size}). Populating with dummy data...")
        for i in range(agent.batch_size * 2): # Add more than batch_size samples
            dummy_space = np.random.rand(*dummy_space_shape).astype(np.float32)
            dummy_item_features = np.random.rand(*dummy_item_features_shape).astype(np.float32)
            dummy_state = (dummy_space, dummy_item_features)
            
            dummy_action = random.randrange(dummy_action_size)
            dummy_reward = random.random()
            
            dummy_next_space = np.random.rand(*dummy_space_shape).astype(np.float32)
            dummy_next_item_features = np.random.rand(*dummy_item_features_shape).astype(np.float32)
            dummy_next_state = (dummy_next_space, dummy_next_item_features)
            
            dummy_done = random.choice([True, False])
            
            agent.remember(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
        print(f"Memory populated. Size: {len(agent.memory)}")

    if len(agent.memory) >= agent.batch_size:
        print("Calling agent.replay()...")
        agent.replay()
        print("agent.replay() called.")
    else:
        print("Skipping replay test as memory is still too small.")

    # Test act
    print("\nTesting act():")
    dummy_space_act = np.random.rand(*dummy_space_shape).astype(np.float32)
    dummy_item_act = np.random.rand(*dummy_item_features_shape).astype(np.float32)
    dummy_state_act = (dummy_space_act, dummy_item_act)
    action = agent.act(dummy_state_act)
    print(f"agent.act() returned action: {action} (Epsilon: {agent.epsilon:.3f})")
    
    # Test epsilon decay
    agent.decay_epsilon()
    print(f"Epsilon after decay_epsilon(): {agent.epsilon:.3f}")

    # Test save and load
    print("\nTesting save() and load():")
    model_weights_file = "dqn_agent_test_weights.h5"
    agent.save(model_weights_file)
    
    # Create a new agent and load weights
    agent_loaded = DQNAgent(state_shape=dummy_state_shape, action_size=dummy_action_size)
    agent_loaded.load(model_weights_file)
    # Here you would typically compare weights or behavior to confirm loading worked.
    # For simplicity, just confirming methods run.
    
    # Example: check if weights are indeed loaded by comparing a few.
    # This is a bit involved, but for a real test:
    # original_weights = agent.model.get_weights()[0][0] # Get some specific weight matrix
    # loaded_weights = agent_loaded.model.get_weights()[0][0]
    # if np.array_equal(original_weights, loaded_weights):
    #     print("Weights appear to be loaded correctly (sample check).")
    # else:
    #     print("Weight loading check failed (sample check).")


    print("\nDQNAgent class implementation draft complete.")
    print("Key features:")
    print("- Dual input (space grid, item features) Keras model.")
    print("- Target network for stable Q-learning updates.")
    print("- Replay buffer (`deque`) for experience storage.")
    print("- `act` method for epsilon-greedy action selection.")
    print("- `replay` method with batch processing for training efficiency.")
    print("- `save`/`load` methods for model weights.")
    print("- Epsilon decay managed by `decay_epsilon()` to be called from training loop.")

```
