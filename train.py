import numpy as np
import random
from environment import PalletEnvironment
from agent import DQNAgent

# 1. Constants and Hyperparameters
CONTAINER_DIMS = (1200, 1000, 1000)  # Length, Width, Height in mm
CARGO_DATA = [
    {"id": "苏打天然水饮品", "length": 301.0, "width": 182.0, "height": 204.0, "weight": 6.15, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
    {"id": "珠江啤酒", "length": 410.0, "width": 273.0, "height": 125.0, "weight": 8.5, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
    {"id": "亚洲沙示汽水", "length": 329.0, "width": 198.0, "height": 210.0, "weight": 9.8, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
    {"id": "苏打水饮品", "length": 296.0, "width": 177.0, "height": 225.0, "weight": 8.3, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
    {"id": "娃哈哈蜂蜜水果绿茶果汁茶饮品", "length": 382.0, "width": 230.0, "height": 233.0, "weight": 13.1, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
    {"id": "咸柠七咸柠檬气泡水", "length": 355.0, "width": 205.0, "height": 217.0, "weight": 10.0, "can_rotate": True, "stackable": True, "stack_face": "width&length"},
    {"id": "怡宝", "length": 371.0, "width": 248.0, "height": 185.0, "weight": 9.0, "can_rotate": True, "stackable": True, "stack_face": "width&length"}
]

EPISODES = 100  # Smaller number for initial testing
MAX_STEPS_PER_EPISODE = len(CARGO_DATA) + 5 # Max attempts: number of items + a few extra
DISCRETIZATION_STEP = 100  # In mm, for x, y, z positions

# Agent Hyperparameters (can use agent defaults or override here)
LEARNING_RATE = 0.001
GAMMA = 0.95 # Discount factor
EPSILON = 1.0
EPSILON_DECAY = 0.995 # Decay rate for exploration probability
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 5000 # Reduced for faster initial testing
BATCH_SIZE = 32 # Size of minibatch from replay buffer

TARGET_UPDATE_FREQ = 5  # Update target network every X episodes
SAVE_MODEL_FREQ = 20    # Save model every X episodes


# 2. Action Decoding Function
def decode_action(action_id, num_x, num_y, num_z, num_orientations, discretization_step):
    """
    Decodes an action ID into physical coordinates and orientation.

    Args:
        action_id: Integer action ID from the agent.
        num_x: Number of discrete positions along X-axis.
        num_y: Number of discrete positions along Y-axis.
        num_z: Number of discrete positions along Z-axis.
        num_orientations: Number of possible orientations (e.g., 2).
        discretization_step: Physical step size (e.g., 100 mm).

    Returns:
        A tuple (pos_x, pos_y, pos_z, orientation_idx)
        where positions are unscaled physical coordinates.
    """
    orientation = action_id % num_orientations
    
    remainder = action_id // num_orientations
    z_idx = remainder % num_z
    pos_z = z_idx * discretization_step
    
    remainder = remainder // num_z
    y_idx = remainder % num_y
    pos_y = y_idx * discretization_step
    
    # x_idx = remainder // num_y # This would be correct if num_x is the last one to be 'divided out'
    # Corrected logic: x_idx is the final remainder
    x_idx = remainder // num_y 
    pos_x = x_idx * discretization_step

    # Boundary check (important if action_size is not perfectly aligned with num_x*num_y*num_z*num_orientations)
    # However, action_size is defined as exactly this product, so x_idx should be < num_x.
    # For robustness, one might add: if x_idx >= num_x: handle error or cap
    
    return int(pos_x), int(pos_y), int(pos_z), int(orientation)


# 3. Initialize Environment and Agent
print("Initializing environment...")
env = PalletEnvironment(CONTAINER_DIMS, CARGO_DATA)

print("Resetting environment to get initial state shapes...")
initial_state = env.reset()
space_shape = initial_state[0].shape
item_features_shape = initial_state[1].shape
state_shape = (space_shape, item_features_shape)
print(f"  Space shape: {space_shape}")
print(f"  Item features shape: {item_features_shape}")
print(f"  Combined state shape for agent: {state_shape}")
print(f"  Environment scaling factor: {env.scaling_factor} (1 means no scaling)")


# Calculate action space size
num_x_positions = CONTAINER_DIMS[0] // DISCRETIZATION_STEP
num_y_positions = CONTAINER_DIMS[1] // DISCRETIZATION_STEP
num_z_positions = CONTAINER_DIMS[2] // DISCRETIZATION_STEP
num_orientations = 2  # 0 for original, 1 for 90-degree rotation

action_size = num_x_positions * num_y_positions * num_z_positions * num_orientations
print(f"Calculated action space size: {action_size}")
print(f"  Num X positions: {num_x_positions} (Discretization: {DISCRETIZATION_STEP}mm)")
print(f"  Num Y positions: {num_y_positions}")
print(f"  Num Z positions: {num_z_positions}")
print(f"  Num Orientations: {num_orientations}")


print("\nInitializing DQN Agent...")
agent = DQNAgent(state_shape=state_shape,
                 action_size=action_size,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 epsilon_decay=EPSILON_DECAY,
                 epsilon_min=EPSILON_MIN,
                 replay_buffer_size=REPLAY_BUFFER_SIZE,
                 batch_size=BATCH_SIZE)
print("DQNAgent initialized.")
# agent.model.summary() # Optional: print model summary


# 4. Training Loop
print(f"\nStarting training for {EPISODES} episodes...")
for episode in range(EPISODES):
    current_state = env.reset()  # (space_array, item_features_array)
    total_reward_episode = 0
    items_placed_episode = 0
    
    # The environment internally manages self.current_item_index after reset.
    # The number of items in CARGO_DATA is the max useful steps for placement.
    # MAX_STEPS_PER_EPISODE allows for some failed attempts per item.
    for step_num in range(MAX_STEPS_PER_EPISODE):
        action_id = agent.act(current_state)
        
        pos_x, pos_y, pos_z, orientation = decode_action(
            action_id,
            num_x_positions,
            num_y_positions,
            num_z_positions,
            num_orientations,
            DISCRETIZATION_STEP
        )
        
        # Environment's step method expects (pos_x, pos_y, pos_z, orientation)
        # where positions are physical, unscaled coordinates.
        env_action = (pos_x, pos_y, pos_z, orientation)
        
        next_state, reward, done, info = env.step(env_action)
        
        # Agent's remember method expects (state, action_id, reward, next_state, done)
        agent.remember(current_state, action_id, reward, next_state, done)
        
        if len(agent.memory) > agent.batch_size : # Start replay only when enough samples are in memory
            agent.replay()
        
        current_state = next_state
        total_reward_episode += reward
        
        if info.get("placed", False): # Check if item was successfully placed
            items_placed_episode += 1
            # print(f"  Step {step_num+1}: Item '{info['item_id_placed']}' placed. Reward: {reward:.2f}")
        # else:
            # print(f"  Step {step_num+1}: Invalid placement. Reward: {reward:.2f}")


        if done:
            # print(f"  Episode finished after {step_num+1} steps (all items processed or max steps reached).")
            break # Break from inner loop (steps)
            
    agent.decay_epsilon() # Decay epsilon after each episode

    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()
        # print(f"Updated target model at episode {episode+1}")

    print(f"Episode: {episode+1}/{EPISODES} | Total Reward: {total_reward_episode:.2f} | "
          f"Items Placed: {items_placed_episode}/{len(CARGO_DATA)} | Epsilon: {agent.epsilon:.3f}")

    if (episode + 1) % SAVE_MODEL_FREQ == 0:
        save_path = f"dqn_pallet_stacking_episode_{episode+1}.weights.h5"
        agent.save(save_path)
        # print(f"Model saved to {save_path}")

# 5. Post-training
final_save_path = "dqn_pallet_stacking_final.weights.h5"
agent.save(final_save_path)
print(f"\nTraining complete. Final model saved to {final_save_path}")

print("\n--- Example of Action Decoding ---")
test_action_id = random.randint(0, action_size -1) # agent.action_size - 1)
decoded_p_x, decoded_p_y, decoded_p_z, decoded_orient = decode_action(
    test_action_id, num_x_positions, num_y_positions, num_z_positions, num_orientations, DISCRETIZATION_STEP
)
print(f"Test Action ID: {test_action_id}")
print(f"  Decoded to -> X: {decoded_p_x}, Y: {decoded_p_y}, Z: {decoded_p_z}, Orient: {decoded_orient}")

test_action_id_max = action_size -1
decoded_p_x, decoded_p_y, decoded_p_z, decoded_orient = decode_action(
    test_action_id_max, num_x_positions, num_y_positions, num_z_positions, num_orientations, DISCRETIZATION_STEP
)
print(f"Test Max Action ID: {test_action_id_max}") # Should correspond to highest x,y,z indices and highest orientation
print(f"  Decoded to -> X: {decoded_p_x}, Y: {decoded_p_y}, Z: {decoded_p_z}, Orient: {decoded_orient}")
print(f"  (Expected max indices: X_idx={num_x_positions-1}, Y_idx={num_y_positions-1}, Z_idx={num_z_positions-1}, Orient_idx={num_orientations-1})")


print("\nScript finished.")
```
