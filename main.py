import argparse
import numpy as np
import random
import matplotlib.pyplot as plt # For plt.show() in evaluate

from environment import PalletEnvironment
from agent import DQNAgent
from utils import visualize_pallet_matplotlib, calculate_space_utilization

# --- Constants ---
# (Consider moving to a config.py in a future refactor)
CARGO_DATA = [
    {"id": "苏打天然水饮品", "length": 301.0, "width": 182.0, "height": 204.0, "weight": 6.15, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 301*182*204},
    {"id": "珠江啤酒", "length": 410.0, "width": 273.0, "height": 125.0, "weight": 8.5, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 410*273*125},
    {"id": "亚洲沙示汽水", "length": 329.0, "width": 198.0, "height": 210.0, "weight": 9.8, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 329*198*210},
    {"id": "苏打水饮品", "length": 296.0, "width": 177.0, "height": 225.0, "weight": 8.3, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 296*177*225},
    {"id": "娃哈哈蜂蜜水果绿茶果汁茶饮品", "length": 382.0, "width": 230.0, "height": 233.0, "weight": 13.1, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 382*230*233},
    {"id": "咸柠七咸柠檬气泡水", "length": 355.0, "width": 205.0, "height": 217.0, "weight": 10.0, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 355*205*217},
    {"id": "怡宝", "length": 371.0, "width": 248.0, "height": 185.0, "weight": 9.0, "can_rotate": True, "stackable": True, "stack_face": "width&length", "volume": 371*248*185}
]
# Pre-calculate volume for each item (in mm^3)
for item in CARGO_DATA:
    item['volume'] = item['length'] * item['width'] * item['height']

CONTAINER_DIMS = (1200, 1000, 1000)  # Length, Width, Height in mm
DEFAULT_TRAIN_EPISODES = 100
DEFAULT_EVAL_EPISODES = 5
MAX_STEPS_PER_EPISODE = len(CARGO_DATA) + 10 # Max attempts: number of items + some extra
DISCRETIZATION_STEP = 100  # In mm, for x, y, z positions

# Agent Hyperparameters (can be part of train_agent or global if fixed)
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_INITIAL = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 5
SAVE_MODEL_FREQ = 20 # Only relevant for train_agent internal saving

DEFAULT_MODEL_PATH = "dqn_pallet_model.weights.h5"

# --- Action Decoding Function ---
def decode_action(action_id, num_x, num_y, num_z, num_orientations, discretization_step):
    """
    Decodes an action ID into physical coordinates and orientation.
    """
    orientation = action_id % num_orientations
    
    remainder = action_id // num_orientations
    z_idx = remainder % num_z
    pos_z = z_idx * discretization_step
    
    remainder = remainder // num_z
    y_idx = remainder % num_y
    pos_y = y_idx * discretization_step
    
    x_idx = remainder // num_y 
    pos_x = x_idx * discretization_step
    
    return int(pos_x), int(pos_y), int(pos_z), int(orientation)

# --- Training Function ---
def train_agent(episodes, max_steps, cargo_data, container_dims, 
                discretization_step, model_save_path):
    print("Starting training process...")
    print(f"  Episodes: {episodes}, Max steps/episode: {max_steps}")
    print(f"  Cargo items: {len(cargo_data)}, Container: {container_dims}")
    print(f"  Discretization: {discretization_step}mm, Model save path: {model_save_path}")

    env = PalletEnvironment(container_dims, cargo_data)
    initial_state = env.reset()
    space_shape = initial_state[0].shape
    item_features_shape = initial_state[1].shape
    state_shape = (space_shape, item_features_shape)

    num_x_positions = container_dims[0] // discretization_step
    num_y_positions = container_dims[1] // discretization_step
    num_z_positions = container_dims[2] // discretization_step
    num_orientations = 2
    action_size = num_x_positions * num_y_positions * num_z_positions * num_orientations

    print(f"  Environment space grid shape: {space_shape}, Item features shape: {item_features_shape}")
    print(f"  Calculated action space size: {action_size}")

    agent = DQNAgent(state_shape=state_shape,
                     action_size=action_size,
                     learning_rate=LEARNING_RATE,
                     gamma=GAMMA,
                     epsilon=EPSILON_INITIAL,
                     epsilon_decay=EPSILON_DECAY,
                     epsilon_min=EPSILON_MIN,
                     replay_buffer_size=REPLAY_BUFFER_SIZE,
                     batch_size=BATCH_SIZE)

    print("DQNAgent initialized for training.")

    for episode in range(episodes):
        current_state = env.reset()
        total_reward_episode = 0
        items_placed_episode = 0
        
        for step_num in range(max_steps):
            action_id = agent.act(current_state)
            
            pos_x, pos_y, pos_z, orientation = decode_action(
                action_id, num_x_positions, num_y_positions, num_z_positions, 
                num_orientations, discretization_step
            )
            env_action = (pos_x, pos_y, pos_z, orientation)
            
            next_state, reward, done, info = env.step(env_action)
            agent.remember(current_state, action_id, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            current_state = next_state
            total_reward_episode += reward
            
            if info.get("placed", False):
                items_placed_episode += 1
            
            if done:
                break
                
        agent.decay_epsilon()

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        print(f"Episode: {episode+1}/{episodes} | Reward: {total_reward_episode:.2f} | "
              f"Items Placed: {items_placed_episode}/{len(cargo_data)} | Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % SAVE_MODEL_FREQ == 0 or (episode + 1) == episodes : # Save at interval or last episode
             intermediate_save_path = model_save_path.replace(".weights.h5", f"_ep{episode+1}.weights.h5")
             agent.save(intermediate_save_path)


    agent.save(model_save_path) # Save final model
    print(f"Training complete. Final model saved to {model_save_path}")

# --- Evaluation Function ---
def evaluate_agent(cargo_data, container_dims, discretization_step, model_load_path, num_episodes=5):
    print(f"Starting evaluation with model: {model_load_path}")
    print(f"  Evaluation episodes: {num_episodes}")

    env = PalletEnvironment(container_dims, cargo_data) # Use a fresh env
    initial_state = env.reset() # To get shapes
    space_shape = initial_state[0].shape
    item_features_shape = initial_state[1].shape
    state_shape = (space_shape, item_features_shape)

    num_x_positions = container_dims[0] // discretization_step
    num_y_positions = container_dims[1] // discretization_step
    num_z_positions = container_dims[2] // discretization_step
    num_orientations = 2
    action_size = num_x_positions * num_y_positions * num_z_positions * num_orientations
    
    print(f"  Environment space grid shape: {space_shape}, Item features shape: {item_features_shape}")
    print(f"  Calculated action space size: {action_size}")

    agent = DQNAgent(state_shape=state_shape, action_size=action_size)
    
    try:
        agent.load(model_load_path)
        print(f"Model weights loaded successfully from {model_load_path}.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_load_path}. Cannot evaluate.")
        return
    except Exception as e:
        print(f"Error loading model weights from {model_load_path}: {e}. Cannot evaluate.")
        return

    agent.epsilon = 0.01 # Use very low epsilon for predominantly greedy actions

    total_container_volume = container_dims[0] * container_dims[1] * container_dims[2]

    for episode in range(num_episodes):
        current_state = env.reset()
        placed_item_volumes_episode = []
        total_items_placed_episode = 0
        
        print(f"\n--- Evaluation Episode {episode + 1}/{num_episodes} ---")

        for step_num in range(MAX_STEPS_PER_EPISODE): # Use MAX_STEPS_PER_EPISODE from constants
            action_id = agent.act(current_state) # Agent acts based on loaded model (mostly greedy)
            
            pos_x, pos_y, pos_z, orientation = decode_action(
                action_id, num_x_positions, num_y_positions, num_z_positions, 
                num_orientations, discretization_step
            )
            env_action = (pos_x, pos_y, pos_z, orientation)
            
            next_state, reward, done, info = env.step(env_action)

            if info.get("placed", False) and reward > 0: # Successfully placed an item
                total_items_placed_episode += 1
                # The item just placed was cargo_items[env.current_item_index - 1]
                # (since current_item_index was incremented after placement)
                if env.current_item_index > 0 and (env.current_item_index -1) < len(env.cargo_items):
                    placed_item_obj = env.cargo_items[env.current_item_index - 1]
                    # Use pre-calculated volume from CARGO_DATA setup
                    original_volume = placed_item_obj.get('volume', 0)
                    if original_volume > 0:
                         placed_item_volumes_episode.append(original_volume)
                    print(f"  Step {step_num+1}: Placed item '{placed_item_obj['id']}' (Vol: {original_volume/1e9:.3f} m^3).")
                else: # Should not happen if logic is correct
                    print(f"  Step {step_num+1}: Item placed, but couldn't retrieve details. Reward: {reward}")

            current_state = next_state
            if done:
                print(f"  Episode finished after {step_num + 1} steps.")
                break
        
        utilization = calculate_space_utilization(placed_item_volumes_episode, total_container_volume)
        print(f"Evaluation Episode {episode + 1} Summary:")
        print(f"  Items Placed: {total_items_placed_episode}/{len(cargo_data)}")
        print(f"  Total Volume of Placed Items: {sum(placed_item_volumes_episode)/1e9:.3f} m^3")
        print(f"  Container Volume: {total_container_volume/1e9:.3f} m^3")
        print(f"  Space Utilization: {utilization * 100:.2f}%")

        # Visualize the final state of the pallet for this episode
        print(f"  Visualizing pallet for episode {episode + 1}...")
        # Pass env.space.shape as container_dims for visualization, as it reflects the grid dimensions
        visualize_pallet_matplotlib(env.space, env.space.shape) 
        plt.show() # Show plot for each evaluation episode

    print("\nEvaluation complete.")


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent for pallet stacking.")
    parser.add_argument('mode', choices=['train', 'evaluate'], help="Mode of operation: 'train' or 'evaluate'.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to save/load model weights (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument('--episodes', type=int, default=DEFAULT_TRAIN_EPISODES,
                        help=f"Number of episodes for training (default: {DEFAULT_TRAIN_EPISODES}).")
    parser.add_argument('--eval_episodes', type=int, default=DEFAULT_EVAL_EPISODES,
                        help=f"Number of episodes for evaluation (default: {DEFAULT_EVAL_EPISODES}).")
    
    args = parser.parse_args()

    # Use constants defined at the top for cargo, container, and discretization
    # MAX_STEPS_PER_EPISODE is also a global constant here.

    if args.mode == 'train':
        train_agent(episodes=args.episodes, 
                    max_steps=MAX_STEPS_PER_EPISODE, 
                    cargo_data=CARGO_DATA, 
                    container_dims=CONTAINER_DIMS, 
                    discretization_step=DISCRETIZATION_STEP, 
                    model_save_path=args.model_path)
    elif args.mode == 'evaluate':
        evaluate_agent(cargo_data=CARGO_DATA, 
                       container_dims=CONTAINER_DIMS, 
                       discretization_step=DISCRETIZATION_STEP, 
                       model_load_path=args.model_path, 
                       num_episodes=args.eval_episodes)
    else:
        print("Invalid mode selected. Choose 'train' or 'evaluate'.")

```
