# 3D Pallet Stacking Optimization using Deep Reinforcement Learning

## Description
This project implements a Deep Q-Network (DQN) agent to learn how to stack a given list of 3D items (cargo) onto a pallet with fixed dimensions. The primary goal is to maximize the number of items placed and the overall space utilization, while adhering to placement constraints.

## Problem Definition
The task is to efficiently pack a set of rectangular items into a larger rectangular container.
*   **Container (Pallet):** A single container with fixed dimensions (e.g., 1200mm Length x 1000mm Width x 1000mm Height).
*   **Cargo:** A predefined list of items, each with specific dimensions (length, width, height). Items may also have properties like rotatability (allowing a 90-degree rotation around the Z-axis, swapping length and width).
*   **Objective:** Place items onto the pallet such that:
    *   Items are entirely within the pallet boundaries.
    *   Items do not overlap with each other.
    *   The number of items placed and the volumetric space utilization are maximized.
*   **Simplifications (Current Version):**
    *   Item weight and complex stability physics (e.g., center of gravity, crushability) are not considered.
    *   Basic stacking is allowed (items can be placed on top of others if the base is supported).
    *   Load order is implicitly tied to the order in the `CARGO_DATA` list.

## Project Structure
The project is organized into the following key files:
```
.
├── main.py             # Main script for training or evaluating the agent
├── environment.py      # Defines the PalletEnvironment (state, actions, rewards)
├── agent.py            # Defines the DQNAgent (neural network, learning algorithm)
├── utils.py            # Utility functions (visualization, metric calculation)
├── README.md           # This file
└── models/             # Directory where trained model weights (e.g., .weights.h5) are saved
```

*   `main.py`: Entry point for running the project. Handles command-line arguments for training or evaluation modes.
*   `environment.py`: Contains the `PalletEnvironment` class, which simulates the pallet, item placement, state representation, and reward calculation.
*   `agent.py`: Implements the `DQNAgent` class, including the Q-network architecture (built with TensorFlow/Keras), replay memory, and the DQN learning algorithm (experience replay, target network updates).
*   `utils.py`: Provides helper functions, currently including `visualize_pallet_matplotlib` for 3D plotting of the pallet state and `calculate_space_utilization`.
*   `models/`: This directory is intended for storing saved model weights. The scripts will save models (e.g., `dqn_pallet_model.weights.h5`) here by default or as specified by the user. (Note: This directory might need to be created manually if not present).

## Setup Instructions

### Prerequisites
*   Python 3.7+ (or a compatible version)
*   pip (Python package installer)

### Dependencies
The project relies on the following Python libraries:
*   NumPy: For numerical operations, especially array manipulation for the pallet grid.
*   TensorFlow (with Keras): For building and training the Deep Q-Network.
*   Matplotlib: For visualizing the pallet and item placements.

### Installation
1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create the `models/` directory (if it doesn't exist):**
    ```bash
    mkdir models
    ```

## Usage
The primary interaction with the project is through `main.py`.

### Training
To train the DQN agent:
```bash
python main.py train
```
This will start the training process using default parameters defined in `main.py`.

**Optional arguments for training:**
*   `--episodes <number>`: Specify the number of episodes to train the agent.
    ```bash
    python main.py train --episodes 500
    ```
    (Default is 100, as set in `main.py`).
*   `--model_path <path>`: Specify the file path to save the trained model weights.
    ```bash
    python main.py train --model_path models/my_custom_model.weights.h5
    ```
    (Default is `models/dqn_pallet_model.weights.h5`).

Training can take a significant amount of time depending on the number of episodes and the complexity of the environment. The script will periodically print progress and save the model weights.

### Evaluation
To evaluate a pre-trained agent:
```bash
python main.py evaluate
```
This will load the default model (`models/dqn_pallet_model.weights.h5`) and run evaluation episodes.

**Optional arguments for evaluation:**
*   `--model_path <path>`: Specify the path to load the pre-trained model weights.
    ```bash
    python main.py evaluate --model_path models/my_custom_model.weights.h5
    ```
*   `--eval_episodes <number>`: Specify the number of evaluation episodes to run.
    ```bash
    python main.py evaluate --eval_episodes 10
    ```
    (Default is 5, as set in `main.py`).

During evaluation, the agent will attempt to stack items using its learned policy. For each episode, metrics like items placed and space utilization will be printed, and a 3D visualization of the final pallet configuration will be displayed.

## Configuration
Several key parameters of the environment and agent can be configured directly within `main.py`:
*   `CONTAINER_DIMS`: Dimensions of the pallet (Length, Width, Height).
*   `CARGO_DATA`: A list of dictionaries, where each dictionary defines an item's properties (id, dimensions, rotatability, etc.).
*   `DISCRETIZATION_STEP`: The granularity (in mm) for discretizing the placement locations (x, y, z) on the pallet. This affects the size of the action space.
*   Agent hyperparameters (e.g., `LEARNING_RATE`, `GAMMA`, `EPSILON_DECAY`, `REPLAY_BUFFER_SIZE`, `BATCH_SIZE`) are also defined in `main.py`.

Modifying these parameters allows for experimentation with different scenarios and agent behaviors.

## Key Algorithm Details
*   **Agent:** Deep Q-Network (DQN). The agent learns a Q-value for each state-action pair, representing the expected future reward.
*   **State Representation:** The state provided to the agent is a tuple consisting of:
    1.  A 3D NumPy array representing the pallet's occupancy grid. Each cell indicates whether it's empty or occupied by a part of an item (and by which item).
    2.  A 1D NumPy array containing features of the current item to be placed (e.g., its dimensions, rotatability).
*   **Action Space:** The agent's actions are discrete. An action corresponds to:
    *   A discretized (x, y, z) coordinate on the pallet for placing the bottom-front-left corner of the item.
    *   An orientation for the item (0 for original, 1 for 90-degree rotation around the Z-axis if the item is rotatable).
*   **Reward Function:** The agent receives a reward based on:
    *   A positive reward proportional to the volume of the item if it's successfully placed.
    *   A negative penalty for attempting an invalid placement (e.g., out of bounds, collision).

## Limitations & Future Work
This project provides a foundational implementation. There are several areas for potential improvement and extension:

*   **Reward Shaping:** The current reward function is relatively simple. More complex rewards could guide learning more effectively (e.g., penalties for instability, bonuses for filling gaps).
*   **Action Space:** Discretization can lead to suboptimal placements if the optimal spot is between discrete points. Exploring continuous action spaces (e.g., using DDPG, SAC) or adaptive discretization could be beneficial.
*   **Physical Constraints:**
    *   **Weight Distribution & Stability:** Currently not considered. Integrating physics for stability (e.g., center of mass, stacking limits based on weight) would make solutions more realistic.
    *   **Load Order Constraints:** The current system processes items in a fixed order. Real-world scenarios might have constraints on which items can be placed before others.
    *   **Fragility/Stacking Rules:** Specific rules like "do not stack heavy items on light/fragile items" are not implemented.
*   **Advanced Agent Architectures:** Exploring improvements to DQN (e.g., Double DQN, Dueling DQN, Prioritized Experience Replay) or other RL algorithms.
*   **Dynamic Item Feeding:** Handling scenarios where items arrive dynamically rather than being known upfront.
*   **Performance Optimization:** For very large state/action spaces, further optimization of the environment and agent might be needed.

Contributions and suggestions for improvements are welcome!
```
