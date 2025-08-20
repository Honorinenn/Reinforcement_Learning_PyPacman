# Reinforcement Learning for Pacman

This project implements and compares two reinforcement learning approaches for playing Pacman:

1. **Q-Learning** - A tabular reinforcement learning method as a baseline
2. **Deep Q-Network (DQN)** - A deep learning approach for handling high-dimensional state spaces

## Features

### Q-Learning Agent
- **Tabular Q-Learning**: Uses a Q-table to store state-action values
- **State Discretization**: Converts continuous game state to discrete representation
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Legal Action Masking**: Only considers valid moves to reduce invalid actions
- **Persistent Q-Table**: Save and load Q-tables for continued training

### DQN Agent
- **Multiple Architectures**: 
  - Standard DQN with fully connected layers
  - Dueling DQN with separate value and advantage streams
  - Convolutional DQN for spatial state representations
- **Advanced Features**:
  - Double DQN to reduce overestimation bias
  - Experience replay buffer for stable learning
  - Target network for stable Q-learning
  - Gradient clipping and learning rate scheduling
  - Legal action masking
- **Enhanced State Representation**: Includes egocentric local view of the game board

## File Structure

```
├── q_learning_agent.py      # Q-Learning implementation
├── q_learning_train.py      # Q-Learning training script
├── dqn_agent.py            # DQN agent implementation
├── dqn_model.py            # Neural network architectures
├── dqn_train.py            # Original DQN training script
├── train_agents.py         # Unified training script for both agents
├── game_manager.py         # Game environment and state management
├── logger.py               # Training metrics and logging
├── entities.py             # Game entities (Pacman, ghosts, pellets)
├── game_layouts.py         # Game board layouts
├── sprite_configs.py       # Sprite configurations
└── models/                 # Saved models directory
    ├── q_learning.pkl      # Q-Learning Q-table
    └── dqn_*.pth          # DQN model checkpoints
```

## Installation

1. Install required dependencies:
```bash
pip install torch pygame numpy matplotlib pandas
```

2. Ensure you have the game assets in the `assets/` directory.

## Usage

### Quick Start - Train Both Agents

```bash
# Train both Q-Learning and DQN agents for 1000 episodes each
python train_agents.py --agent both --episodes 1000

# Train with display enabled (slower but visual)
python train_agents.py --agent both --episodes 500 --display

# Train with custom tag for organization
python train_agents.py --agent both --episodes 1000 --tag experiment1
```

### Train Individual Agents

#### Q-Learning
```bash
# Basic Q-Learning training
python train_agents.py --agent q_learning --episodes 2000

# With custom hyperparameters
python train_agents.py --agent q_learning \
    --episodes 2000 \
    --q_learning_rate 0.1 \
    --gamma 0.95 \
    --epsilon_decay 0.995
```

#### DQN
```bash
# Standard DQN
python train_agents.py --agent dqn --episodes 1000 --dqn_architecture standard

# Dueling DQN
python train_agents.py --agent dqn --episodes 1000 --dqn_architecture dueling

# Convolutional DQN
python train_agents.py --agent dqn --episodes 1000 --dqn_architecture conv

# With custom hyperparameters
python train_agents.py --agent dqn \
    --episodes 1000 \
    --dqn_architecture dueling \
    --memory_size 20000 \
    --batch_size 128 \
    --dqn_learning_rate 0.0005
```

### Advanced Options

```bash
# Load existing models and continue training
python train_agents.py --agent both --episodes 500 --load_model

# Custom hyperparameters
python train_agents.py --agent both \
    --episodes 1000 \
    --gamma 0.99 \
    --epsilon 0.9 \
    --epsilon_min 0.05 \
    --epsilon_decay 0.998 \
    --save_frequency 50 \
    --print_frequency 50

# DQN-specific options
python train_agents.py --agent dqn \
    --dqn_architecture dueling \
    --memory_size 15000 \
    --batch_size 64 \
    --target_update_freq 20 \
    --use_double_dqn
```

### Legacy Training Scripts

You can also use the individual training scripts:

```bash
# Q-Learning only
python q_learning_train.py

# DQN only
python dqn_train.py
```

## State Representations

### Q-Learning State
- Pacman position (x, y)
- Ghost positions (up to 4 ghosts)
- Pellet count
- **Total features**: 11 (discretized for Q-table)

### DQN State
- Pacman position (x, y)
- Ghost positions
- Pellet count
- **Egocentric local view**: 7x7 grid around Pacman with 3 channels:
  - Channel 1: Pellets
  - Channel 2: Walls
  - Channel 3: Ghosts
- **Total features**: 158 (11 scalar + 147 spatial)

## Network Architectures

### Standard DQN
- Fully connected layers: 158 → 256 → 256 → 128 → 4
- Layer normalization and dropout for regularization
- ReLU activations

### Dueling DQN
- Shared feature extraction: 158 → 256 → 256
- Value stream: 256 → 128 → 1
- Advantage stream: 256 → 128 → 4
- Combined: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

### Convolutional DQN
- Convolutional layers for spatial features: 3@7x7 → 32@7x7 → 64@7x7 → 64@7x7
- Fully connected layers: (64×7×7 + 11) → 512 → 256 → 4
- Processes both spatial and scalar features

## Training Features

### Reward Structure
- **Step penalty**: -0.02 (encourages efficiency)
- **Pellet reward**: +1.0 per pellet
- **Win bonus**: +300.0 (complete level)
- **Death penalty**: -200.0 (collision with ghost)
- **Exploration bonus**: +0.05 (visiting new cells)
- **Distance shaping**: Small reward for getting closer to pellets

### Advanced DQN Features
- **Double DQN**: Reduces overestimation bias
- **Experience Replay**: Breaks correlation between consecutive samples
- **Target Network**: Provides stable learning targets
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rate decay
- **Legal Action Masking**: Prevents invalid moves

## Monitoring and Logging

Training metrics are automatically logged to the `runs/` directory:

- **CSV files**: Episode-by-episode metrics
- **JSONL files**: Structured logging data
- **Model checkpoints**: Best and latest models
- **Configuration files**: Hyperparameters and settings

### Key Metrics
- Episode reward
- Pellet clear rate (percentage of pellets collected)
- Epsilon (exploration rate)
- Loss (DQN only)
- Q-table size (Q-Learning only)
- Moving average reward

## Performance Tips

### For Q-Learning
- Start with higher learning rate (0.1-0.3)
- Use longer training (2000+ episodes)
- Monitor Q-table growth
- Adjust epsilon decay for exploration

### For DQN
- Use larger replay buffer for better sample diversity
- Tune batch size based on available memory
- Monitor loss convergence
- Try different architectures for your specific problem

### General
- Use `--display` for debugging but disable for fast training
- Save models frequently with `--save_frequency`
- Use tags to organize different experiments
- Monitor moving average reward for training progress

## Example Training Session

```bash
# Train both agents with good hyperparameters
python train_agents.py \
    --agent both \
    --episodes 1500 \
    --gamma 0.95 \
    --epsilon_decay 0.995 \
    --q_learning_rate 0.15 \
    --dqn_learning_rate 0.001 \
    --dqn_architecture dueling \
    --memory_size 15000 \
    --batch_size 64 \
    --save_frequency 100 \
    --print_frequency 50 \
    --tag comparison_run
```

This will create:
- `models/q_learning_comparison_run.pkl`
- `models/dqn_dueling_comparison_run.pth`
- Training logs in `runs/RL_PacMan/`

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or memory size
2. **Slow training**: Disable display, increase print frequency
3. **Poor performance**: Check reward structure, adjust hyperparameters
4. **Q-table too large**: Improve state discretization
5. **DQN not learning**: Check learning rate, replay buffer size

### Performance Optimization

- Use GPU if available (automatic detection)
- Increase batch size for better GPU utilization
- Use headless mode for faster training
- Adjust FPS in game manager for training speed

## Results Analysis

After training, you can analyze results by:

1. Checking the `runs/` directory for training curves
2. Loading saved models for evaluation
3. Comparing Q-Learning vs DQN performance
4. Analyzing pellet clear rates and episode lengths

The logging system provides comprehensive metrics for detailed analysis of agent performance and learning progress.
