# File: q_learning_agent.py

import numpy as np
import random
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        """
        Q-Learning agent for Pacman game.

        Args:
            action_size: Number of possible actions (4 for up, down, left, right)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table using defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        # Statistics tracking
        self.total_steps = 0
        self.episodes_trained = 0

    def get_state_key(self, state):
        """
        Convert state array to a hashable key for Q-table lookup.
        For high-dimensional states, we'll use discretization.
        """
        # For the basic Q-learning, we'll use a simplified state representation
        # focusing on key features: pacman position, nearest ghost, nearest pellet

        if (
            len(state) >= 11
        ):  # Basic state with pacman pos + ghost positions + pellet count
            # Extract key features and discretize them
            pacman_x = int(state[0] // 24)  # Discretize to grid cells
            pacman_y = int(state[1] // 24)

            # Ghost positions (first ghost only for simplicity)
            ghost_x = int(state[2] // 24) if len(state) > 2 else 0
            ghost_y = int(state[3] // 24) if len(state) > 3 else 0

            # Pellet count (discretized into ranges)
            pellet_count = int(state[-1])
            pellet_range = min(pellet_count // 10, 20)  # Group into ranges of 10

            # Calculate relative positions
            rel_ghost_x = min(max(ghost_x - pacman_x, -5), 5)  # Clamp to [-5, 5]
            rel_ghost_y = min(max(ghost_y - pacman_y, -5), 5)

            return (pacman_x, pacman_y, rel_ghost_x, rel_ghost_y, pellet_range)
        else:
            # Fallback for unexpected state format
            return tuple(int(x) for x in state[: min(len(state), 5)])

    def get_action(self, state, legal_actions=None):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            legal_actions: List of legal actions (optional)

        Returns:
            Selected action
        """
        if legal_actions is not None and len(legal_actions) == 0:
            legal_actions = None

        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            # Exploration: random action
            if legal_actions:
                return random.choice(legal_actions)
            else:
                return random.randrange(self.action_size)
        else:
            # Exploitation: best known action
            state_key = self.get_state_key(state)
            q_values = self.q_table[state_key]

            if legal_actions is None:
                return np.argmax(q_values)
            else:
                # Choose best action among legal actions
                legal_q_values = [
                    (action, q_values[action]) for action in legal_actions
                ]
                return max(legal_q_values, key=lambda x: x[1])[0]

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Next state's maximum Q-value
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_key])

        # Q-learning update rule
        target_q = reward + self.discount_factor * next_max_q
        self.q_table[state_key][action] = current_q + self.learning_rate * (
            target_q - current_q
        )

        self.total_steps += 1

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.episodes_trained += 1

    def save_q_table(self, filepath):
        """Save Q-table to file."""
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        save_data = {
            "q_table": q_table_dict,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "episodes_trained": self.episodes_trained,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table from file."""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)

            # Restore Q-table as defaultdict
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            self.q_table.update(save_data["q_table"])

            # Restore other attributes
            self.epsilon = save_data.get("epsilon", self.epsilon)
            self.total_steps = save_data.get("total_steps", 0)
            self.episodes_trained = save_data.get("episodes_trained", 0)

            print(f"Q-table loaded from {filepath}")
            print(
                f"Loaded {len(save_data['q_table'])} states, epsilon={self.epsilon:.3f}"
            )
        else:
            print(f"No saved Q-table found at {filepath}")

    def get_q_table_stats(self):
        """Get statistics about the Q-table."""
        return {
            "num_states": len(self.q_table),
            "total_steps": self.total_steps,
            "episodes_trained": self.episodes_trained,
            "current_epsilon": self.epsilon,
            "avg_q_value": (
                np.mean([np.mean(q_vals) for q_vals in self.q_table.values()])
                if self.q_table
                else 0
            ),
        }
