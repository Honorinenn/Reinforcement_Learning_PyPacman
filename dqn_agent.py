
# File: dqn_agent.py

import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from dqn_model import DQN, DuelingDQN, ConvDQN


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        architecture="standard",
        memory_size=10000,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        batch_size=64,
        target_update_freq=10,
        use_double_dqn=True,
        use_prioritized_replay=False,
    ):
        """
        Enhanced DQN Agent with multiple architecture options and advanced features.

        Args:
            state_size: Size of the state space
            action_size: Number of possible actions
            architecture: Network architecture ("standard", "dueling", "conv")
            memory_size: Size of the replay buffer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            use_double_dqn: Whether to use Double DQN
            use_prioritized_replay: Whether to use prioritized experience replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.architecture = architecture
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize memory
        self.memory = deque(maxlen=memory_size)

        # Initialize networks based on architecture
        self.policy_net = self._create_network().to(self.device)
        self.target_net = self._create_network().to(self.device)

        # Initialize optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )

        self.update_target_network()

        # Training statistics
        self.training_steps = 0
        self.losses = []

    def _create_network(self):
        """Create network based on specified architecture."""
        if self.architecture == "dueling":
            return DuelingDQN(self.state_size, self.action_size)
        elif self.architecture == "conv":
            # For conv architecture, assume state includes both grid and scalar features
            # Grid: 3 channels (pellets, walls, ghosts) x 7x7 = 147 features
            # Scalar: 11 features (positions, etc.)
            return ConvDQN(
                grid_channels=3,
                grid_size=7,
                scalar_features=11,
                action_size=self.action_size,
            )
        else:  # standard
            return DQN(self.state_size, self.action_size)

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, legal_actions=None):
        """
        Choose action using epsilon-greedy policy with legal action masking.

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
            return (
                random.choice(legal_actions)
                if legal_actions
                else random.randrange(self.action_size)
            )

        # Get Q-values from policy network
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0)

        if legal_actions is None:
            return int(torch.argmax(q_values).item())
        else:
            # Mask illegal actions
            mask = torch.full_like(q_values, float("-inf"))
            mask[
                torch.tensor(legal_actions, dtype=torch.long, device=q_values.device)
            ] = 0.0
            masked_q = q_values + mask
            return int(torch.argmax(masked_q).item())

    def train(self, batch_size=None):
        """
        Enhanced training method with Double DQN and gradient clipping.

        Args:
            batch_size: Batch size for training (uses default if None)

        Returns:
            Loss value or None if not enough samples
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return None

        # Sample from memory
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to numpy arrays first, then to tensors for better performance
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Calculate target Q-values
        if self.use_double_dqn:
            # Double DQN: use policy network to select actions, target network to evaluate
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = (
                    self.target_net(next_states).gather(1, next_actions).squeeze(1)
                )
        else:
            # Standard DQN
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Huber loss is more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Update training statistics
        self.training_steps += 1
        self.losses.append(loss.item())

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        """Save the trained model."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
                "architecture": self.architecture,
                "hyperparameters": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "memory_size": self.memory_size,
                    "gamma": self.gamma,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "use_double_dqn": self.use_double_dqn,
                },
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.training_steps = checkpoint.get("training_steps", 0)
        print(f"Model loaded from {filepath}")
        print(f"Training steps: {self.training_steps}, Epsilon: {self.epsilon:.3f}")

    def get_stats(self):
        """Get training statistics."""
        return {
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0,
            "architecture": self.architecture,
            "use_double_dqn": self.use_double_dqn,
        }
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# from collections import deque

# class DQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_size)
#         )

#     def forward(self, x):
#         return self.fc(x)

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.99
#         self.epsilon = 1.0
#         self.epsilon_min = 0.1
#         self.epsilon_decay = 0.995
#         self.model = DQN(state_size, action_size)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.loss_fn = nn.MSELoss()

#     def act(self, state):
#         if random.random() < self.epsilon:
#             return random.randrange(self.action_size)
#         state = torch.FloatTensor(state).unsqueeze(0)
#         with torch.no_grad():
#             return torch.argmax(self.model(state)).item()

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def replay(self, batch_size=32):
#         if len(self.memory) < batch_size:
#             return
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state)))
#             q_values = self.model(torch.FloatTensor(state))
#             target_f = q_values.clone()
#             target_f[action] = target
#             loss = self.loss_fn(q_values, target_f.detach())
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

