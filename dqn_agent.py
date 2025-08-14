# # File: dqn_agent.py

# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# from dqn_model import DQN


# class DQNAgent:
#     def __init__(
#         self,
#         state_size,
#         action_size,
#         learning_rate=0.001,
#         gamma=0.95,
#         epsilon_start=1.0,
#         epsilon_end=0.01,
#         epsilon_decay=0.995,
#         memory_size=10000,
#     ):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay

#         self.memory = deque(maxlen=memory_size)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.policy_net = DQN(state_size, action_size).to(self.device)
#         self.target_net = DQN(state_size, action_size).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()

#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
#         self.criterion = nn.MSELoss()

#     def get_action(self, state, legal_actions=None):
#         """Epsilon-greedy; if legal_actions provided, sample within them and mask illegal for argmax."""
#         if legal_actions is not None and len(legal_actions) == 0:
#             legal_actions = None  # fallback
#         if random.random() <= self.epsilon:
#             return (
#                 random.choice(legal_actions)
#                 if legal_actions
#                 else random.randrange(self.action_size)
#             )
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             q_values = self.policy_net(state).squeeze(0)  # [A]
#         if legal_actions is None:
#             return int(torch.argmax(q_values).item())
#         mask = torch.full_like(q_values, float("-inf"))
#         mask[torch.tensor(legal_actions, dtype=torch.long, device=q_values.device)] = (
#             0.0
#         )
#         masked_q = q_values + mask
#         return int(torch.argmax(masked_q).item())

#     def store_experience(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def train(self, batch_size):
#         if len(self.memory) < batch_size:
#             return None

#         minibatch = random.sample(self.memory, batch_size)
#         states, actions, rewards, next_states, dones = zip(*minibatch)

#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)

#         q_values = self.policy_net(states).gather(1, actions)

#         with torch.no_grad():
#             next_q_values = self.target_net(next_states).max(1)[0].detach()

#         target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
#         target_q_values = target_q_values.unsqueeze(1)

#         loss = self.criterion(q_values, target_q_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()

#     def decay_epsilon(self):
#         self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

#     def update_target_network(self):
#         self.target_net.load_state_dict(self.policy_net.state_dict())

import random
import torch
import torch.nn.functional as F
from collections import deque
from dqn_model import DQN


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, legal_actions=None):
        if legal_actions is not None and len(legal_actions) == 0:
            legal_actions = None
        if random.random() <= self.epsilon:
            return (
                random.choice(legal_actions)
                if legal_actions
                else random.randrange(self.action_size)
            )
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0)
        if legal_actions is None:
            return int(torch.argmax(q_values).item())
        mask = torch.full_like(q_values, float("-inf"))
        mask[torch.tensor(legal_actions, dtype=torch.long, device=q_values.device)] = (
            0.0
        )
        masked_q = q_values + mask
        return int(torch.argmax(masked_q).item())

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
