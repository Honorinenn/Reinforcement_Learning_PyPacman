# File: dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for handling high-dimensional state spaces.
    Enhanced architecture with dropout and layer normalization for better performance.
    """

    def __init__(
        self, state_size, action_size, hidden_sizes=[256, 256, 128], dropout_rate=0.2
    ):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Build the network layers dynamically
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.LayerNorm(hidden_size),  # LayerNorm works with single samples
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_size = hidden_size

        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    Better for environments where some actions don't significantly affect the value.
    """

    def __init__(
        self, state_size, action_size, hidden_sizes=[256, 256], dropout_rate=0.2
    ):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),  # LayerNorm works with single samples
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),  # LayerNorm works with single samples
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], 128), nn.ReLU(), nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], 128), nn.ReLU(), nn.Linear(128, action_size)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through the dueling architecture."""
        features = self.feature_layers(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class ConvDQN(nn.Module):
    """
    Convolutional DQN for processing grid-based state representations.
    Useful when the state includes spatial information like the game board.
    """

    def __init__(self, grid_channels=3, grid_size=7, scalar_features=11, action_size=4):
        super(ConvDQN, self).__init__()
        self.grid_channels = grid_channels
        self.grid_size = grid_size
        self.scalar_features = scalar_features
        self.action_size = action_size

        # Convolutional layers for processing the grid
        self.conv_layers = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate the size after convolution
        conv_output_size = 64 * grid_size * grid_size

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + scalar_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass expecting input with both grid and scalar features.
        Input format: [batch_size, grid_channels * grid_size^2 + scalar_features]
        """
        batch_size = x.size(0)

        # Split input into grid and scalar parts
        grid_features = x[:, : self.grid_channels * self.grid_size * self.grid_size]
        scalar_features = x[:, self.grid_channels * self.grid_size * self.grid_size :]

        # Reshape grid features for convolution
        grid_features = grid_features.view(
            batch_size, self.grid_channels, self.grid_size, self.grid_size
        )

        # Process grid through convolution
        conv_output = self.conv_layers(grid_features)
        conv_output = conv_output.view(batch_size, -1)  # Flatten

        # Combine conv output with scalar features
        combined_features = torch.cat([conv_output, scalar_features], dim=1)

        # Process through fully connected layers
        return self.fc_layers(combined_features)
