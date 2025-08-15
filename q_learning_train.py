# File: q_learning_train.py

import numpy as np
import time
import pygame
from q_learning_agent import QLearningAgent
from game_manager import GameManager
from logger import TrainingLogger, EpisodeLog


def get_state(game_manager):
    """
    Converts the current game state into a feature vector for Q-Learning.
    Uses a simplified state representation compared to DQN.
    """
    pacman = game_manager.pacman
    ghosts = game_manager.ghosts
    pellets = game_manager.pellets

    # Basic state features: pacman position + ghost positions + pellet count
    state_features = [pacman.rect.x, pacman.rect.y]

    # Add ghost positions (up to 4 ghosts)
    ghost_list = list(ghosts.sprites())
    for i in range(4):  # Assume max 4 ghosts
        if i < len(ghost_list):
            state_features.extend([ghost_list[i].rect.x, ghost_list[i].rect.y])
        else:
            state_features.extend([0, 0])  # Padding for missing ghosts

    # Add pellet count
    state_features.append(len(pellets))

    return np.array(state_features, dtype=np.float32)


def train_q_learning_agent():
    """Train the Q-Learning agent."""
    # Hyperparameters
    action_size = 4
    num_episodes = 2000
    display_enabled = True
    save_frequency = 100

    # Q-Learning specific parameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # Initialize agent and game
    agent = QLearningAgent(
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    game_manager = GameManager(headless=not display_enabled)
    clock = pygame.time.Clock()

    # Initialize logger
    logger = TrainingLogger(
        project="RL_PacMan", tag="qlearning_baseline", avg_window=50
    )
    logger.save_config(
        {
            "algorithm": "Q-Learning",
            "action_size": action_size,
            "episodes": num_episodes,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon_start": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        }
    )

    # Try to load existing Q-table
    q_table_path = "models/q_learning_agent.pkl"
    agent.load_q_table(q_table_path)

    print("Starting Q-Learning training...")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {discount_factor}")
    print(f"Initial epsilon: {epsilon}")

    for episode in range(num_episodes):
        game_manager.reset()
        state = get_state(game_manager)
        done = False
        total_reward = 0
        ep_steps = 0
        t0 = time.time()

        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get legal actions to reduce invalid moves
            legal_actions = game_manager.legal_actions()

            # Choose action
            action = agent.get_action(state, legal_actions=legal_actions)

            # Take action
            next_state_raw, reward, done = game_manager.step(action)
            next_state = get_state(game_manager)

            # Update Q-table
            agent.update_q_value(state, action, reward, next_state, done)

            # Update state and tracking
            state = next_state
            total_reward += reward
            ep_steps += 1

            # Display if enabled
            if display_enabled:
                game_manager.display()
                clock.tick(game_manager.FPS)

        # Decay epsilon
        agent.decay_epsilon()

        # Calculate metrics
        duration = time.time() - t0
        pellets_initial = game_manager.initial_pellet_count
        pellets_left = len(game_manager.pellets)
        pellet_clear_rate = (pellets_initial - pellets_left) / max(1, pellets_initial)

        # Get Q-table statistics
        q_stats = agent.get_q_table_stats()

        # Log episode
        ma_reward = logger.log_episode(
            EpisodeLog(
                episode=episode,
                reward=total_reward,
                epsilon=agent.epsilon,
                steps=ep_steps,
                duration_sec=duration,
                info={
                    "pellet_clear_rate": pellet_clear_rate,
                    "q_table_size": q_stats["num_states"],
                    "avg_q_value": q_stats["avg_q_value"],
                },
            )
        )

        # Save Q-table periodically
        if episode % save_frequency == 0:
            agent.save_q_table(q_table_path)

        # Print progress
        if episode % 100 == 0:
            print(f"Episode: {episode}")
            print(f"  Reward: {total_reward:.1f}")
            print(f"  Pellet Clear Rate: {pellet_clear_rate*100:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table size: {q_stats['num_states']}")
            print(f"  Moving Average Reward: {ma_reward:.2f}")
            print()

    # Final save
    agent.save_q_table(q_table_path)

    # Print final statistics
    final_stats = agent.get_q_table_stats()
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Total episodes: {num_episodes}")
    print(f"Final epsilon: {final_stats['current_epsilon']:.3f}")
    print(f"Q-table size: {final_stats['num_states']} states")
    print(f"Total steps: {final_stats['total_steps']}")
    print(f"Average Q-value: {final_stats['avg_q_value']:.3f}")
    print(f"Final moving average reward: {ma_reward:.2f}")

    pygame.quit()


if __name__ == "__main__":
    train_q_learning_agent()
