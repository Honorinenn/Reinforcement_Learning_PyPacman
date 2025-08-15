# File: train_agents.py

import argparse
import numpy as np
import time
import pygame
import os
from pathlib import Path

# Import agents
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from game_manager import GameManager
from logger import TrainingLogger, EpisodeLog


def get_state_for_agent(game_manager, agent_type="q_learning"):
    """
    Get state representation based on agent type.

    Args:
        game_manager: Game manager instance
        agent_type: Type of agent ("q_learning" or "dqn")

    Returns:
        State representation as numpy array
    """
    if agent_type == "q_learning":
        # Simplified state for Q-Learning
        pacman = game_manager.pacman
        ghosts = game_manager.ghosts
        pellets = game_manager.pellets

        state_features = [pacman.rect.x, pacman.rect.y]

        # Add ghost positions (up to 4 ghosts)
        ghost_list = list(ghosts.sprites())
        for i in range(4):
            if i < len(ghost_list):
                state_features.extend([ghost_list[i].rect.x, ghost_list[i].rect.y])
            else:
                state_features.extend([0, 0])

        state_features.append(len(pellets))
        return np.array(state_features, dtype=np.float32)

    elif agent_type == "dqn":
        # Enhanced state for DQN with egocentric view
        pacman = game_manager.pacman
        ghosts = game_manager.ghosts
        pellets = game_manager.pellets

        # Basic features
        state_features = [pacman.rect.x, pacman.rect.y]

        for ghost in ghosts.sprites():
            state_features.append(ghost.rect.x)
            state_features.append(ghost.rect.y)

        state_features.append(len(pellets))

        # Add egocentric local patch
        local_patch = game_manager.crop_egocentric(radius=3)
        state_features.extend(local_patch)

        return np.array(state_features, dtype=np.float32)


def train_q_learning(args):
    """Train Q-Learning agent."""
    print("=" * 60)
    print("TRAINING Q-LEARNING AGENT")
    print("=" * 60)

    # Initialize agent
    agent = QLearningAgent(
        action_size=4,
        learning_rate=args.q_learning_rate,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    # Initialize game and logger
    game_manager = GameManager(headless=not args.display)
    logger = TrainingLogger(
        project="RL_PacMan",
        tag=f"qlearning_{args.tag}" if args.tag else "qlearning",
        avg_window=args.avg_window,
    )

    # Save configuration
    logger.save_config(
        {
            "algorithm": "Q-Learning",
            "episodes": args.episodes,
            "learning_rate": args.q_learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
        }
    )

    # Load existing Q-table if available
    q_table_path = (
        f"models/q_learning_{args.tag}.pkl" if args.tag else "models/q_learning.pkl"
    )
    if args.load_model and os.path.exists(q_table_path):
        agent.load_q_table(q_table_path)

    clock = pygame.time.Clock() if args.display else None

    print(f"Training for {args.episodes} episodes...")
    print(f"Learning rate: {args.q_learning_rate}")
    print(f"Discount factor: {args.gamma}")
    print(f"Initial epsilon: {args.epsilon}")

    for episode in range(args.episodes):
        game_manager.reset()
        state = get_state_for_agent(game_manager, "q_learning")
        done = False
        total_reward = 0
        ep_steps = 0
        t0 = time.time()

        while not done:
            if args.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # Get legal actions and choose action
            legal_actions = game_manager.legal_actions()
            action = agent.get_action(state, legal_actions=legal_actions)

            # Take action
            _, reward, done = game_manager.step(action)
            next_state = get_state_for_agent(game_manager, "q_learning")

            # Update Q-table
            agent.update_q_value(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            ep_steps += 1

            if args.display:
                game_manager.display()
                clock.tick(game_manager.FPS)

        # Decay epsilon
        agent.decay_epsilon()

        # Calculate metrics
        duration = time.time() - t0
        pellets_initial = game_manager.initial_pellet_count
        pellets_left = len(game_manager.pellets)
        pellet_clear_rate = (pellets_initial - pellets_left) / max(1, pellets_initial)
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

        # Save periodically
        if episode % args.save_frequency == 0:
            os.makedirs("models", exist_ok=True)
            agent.save_q_table(q_table_path)

        # Print progress
        if episode % args.print_frequency == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Clear: {pellet_clear_rate*100:4.1f}% | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Q-states: {q_stats['num_states']:4d} | "
                f"MA: {ma_reward:6.2f}"
            )

    # Final save
    os.makedirs("models", exist_ok=True)
    agent.save_q_table(q_table_path)

    # Print final statistics
    final_stats = agent.get_q_table_stats()
    print("\n" + "=" * 60)
    print("Q-LEARNING TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Final epsilon: {final_stats['current_epsilon']:.3f}")
    print(f"Q-table size: {final_stats['num_states']} states")
    print(f"Total steps: {final_stats['total_steps']}")
    print(f"Average Q-value: {final_stats['avg_q_value']:.3f}")
    print(f"Final moving average reward: {ma_reward:.2f}")

    if args.display:
        pygame.quit()

    return agent, logger


def train_dqn(args):
    """Train DQN agent."""
    print("=" * 60)
    print("TRAINING DQN AGENT")
    print("=" * 60)

    # State size calculation
    if args.dqn_architecture == "conv":
        state_size = 158  # 3*7*7 + 11 for conv architecture
    else:
        state_size = 158  # Standard/Dueling with egocentric view

    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=4,
        architecture=args.dqn_architecture,
        memory_size=args.memory_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.dqn_learning_rate,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        use_double_dqn=args.use_double_dqn,
    )

    # Initialize game and logger
    game_manager = GameManager(headless=not args.display)
    logger = TrainingLogger(
        project="RL_PacMan",
        tag=(
            f"dqn_{args.dqn_architecture}_{args.tag}"
            if args.tag
            else f"dqn_{args.dqn_architecture}"
        ),
        avg_window=args.avg_window,
    )

    # Save configuration
    logger.save_config(
        {
            "algorithm": "DQN",
            "architecture": args.dqn_architecture,
            "episodes": args.episodes,
            "state_size": state_size,
            "memory_size": args.memory_size,
            "batch_size": args.batch_size,
            "learning_rate": args.dqn_learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "target_update_freq": args.target_update_freq,
            "use_double_dqn": args.use_double_dqn,
        }
    )

    # Load existing model if available
    model_path = (
        f"models/dqn_{args.dqn_architecture}_{args.tag}.pth"
        if args.tag
        else f"models/dqn_{args.dqn_architecture}.pth"
    )
    if args.load_model and os.path.exists(model_path):
        agent.load_model(model_path)

    clock = pygame.time.Clock() if args.display else None
    global_step = 0

    print(f"Training for {args.episodes} episodes...")
    print(f"Architecture: {args.dqn_architecture}")
    print(f"State size: {state_size}")
    print(f"Memory size: {args.memory_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.dqn_learning_rate}")
    print(f"Use Double DQN: {args.use_double_dqn}")

    for episode in range(args.episodes):
        game_manager.reset()
        state = get_state_for_agent(game_manager, "dqn")
        done = False
        total_reward = 0
        ep_loss_sum = 0.0
        ep_steps = 0
        t0 = time.time()

        while not done:
            if args.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # Get legal actions and choose action
            legal_actions = game_manager.legal_actions()
            action = agent.get_action(state, legal_actions=legal_actions)

            # Take action
            _, reward, done = game_manager.step(action)
            next_state = get_state_for_agent(game_manager, "dqn")

            # Store experience and train
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train(args.batch_size)
            if loss is not None:
                ep_loss_sum += loss

            state = next_state
            total_reward += reward
            ep_steps += 1
            global_step += 1

            if args.display:
                game_manager.display()
                clock.tick(game_manager.FPS)

        # Decay epsilon
        agent.decay_epsilon()

        # Update target network
        if episode % args.target_update_freq == 0:
            agent.update_target_network()

        # Calculate metrics
        duration = time.time() - t0
        avg_loss = ep_loss_sum / max(1, ep_steps)
        pellets_initial = game_manager.initial_pellet_count
        pellets_left = len(game_manager.pellets)
        pellet_clear_rate = (pellets_initial - pellets_left) / max(1, pellets_initial)

        # Log episode
        ma_reward = logger.log_episode(
            EpisodeLog(
                episode=episode,
                reward=total_reward,
                loss=avg_loss,
                epsilon=agent.epsilon,
                steps=ep_steps,
                duration_sec=duration,
                info={"pellet_clear_rate": pellet_clear_rate},
            )
        )

        # Save periodically
        if episode % args.save_frequency == 0:
            os.makedirs("models", exist_ok=True)
            agent.save_model(model_path)
            logger.maybe_checkpoint(
                model=agent.policy_net,
                optimizer=agent.optimizer,
                global_step=global_step,
                extra={"episode": episode, "ma_reward": ma_reward},
            )

        # Print progress
        if episode % args.print_frequency == 0:
            stats = agent.get_stats()
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Clear: {pellet_clear_rate*100:4.1f}% | "
                f"Loss: {avg_loss:6.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Mem: {stats['memory_size']:4d} | "
                f"MA: {ma_reward:6.2f}"
            )

    # Final save
    os.makedirs("models", exist_ok=True)
    agent.save_model(model_path)

    # Print final statistics
    stats = agent.get_stats()
    print("\n" + "=" * 60)
    print("DQN TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Architecture: {args.dqn_architecture}")
    print(f"Final epsilon: {stats['epsilon']:.3f}")
    print(f"Training steps: {stats['training_steps']}")
    print(f"Memory size: {stats['memory_size']}")
    print(f"Average loss: {stats['avg_loss']:.4f}")
    print(f"Final moving average reward: {ma_reward:.2f}")

    if args.display:
        pygame.quit()

    return agent, logger


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for Pacman")

    # General arguments
    parser.add_argument(
        "--agent",
        type=str,
        choices=["q_learning", "dqn", "both"],
        default="both",
        help="Which agent to train",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--display", action="store_true", help="Show game display during training"
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Tag for saving models and logs"
    )
    parser.add_argument(
        "--load_model", action="store_true", help="Load existing model if available"
    )

    # Hyperparameters
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial exploration rate"
    )
    parser.add_argument(
        "--epsilon_min", type=float, default=0.01, help="Minimum exploration rate"
    )
    parser.add_argument(
        "--epsilon_decay", type=float, default=0.995, help="Exploration decay rate"
    )

    # Q-Learning specific
    parser.add_argument(
        "--q_learning_rate", type=float, default=0.1, help="Q-Learning learning rate"
    )

    # DQN specific
    parser.add_argument(
        "--dqn_learning_rate", type=float, default=0.001, help="DQN learning rate"
    )
    parser.add_argument(
        "--dqn_architecture",
        type=str,
        choices=["standard", "dueling", "conv"],
        default="standard",
        help="DQN architecture type",
    )
    parser.add_argument(
        "--memory_size", type=int, default=10000, help="Replay buffer size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=10,
        help="Target network update frequency",
    )
    parser.add_argument(
        "--use_double_dqn", action="store_true", default=True, help="Use Double DQN"
    )

    # Logging and saving
    parser.add_argument(
        "--save_frequency", type=int, default=100, help="Model save frequency"
    )
    parser.add_argument(
        "--print_frequency", type=int, default=100, help="Progress print frequency"
    )
    parser.add_argument(
        "--avg_window", type=int, default=50, help="Moving average window size"
    )

    args = parser.parse_args()

    # Create models directory
    os.makedirs("models", exist_ok=True)

    print("Starting Reinforcement Learning Training for Pacman")
    print(f"Agent(s): {args.agent}")
    print(f"Episodes: {args.episodes}")
    print(f"Display: {args.display}")
    print()

    results = {}

    if args.agent in ["q_learning", "both"]:
        agent, logger = train_q_learning(args)
        results["q_learning"] = {"agent": agent, "logger": logger}

    if args.agent in ["dqn", "both"]:
        agent, logger = train_dqn(args)
        results["dqn"] = {"agent": agent, "logger": logger}

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED!")
    print("=" * 60)

    if args.agent == "both":
        print("Both Q-Learning and DQN agents have been trained.")
        print("Check the 'runs/' directory for training logs and metrics.")
        print("Check the 'models/' directory for saved models.")


if __name__ == "__main__":
    main()
