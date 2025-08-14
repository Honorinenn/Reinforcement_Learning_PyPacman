# # File: dqn_train.py

# import torch
# import numpy as np
# from dqn_agent import DQNAgent
# from game_manager import GameManager
# import pygame


# def get_state(game_manager):
#     """
#     This function translates the current game state into a feature vector for the DQN.
#     The state includes Pacman's position, the positions of the four ghosts, and
#     the number of remaining pellets.
#     """
#     pacman = game_manager.pacman
#     ghosts = game_manager.ghosts
#     pellets = game_manager.pellets

#     state_features = []

#     # Pacman's position
#     state_features.append(pacman.rect.x)
#     state_features.append(pacman.rect.y)

#     # Ghost positions
#     for ghost in ghosts.sprites():
#         state_features.append(ghost.rect.x)
#         state_features.append(ghost.rect.y)

#     # Number of pellets
#     state_features.append(len(pellets))

#     return np.array(state_features, dtype=np.float32)


# def train_agent():
#     # Define hyperparameters
#     state_size = 11  # Pacman (x,y) + 4 ghosts (x,y) + pellet count
#     action_size = 4  # Up, Down, Left, Right
#     batch_size = 64
#     num_episodes = 5000
#     target_update_frequency = 10

#     # Flag to enable/disable the game display
#     display_enabled = True

#     agent = DQNAgent(state_size=state_size, action_size=action_size)
#     game_manager = GameManager(headless=not display_enabled)
#     clock = pygame.time.Clock()

#     for episode in range(num_episodes):
#         game_manager.reset()
#         state = get_state(game_manager)
#         done = False
#         total_reward = 0

#         while not done:
#             # Handle Pygame events to close the window
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     return

#             action = agent.get_action(state)
#             next_state_raw, reward, done = game_manager.step(action)
#             next_state = get_state(game_manager)

#             agent.store_experience(state, action, reward, next_state, done)
#             agent.train(batch_size)

#             state = next_state
#             total_reward += reward

#             if display_enabled:
#                 game_manager.display()
#                 clock.tick(game_manager.FPS)

#         agent.decay_epsilon()

#         # Update the target network every 'target_update_frequency' episodes
#         if episode % target_update_frequency == 0:
#             agent.update_target_network()

#         if episode % 100 == 0:
#             print(
#                 f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
#             )
#             # Optionally, save the model weights
#             # torch.save(agent.policy_net.state_dict(), f"dqn_pacman_{episode}.pth")

#     pygame.quit()


# if __name__ == "__main__":
#     train_agent()
# File: dqn_train.py

# import torch
# import numpy as np
# from dqn_agent import DQNAgent
# from game_manager import GameManager
# import pygame
# import time

# # Import the logger utilities
# from logger import TrainingLogger, EpisodeLog


# def get_state(game_manager):
#     """
#     Converts the current game state into a feature vector for the DQN.
#     """
#     pacman = game_manager.pacman
#     ghosts = game_manager.ghosts
#     pellets = game_manager.pellets

#     state_features = [pacman.rect.x, pacman.rect.y]

#     for ghost in ghosts.sprites():
#         state_features.append(ghost.rect.x)
#         state_features.append(ghost.rect.y)

#     state_features.append(len(pellets))
#     return np.array(state_features, dtype=np.float32)


# def train_agent():
#     state_size = 11
#     action_size = 4
#     batch_size = 64
#     num_episodes = 1000
#     target_update_frequency = 10
#     display_enabled = True

#     agent = DQNAgent(state_size=state_size, action_size=action_size)
#     game_manager = GameManager(headless=not display_enabled)
#     clock = pygame.time.Clock()

#     # === Initialize logger ===
#     logger = TrainingLogger(project="RL_PacMan", tag="dqn_run", avg_window=20)
#     logger.save_config(
#         {
#             "state_size": state_size,
#             "action_size": action_size,
#             "batch_size": batch_size,
#             "episodes": num_episodes,
#             "target_update_frequency": target_update_frequency,
#         }
#     )

#     global_step = 0

#     for episode in range(num_episodes):
#         game_manager.reset()
#         state = get_state(game_manager)
#         done = False
#         total_reward = 0
#         episode_loss_sum = 0.0
#         steps = 0
#         start_time = time.time()

#         while not done:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     return

#             action = agent.get_action(state)
#             next_state_raw, reward, done = game_manager.step(action)
#             next_state = get_state(game_manager)

#             agent.store_experience(state, action, reward, next_state, done)
#             loss = agent.train(batch_size)
#             if loss is not None:
#                 episode_loss_sum += loss

#             state = next_state
#             total_reward += reward
#             steps += 1
#             global_step += 1

#             if display_enabled:
#                 game_manager.display()
#                 clock.tick(game_manager.FPS)

#         agent.decay_epsilon()

#         if episode % target_update_frequency == 0:
#             agent.update_target_network()

#         avg_loss = episode_loss_sum / max(1, steps)
#         duration = time.time() - start_time

#         # === Log results ===
#         ma = logger.log_episode(
#             EpisodeLog(
#                 episode=episode,
#                 reward=total_reward,
#                 loss=avg_loss,
#                 epsilon=agent.epsilon,
#                 steps=steps,
#                 duration_sec=duration,
#             )
#         )

#         # === Checkpoint save every 50 episodes ===
#         if episode % 50 == 0:
#             logger.maybe_checkpoint(
#                 model=agent.policy_net,
#                 optimizer=agent.optimizer,
#                 global_step=global_step,
#                 extra={"episode": episode, "ma_reward": ma},
#             )

#         if episode % 100 == 0:
#             print(
#                 f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
#             )

#     # Final save
#     logger.maybe_checkpoint(
#         agent.policy_net, agent.optimizer, global_step, extra={"final": True}
#     )
#     logger.save_plot()

#     pygame.quit()


# if __name__ == "__main__":
#     train_agent()

import torch
import numpy as np
from dqn_agent import DQNAgent
from game_manager import GameManager
import pygame
import time

# logging
from logger import TrainingLogger, EpisodeLog


def get_state(game_manager):
    """
    Converts the current game state into a feature vector for the DQN.
    """
    pacman = game_manager.pacman
    ghosts = game_manager.ghosts
    pellets = game_manager.pellets

    state_features = [pacman.rect.x, pacman.rect.y]

    for ghost in ghosts.sprites():
        state_features.append(ghost.rect.x)
        state_features.append(ghost.rect.y)

    state_features.append(len(pellets))
    return np.array(state_features, dtype=np.float32)


def train_agent():
    state_size = 11
    action_size = 4
    batch_size = 64
    num_episodes = 1000
    target_update_frequency = 10
    display_enabled = True

    agent = DQNAgent(state_size=state_size, action_size=action_size)
    game_manager = GameManager(headless=not display_enabled)
    clock = pygame.time.Clock()

    # === init logger ===
    logger = TrainingLogger(project="RL_PacMan", tag="dqn_run", avg_window=20)
    logger.save_config(
        {
            "state_size": state_size,
            "action_size": action_size,
            "batch_size": batch_size,
            "episodes": num_episodes,
            "target_update_frequency": target_update_frequency,
        }
    )
    global_step = 0

    for episode in range(num_episodes):
        game_manager.reset()
        state = get_state(game_manager)
        done = False
        total_reward = 0
        ep_loss_sum = 0.0
        ep_steps = 0
        t0 = time.time()

        while not done:
            # Handle Pygame events to close the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # use legal action mask to reduce thrashing into walls
            legal = game_manager.legal_actions()
            action = agent.get_action(state, legal_actions=legal)
            next_state_raw, reward, done = game_manager.step(action)
            next_state = get_state(game_manager)

            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train(batch_size)
            if loss is not None:
                ep_loss_sum += float(loss)

            state = next_state
            total_reward += reward
            ep_steps += 1
            global_step += 1

            if display_enabled:
                game_manager.display()
                clock.tick(game_manager.FPS)

        agent.decay_epsilon()

        # Update the target network every 'target_update_frequency' episodes
        if episode % target_update_frequency == 0:
            agent.update_target_network()

        # --- metrics & logging ---
        duration = time.time() - t0
        avg_loss = ep_loss_sum / max(1, ep_steps)
        pellets_initial = game_manager.initial_pellet_count
        pellets_left = len(game_manager.pellets)
        pellet_clear_rate = (pellets_initial - pellets_left) / max(1, pellets_initial)

        ma = logger.log_episode(
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

        # checkpoint every 50 eps
        if episode % 50 == 0:
            logger.maybe_checkpoint(
                model=agent.policy_net,
                optimizer=agent.optimizer,
                global_step=global_step,
                extra={"episode": episode, "ma_reward": ma},
            )

        if episode % 100 == 0:
            print(
                f"Episode: {episode}, Reward: {total_reward:.1f}, Clear%: {pellet_clear_rate*100:.1f}, Eps: {agent.epsilon:.2f}"
            )

    pygame.quit()


if __name__ == "__main__":
    train_agent()
