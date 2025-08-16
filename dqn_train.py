# File: dqn_train.py

import torch
import numpy as np
from dqn_agent import DQNAgent
from game_manager import GameManager
import pygame
import time

# logging
from logger import TrainingLogger, EpisodeLog


def get_state(game_manager, radius=3):
    pac = game_manager.pacman
    ghosts = game_manager.ghosts
    pellets = game_manager.pellets

    # old scalar features
    f = [pac.rect.x, pac.rect.y]
    for g in ghosts.sprites():
        f.extend([g.rect.x, g.rect.y])
    f.append(len(pellets))

    # new: egocentric 3-channel 7x7 (pellets/walls/ghosts)
    local_patch = game_manager.crop_egocentric(radius=radius)  # 3*49 = 147
    f.extend(local_patch)
    return np.array(f, dtype=np.float32)


def train_agent():
    # state_size = 11
    state_size = 158
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

    # Track win rate.
    win_rate_window = 100
    win_history = []

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

        # Check for win and update win history.
        if game_manager.has_won:
            win_history.append(1)
        else:
            win_history.append(0)

        # Keep the win history to the size of the window.
        if len(win_history) > win_rate_window:
            win_history.pop(0)

        # Calculate and log the current win rate.
        current_win_rate = (sum(win_history) / len(win_history)) * 100
        print(f"Episode: {episode+1}, Win Rate: {current_win_rate:.2f}%")

        # Stop training if a 100% win rate is achieved.
        if current_win_rate == 100 and len(win_history) == win_rate_window:
            print(
                f"100% win rate achieved over the last {win_rate_window} episodes! Stopping training."
            )
            break

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
