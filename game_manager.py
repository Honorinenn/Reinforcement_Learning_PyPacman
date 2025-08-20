# game_manager.py
import pygame
import random
import os
import sys

# Import game entities
from entities import Wall, Pellet, Pacman, Ghost

# Import game layouts and sprite configurations
from game_layouts import LAYOUTS
from sprite_configs import PACMAN_PATHS, GHOST_PATHS

# --- Configuration (Centralized here) ---
CELL_SIZE = 24
GRID_WIDTH = 28
GRID_HEIGHT = 31
SCORE_AREA_HEIGHT = 50
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + SCORE_AREA_HEIGHT
FPS = 120  # Main game speed control

HIGHSCORE_FILE = "highscore.txt"

# === RL reward constants (tunable) ===
STEP_PENALTY = -0.02
PELLET_R = 1.0
WIN_BONUS = 1000.0  # Increased from 300.0 for a stronger incentive
DEATH_PENALTY = -1000.0  # Increased from -200.0 for a stronger penalty
SHAPING_SCALE = 0.002  # tiny weight for distance shaping
STUCK_LIMIT = 400  # steps since last pellet before early terminate

# Colors (Centralized here)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)

# Directions mapping for the agent's actions
ACTION_DIRECTIONS = {
    0: pygame.Vector2(0, -1),  # Up
    1: pygame.Vector2(0, 1),  # Down
    2: pygame.Vector2(-1, 0),  # Left
    3: pygame.Vector2(1, 0),  # Right
}


class GameManager:
    def __init__(self, headless=True):
        """
        Initializes the game manager.
        :param headless: If True, runs without a visible display, suitable for training.
        """
        self.headless = headless
        pygame.init()
        if self.headless:
            # Create a dummy display surface to allow sprites to be loaded
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
        else:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("PyPacman")

        self.pacman = None
        self.ghosts = pygame.sprite.Group()
        self.pellets = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.has_won = False
        self.current_layout = LAYOUTS[1]
        self.high_score = self._load_highscore()
        self.font = pygame.font.Font(None, 36)
        self.win_font = pygame.font.Font(None, 72)
        self.FPS = FPS

        # Initialize exploration bookkeeping variables before loading the layout
        self.visited = set()
        self.steps_since_last_pellet = 0

        self._load_game_layout()

    def _load_game_layout(self):
        """
        Loads the game layout, setting up the board, pacman, and ghosts.
        """
        layout_data = self.current_layout

        # Reset game elements by clearing sprite groups
        self.walls.empty()
        self.pellets.empty()
        self.ghosts.empty()
        self.pacman = None

        pacman_pos = None
        ghost_starts = []
        pellet_positions = []

        for y, row in enumerate(layout_data):
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add(
                        Wall(x * CELL_SIZE, y * CELL_SIZE + SCORE_AREA_HEIGHT)
                    )
                elif char == "P":
                    pacman_pos = (x * CELL_SIZE, y * CELL_SIZE + SCORE_AREA_HEIGHT)
                elif char == "G":
                    ghost_starts.append(
                        (x * CELL_SIZE, y * CELL_SIZE + SCORE_AREA_HEIGHT)
                    )
                elif char == ".":
                    pellet_positions.append(
                        (x * CELL_SIZE, y * CELL_SIZE + SCORE_AREA_HEIGHT)
                    )

        # Instantiate entities
        if pacman_pos:
            self.pacman = Pacman(pacman_pos[0], pacman_pos[1], PACMAN_PATHS)
        for i, pos in enumerate(ghost_starts):
            self.ghosts.add(Ghost(pos[0], pos[1], f"ghost_{i}", GHOST_PATHS, FPS))
        for pos in pellet_positions:
            self.pellets.add(Pellet(pos[0], pos[1]))
        self.initial_pellet_count = len(self.pellets)

        # Reset all ghosts to ensure they start fresh
        for ghost in self.ghosts:
            ghost.reset()
        # The following lines are removed from this method because they are
        # initialized in __init__ and will be reset in the `reset()` method.
        # self.visited.clear()
        # self.steps_since_last_pellet = 0

    def _load_highscore(self):
        """Loads the high score from a file."""
        if os.path.exists(HIGHSCORE_FILE):
            try:
                with open(HIGHSCORE_FILE, "r") as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                return 0
        return 0

    def _save_highscore(self):
        """Saves the current score as the new high score if it's higher."""
        try:
            with open(HIGHSCORE_FILE, "w") as f:
                f.write(str(self.high_score))
        except IOError:
            print(f"Error: Could not save high score to {HIGHSCORE_FILE}")

    def reset(self):
        """
        Resets the game state to the beginning of a new episode.
        """
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.has_won = False

        # Reset exploration bookkeeping variables
        self.visited.clear()
        self.steps_since_last_pellet = 0

        self._load_game_layout()

    # Add to GameManager
    def get_grid(self, px=None, py=None):
        # integer grid coords for pacman center
        if px is None or py is None:
            px = self.pacman.rect.centerx // CELL_SIZE
            py = (self.pacman.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE

        W, H = GRID_WIDTH, GRID_HEIGHT
        pellets = [[0] * W for _ in range(H)]
        walls = [[0] * W for _ in range(H)]
        ghosts = [[0] * W for _ in range(H)]

        for p in self.pellets:
            x = p.rect.centerx // CELL_SIZE
            y = (p.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE
            if 0 <= x < W and 0 <= y < H:
                pellets[y][x] = 1

        for w in self.walls:
            x = w.rect.x // CELL_SIZE
            y = (w.rect.y - SCORE_AREA_HEIGHT) // CELL_SIZE
            if 0 <= x < W and 0 <= y < H:
                walls[y][x] = 1

        for g in self.ghosts:
            x = g.rect.centerx // CELL_SIZE
            y = (g.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE
            if 0 <= x < W and 0 <= y < H:
                ghosts[y][x] = 1

        return pellets, walls, ghosts, (px, py)

    def crop_egocentric(self, radius=3):
        pellets, walls, ghosts, (px, py) = self.get_grid()
        patch = []
        for ch in (pellets, walls, ghosts):
            band = []
            for dy in range(-radius, radius + 1):
                row = []
                for dx in range(-radius, radius + 1):
                    x, y = px + dx, py + dy
                    val = 0
                    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                        val = ch[y][x]
                    row.append(val)
                band.extend(row)
            patch.extend(band)  # flatten channels
        return patch  # length = 3 * (2r+1)^2

    # --- Helper: nearest pellet distance (pixel manhattan) ---
    def nearest_pellet_distance(self):
        px, py = self.pacman.rect.centerx, self.pacman.rect.centery
        if not self.pellets:
            return 0
        return min(
            abs(px - p.rect.centerx) + abs(py - p.rect.centery) for p in self.pellets
        )

    # --- Helper: legal actions from current pacman cell ---
    def legal_actions(self):
        acts = []
        for a, direction in ACTION_DIRECTIONS.items():
            test = self.pacman.rect.move(
                direction.x * self.pacman.speed, direction.y * self.pacman.speed
            )
            if not any(test.colliderect(w.rect) for w in self.walls):
                acts.append(a)
        return acts

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and done flag.
        """
        # Potential-based shaping (pre-move potential)
        prev_phi = -self.nearest_pellet_distance()

        # Get the direction from the action
        direction = ACTION_DIRECTIONS.get(action)
        if direction:
            # Update Pacman's next direction
            self.pacman.next_direction = direction

        # Update Pacman's position
        self.pacman.update({}, self.walls)

        # Update Ghosts' positions
        for ghost in self.ghosts:
            ghost.update(self.walls, self.pacman, FPS)

        # Calculate reward
        reward = STEP_PENALTY
        done = False

        # Check for pellet collision
        pellets_eaten = pygame.sprite.spritecollide(self.pacman, self.pellets, True)
        self.score += len(pellets_eaten) * 10  # keep UI scoring
        if len(pellets_eaten) > 0:
            self.steps_since_last_pellet = 0
        reward += len(pellets_eaten) * PELLET_R

        if not self.pellets:
            reward += WIN_BONUS
            done = True
            self.has_won = True
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_highscore()

        # Check for ghost collision
        if pygame.sprite.spritecollideany(self.pacman, self.ghosts):
            reward += DEATH_PENALTY
            done = True
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_highscore()
        # Novelty bonus for visiting new cells
        cell = (
            self.pacman.rect.centerx // CELL_SIZE,
            (self.pacman.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE,
        )
        if cell not in self.visited:
            self.visited.add(cell)
            reward += 0.05

        # Potential-based shaping (post-move)
        new_phi = -self.nearest_pellet_distance()
        gamma = 0.99
        reward += (new_phi - gamma * prev_phi) * SHAPING_SCALE

        # Early termination if stuck too long without progress
        self.steps_since_last_pellet += 1
        if self.steps_since_last_pellet >= STUCK_LIMIT and not done:
            done = True
            reward += -5.0  # mild penalty for stalling

        return self.pacman, reward, done

    def display(self):
        """Draws the game state to the screen."""
        self.screen.fill(BLACK)
        self.walls.draw(self.screen)
        self.pellets.draw(self.screen)
        self.pacman.draw(self.screen)
        self.ghosts.draw(self.screen)

        # Draw the score and high score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        high_score_text = self.font.render(
            f"High Score: {self.high_score}", True, WHITE
        )
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(
            high_score_text, (SCREEN_WIDTH - high_score_text.get_width() - 10, 10)
        )

        pygame.draw.line(
            self.screen,
            WHITE,
            (0, SCORE_AREA_HEIGHT - 2),
            (SCREEN_WIDTH, SCORE_AREA_HEIGHT - 2),
            2,
        )

        if self.game_over:
            if self.has_won:
                win_text = self.win_font.render("YOU WIN!", True, GREEN)
                text_rect = win_text.get_rect(
                    center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                )
                self.screen.blit(win_text, text_rect)
            else:
                game_over_text = self.win_font.render("GAME OVER!", True, RED)
                text_rect = game_over_text.get_rect(
                    center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                )
                self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()
