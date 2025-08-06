# game_manager.py
import pygame
import random
import os
import sys

# Import game entities
from entities import Wall, Pellet, Pacman, Ghost

# Import game layouts and sprite configurations (assuming these are separate files)
from game_layouts import LAYOUTS
from sprite_configs import PACMAN_PATHS, GHOST_PATHS

# --- Configuration (Centralized here) ---
CELL_SIZE = 24
GRID_WIDTH = 28
GRID_HEIGHT = 31
SCORE_AREA_HEIGHT = 50
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + SCORE_AREA_HEIGHT
FPS = 60  # Main game speed control

HIGHSCORE_FILE = "highscore.txt"

# Colors (Centralized here)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)


class GameManager:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman Enhanced")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.win_font = pygame.font.SysFont("Arial", 72, bold=True)

        self.score = 0
        self.high_score = self._load_highscore()

        self.walls = pygame.sprite.Group()
        self.pellets = pygame.sprite.Group()
        self.pacman = None
        self.ghosts = pygame.sprite.Group()

        self.game_over = False
        self.game_won = False

        self._load_game_layout()

        # Group for drawing all static elements (walls, pellets)
        self.all_sprites_except_pacman_ghosts = pygame.sprite.Group(
            *self.walls, *self.pellets
        )

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

    def _load_game_layout(self):
        """Loads the game layout from a randomly selected map."""
        selected_layout = LAYOUTS[10]
        # selected_layout = random.choice(LAYOUTS)
        ghost_types = ["blinky", "pinky", "inky", "clyde"]
        ghost_count = 0
        MAX_GHOSTS = 5

        for row_idx, row in enumerate(selected_layout):
            for col_idx, cell in enumerate(row):
                x = col_idx * CELL_SIZE
                y = row_idx * CELL_SIZE + SCORE_AREA_HEIGHT
                if cell == "#":
                    self.walls.add(Wall(x, y))
                elif cell == ".":
                    self.pellets.add(Pellet(x, y))
                elif cell == "P":
                    self.pacman = Pacman(x, y, PACMAN_PATHS)
                elif cell == "G" and ghost_count < MAX_GHOSTS:
                    ghost_type = ghost_types[ghost_count % len(ghost_types)]
                    self.ghosts.add(
                        Ghost(x, y, ghost_type, GHOST_PATHS, FPS)
                    )  # Pass FPS to Ghost
                    ghost_count += 1

    def _handle_input(self):
        """Handles user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.keys = pygame.key.get_pressed()

    def _update_game_state(self):
        """Updates all game objects and checks for collisions/win conditions."""
        if not self.game_over and not self.game_won:
            self.pacman.update(self.keys, self.walls)

            for ghost in self.ghosts:
                ghost.update(self.walls, self.pacman, FPS)  # Pass FPS to ghost update

            collided_pellets = pygame.sprite.spritecollide(
                self.pacman, self.pellets, True
            )
            self.score += len(collided_pellets) * 10

            if not self.pellets:
                self.game_won = True
                self.game_over = True

            if pygame.sprite.spritecollideany(self.pacman, self.ghosts):
                self.game_over = True

    def _draw_elements(self):
        """Draws all game elements to the screen."""
        self.screen.fill(BLACK)

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

        self.all_sprites_except_pacman_ghosts.draw(self.screen)
        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)

        if self.game_over:
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_highscore()

            if self.game_won:
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

    def run(self):
        """Main game loop."""
        self.running = True
        while self.running:
            self._handle_input()
            self._update_game_state()
            self._draw_elements()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
