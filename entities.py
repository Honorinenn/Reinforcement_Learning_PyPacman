# entities.py
import pygame
import random
from collections import deque
import math

# --- Configuration (Copied from main PyPacman for self-containment of entities) ---
CELL_SIZE = 24
GRID_WIDTH = 28
GRID_HEIGHT = 31
SCORE_AREA_HEIGHT = 50

# Colors (Copied for self-containment)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 150)
YELLOW = (255, 255, 0)
PINK = (255, 105, 180)
RED = (255, 0, 0)
GREEN = (0, 200, 0)

# Define additional colors for wall styling (Copied for self-containment)
WALL_BASE_COLOR = (24, 24, 80)
WALL_INNER_GLOW = (50, 50, 150)
WALL_BORDER_COLOR = (10, 10, 50)


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        self.image.fill((0, 0, 0, 0))  # Start with a transparent background

        border_radius = CELL_SIZE // 4

        pygame.draw.rect(
            self.image,
            WALL_BORDER_COLOR,
            self.image.get_rect(),
            border_radius=border_radius,
        )
        main_rect = self.image.get_rect().inflate(-2, -2)
        pygame.draw.rect(
            self.image, WALL_BASE_COLOR, main_rect, border_radius=border_radius
        )
        glow_rect = main_rect.inflate(-4, -4)
        pygame.draw.rect(
            self.image, WALL_INNER_GLOW, glow_rect, border_radius=border_radius
        )

        self.rect = self.image.get_rect(topleft=(x, y))


class Pellet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (CELL_SIZE // 2, CELL_SIZE // 2), 4)
        self.rect = self.image.get_rect(topleft=(x, y))


class Pacman(pygame.sprite.Sprite):
    def __init__(self, x, y, pacman_paths):
        super().__init__()
        self.images = {
            direction: [
                pygame.transform.scale(
                    pygame.image.load(path).convert_alpha(), (CELL_SIZE, CELL_SIZE)
                )
                for path in paths
            ]
            for direction, paths in pacman_paths.items()
        }
        self.direction_str = "right"
        self.anim_index = 0
        self.image = self.images[self.direction_str][self.anim_index]
        self.rect = self.image.get_rect(topleft=(x, y))
        self.speed = 3
        self.direction = pygame.Vector2(1, 0)
        self.next_direction = pygame.Vector2(1, 0)
        self.anim_speed = 0.15
        self.frame_counter = 0

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def update(self, keys, walls):
        # Handle input and set the next direction
        if keys[pygame.K_LEFT]:
            self.next_direction = pygame.Vector2(-1, 0)
        elif keys[pygame.K_RIGHT]:
            self.next_direction = pygame.Vector2(1, 0)
        elif keys[pygame.K_UP]:
            self.next_direction = pygame.Vector2(0, -1)
        elif keys[pygame.K_DOWN]:
            self.next_direction = pygame.Vector2(0, 1)

        # Try to change direction
        if self.next_direction != self.direction:
            if (self.rect.x - (self.rect.x // CELL_SIZE * CELL_SIZE)) == 0 and (
                self.rect.y - SCORE_AREA_HEIGHT
            ) % CELL_SIZE == 0:
                test_rect = self.rect.move(
                    self.next_direction.x * self.speed,
                    self.next_direction.y * self.speed,
                )
                if not any(test_rect.colliderect(wall.rect) for wall in walls):
                    self.direction = self.next_direction

        # Update direction string for animation
        if self.direction.x == -1:
            self.direction_str = "left"
        elif self.direction.x == 1:
            self.direction_str = "right"
        elif self.direction.y == -1:
            self.direction_str = "up"
        elif self.direction.y == 1:
            self.direction_str = "down"

        # Animate Pac-Man
        self.frame_counter += self.anim_speed
        self.anim_index = int(self.frame_counter) % len(self.images[self.direction_str])
        self.image = self.images[self.direction_str][self.anim_index]

        # Move Pac-Man
        self.rect.x += self.direction.x * self.speed
        if pygame.sprite.spritecollideany(self, walls):
            self.rect.x -= self.direction.x * self.speed
            if (self.direction.x > 0 and self.rect.right % CELL_SIZE != 0) or (
                self.direction.x < 0 and self.rect.left % CELL_SIZE != 0
            ):
                pass
            else:
                self.rect.x = round(self.rect.x / CELL_SIZE) * CELL_SIZE

        self.rect.y += self.direction.y * self.speed
        if pygame.sprite.spritecollideany(self, walls):
            self.rect.y -= self.direction.y * self.speed
            if (
                self.direction.y > 0
                and (self.rect.bottom - SCORE_AREA_HEIGHT) % CELL_SIZE != 0
            ) or (
                self.direction.y < 0
                and (self.rect.top - SCORE_AREA_HEIGHT) % CELL_SIZE != 0
            ):
                pass
            else:
                self.rect.y = (
                    round((self.rect.y - SCORE_AREA_HEIGHT) / CELL_SIZE) * CELL_SIZE
                    + SCORE_AREA_HEIGHT
                )


class Ghost(pygame.sprite.Sprite):
    def __init__(self, x, y, ghost_type, ghost_paths, fps):
        super().__init__()
        self.image = pygame.transform.scale(
            pygame.image.load(ghost_paths[ghost_type][0]).convert_alpha(),
            (CELL_SIZE, CELL_SIZE),
        )
        self.rect = self.image.get_rect(topleft=(x, y))
        self.speed = 2
        self.path = []
        self.ghost_type = ghost_type

        self.behavior_strategy = random.choice(["chase", "ambush", "random_wander"])

        if self.behavior_strategy == "random_wander":
            self.random_target_timer = 0
            self.random_target_interval = fps * random.randint(2, 5)
            self.current_random_goal = None

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def get_grid_pos(self):
        return (
            round(self.rect.centerx / CELL_SIZE),
            round((self.rect.centery - SCORE_AREA_HEIGHT) / CELL_SIZE),
        )

    def bfs(self, start, goal, walls):
        wall_positions = {
            (w.rect.x // CELL_SIZE, (w.rect.y - SCORE_AREA_HEIGHT) // CELL_SIZE)
            for w in walls
        }
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == goal:
                return path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < GRID_WIDTH
                    and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in wall_positions
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return []

    def update(self, walls, pacman, fps):
        pacman_grid_x, pacman_grid_y = (
            pacman.rect.centerx // CELL_SIZE,
            (pacman.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE,
        )
        current_grid_pos = self.get_grid_pos()
        wall_positions = {
            (w.rect.x // CELL_SIZE, (w.rect.y - SCORE_AREA_HEIGHT) // CELL_SIZE)
            for w in walls
        }

        goal = (pacman_grid_x, pacman_grid_y)

        if self.behavior_strategy == "ambush":
            if pacman.direction.y == -1:
                target_x = pacman_grid_x - 4
                target_y = pacman_grid_y - 4
            else:
                target_x = pacman_grid_x + int(pacman.direction.x * 4)
                target_y = pacman_grid_y + int(pacman.direction.y * 4)

            target_x = max(0, min(GRID_WIDTH - 1, target_x))
            target_y = max(0, min(GRID_HEIGHT - 1, target_y))

            temp_goal = (target_x, target_y)
            if temp_goal in wall_positions:
                goal = (pacman_grid_x, pacman_grid_y)
            else:
                goal = temp_goal

        elif self.behavior_strategy == "random_wander":
            self.random_target_timer += 1
            if (
                self.random_target_timer >= self.random_target_interval
                or self.current_random_goal is None
                or (
                    self.path
                    and len(self.path) == 1
                    and current_grid_pos == self.current_random_goal
                )
            ):
                new_target_found = False
                while not new_target_found:
                    rand_x = random.randint(0, GRID_WIDTH - 1)
                    rand_y = random.randint(0, GRID_HEIGHT - 1)
                    if (rand_x, rand_y) not in wall_positions:
                        self.current_random_goal = (rand_x, rand_y)
                        new_target_found = True
                self.random_target_timer = 0
            goal = self.current_random_goal

        path_segment_completed = False
        if self.path:
            next_cell_grid = self.path[0]
            target_pos_pixel = (
                next_cell_grid[0] * CELL_SIZE + CELL_SIZE // 2,
                next_cell_grid[1] * CELL_SIZE + CELL_SIZE // 2 + SCORE_AREA_HEIGHT,
            )
            distance_to_next_cell_center = pygame.Vector2(self.rect.center).distance_to(
                target_pos_pixel
            )

            if distance_to_next_cell_center < self.speed:
                self.rect.center = target_pos_pixel
                self.path.pop(0)
                path_segment_completed = True

        if not self.path:
            self.path = self.bfs(current_grid_pos, goal, walls)

        if self.path:
            next_cell_grid = self.path[0]
            target_pos_pixel = (
                next_cell_grid[0] * CELL_SIZE + CELL_SIZE // 2,
                next_cell_grid[1] * CELL_SIZE + CELL_SIZE // 2 + SCORE_AREA_HEIGHT,
            )

            move_vector = pygame.Vector2(target_pos_pixel) - pygame.Vector2(
                self.rect.center
            )
            distance = move_vector.length()

            if distance > 0:
                if distance > self.speed:
                    move_vector.normalize_ip()
                    move_vector *= self.speed
                else:
                    move_vector = move_vector

                self.rect.x += move_vector.x
                self.rect.y += move_vector.y
