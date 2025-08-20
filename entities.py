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

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class Pellet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (CELL_SIZE // 2, CELL_SIZE // 2), 4)
        self.rect = self.image.get_rect(topleft=(x, y))

    def draw(self, surface):
        surface.blit(self.image, self.rect)


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
        if keys:  # keys is an empty dict from the agent.
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
            pygame.image.load(
                ghost_paths[list(ghost_paths.keys())[0]][0]
            ).convert_alpha(),
            (CELL_SIZE, CELL_SIZE),
        )
        self.rect = self.image.get_rect(topleft=(x, y))
        self.pos = pygame.Vector2(self.rect.center)  # subpixel position
        self.initial_pos = (x, y)
        self.speed = 2
        self.path = []
        self.ghost_type = ghost_type
        self.fps = fps
        self.behavior_strategy = random.choice(["chase", "ambush", "random_wander"])

        if self.behavior_strategy == "random_wander":
            self.random_target_timer = 0
            self.random_target_interval = fps * random.randint(2, 5)
            self.current_random_goal = None

    def reset(self):
        self.rect.topleft = self.initial_pos
        self.pos = pygame.Vector2(self.rect.center)  # keep in sync
        self.path = []
        self.behavior_strategy = random.choice(["chase", "ambush", "random_wander"])

        if self.behavior_strategy == "random_wander":
            self.random_target_timer = 0
            self.random_target_interval = self.fps * random.randint(2, 5)
            self.current_random_goal = None

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def get_grid_pos(self):
        return (
            int(self.rect.centerx // CELL_SIZE),
            int((self.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE),
        )

    def bfs(self, start, goal, walls):
        wall_positions = {
            (w.rect.x // CELL_SIZE, (w.rect.y - SCORE_AREA_HEIGHT) // CELL_SIZE)
            for w in walls.sprites()
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
        pacman_grid_x = int(pacman.rect.centerx // CELL_SIZE)
        pacman_grid_y = int((pacman.rect.centery - SCORE_AREA_HEIGHT) // CELL_SIZE)
        current_grid_pos = self.get_grid_pos()

        wall_positions = {
            (w.rect.x // CELL_SIZE, (w.rect.y - SCORE_AREA_HEIGHT) // CELL_SIZE)
            for w in walls.sprites()
        }

        goal = (pacman_grid_x, pacman_grid_y)

        if self.behavior_strategy == "ambush":
            tx = pacman_grid_x + int(getattr(pacman.direction, "x", 0) * 4)
            ty = pacman_grid_y + int(getattr(pacman.direction, "y", 0) * 4)
            tx = max(0, min(GRID_WIDTH - 1, tx))
            ty = max(0, min(GRID_HEIGHT - 1, ty))
            if (tx, ty) not in wall_positions:
                goal = (tx, ty)

        elif self.behavior_strategy == "random_wander":
            if not hasattr(self, "random_target_timer"):
                self.random_target_timer = 0
                self.random_target_interval = fps * random.randint(2, 5)
                self.current_random_goal = None
            self.random_target_timer += 1
            # Check if a new random goal is needed
            need_new = (
                self.random_target_timer >= self.random_target_interval
                or self.current_random_goal is None
                or current_grid_pos == self.current_random_goal
            )
            if need_new:
                # Find a new valid random goal
                for _ in range(200):
                    rx = random.randint(0, GRID_WIDTH - 1)
                    ry = random.randint(0, GRID_HEIGHT - 1)
                    if (rx, ry) in wall_positions:
                        continue
                    pth = self.bfs(current_grid_pos, (rx, ry), walls)
                    if pth:
                        self.current_random_goal = (rx, ry)
                        break
                self.random_target_timer = 0
            if self.current_random_goal:
                goal = self.current_random_goal

        # Recompute path only if needed (e.g., path is empty or a new goal is set)
        if not self.path or current_grid_pos == goal:
            self.path = self.bfs(current_grid_pos, goal, walls)
            if not self.path:  # If no path is found, pick a random direction to move

                # Logic to move in a random direction and avoid walls
                valid_moves = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = current_grid_pos[0] + dx, current_grid_pos[1] + dy
                    if (nx, ny) not in wall_positions:
                        valid_moves.append((nx, ny))
                if valid_moves:
                    self.path.append(random.choice(valid_moves))
                else:
                    return

        # If still no path, something is wrong, so stop
        if not self.path:
            return

        next_cell = self.path[0]
        target_px = (
            next_cell[0] * CELL_SIZE + CELL_SIZE // 2,
            next_cell[1] * CELL_SIZE + CELL_SIZE // 2 + SCORE_AREA_HEIGHT,
        )

        target_vec = pygame.Vector2(target_px)
        move_vec = target_vec - self.pos
        dist = move_vec.length()

        # This section ensures that once a cell is reached, the ghost moves to the next cell in the path.
        if dist < self.speed:
            self.pos = pygame.Vector2(target_px)
            self.rect.center = (int(self.pos.x), int(self.pos.y))
            self.path.pop(0)
            if not self.path:
                return  # Stop if the path is complete
        else:
            if dist > 0:
                move_vec.scale_to_length(self.speed)
            self.pos += move_vec
            self.rect.center = (int(self.pos.x), int(self.pos.y))
