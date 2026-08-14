import pygame
from constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    GROUND_HEIGHT,
    PLATFORM_COLOR,
    WALL_WIDTH,
    WALL_COLOR,
    HAZARD_COLOR,
    HAZARD_SIZE,
    HAZARD_SPEED,
    CROWN_SIZE,
    CROWN_COLOR,
)


class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color=PLATFORM_COLOR):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))


class Hazard(pygame.sprite.Sprite):
    """A triangular spike hazard attached to a platform.

    The spike sits on either the top or bottom edge of the platform
    it belongs to, and patrols back and forth across the platform's
    width only (it never travels past the platform's left/right edges).
    """

    def __init__(
        self,
        platform_x,
        platform_y,
        platform_width,
        platform_height,
        facing="up",
        size=HAZARD_SIZE,
        speed=HAZARD_SPEED,
        color=HAZARD_COLOR,
    ):
        super().__init__()
        self.size = size
        self.facing = facing
        self.speed = speed
        self.color = color

        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        self._draw_triangle()

        # Horizontal patrol bounds match the platform's width exactly,
        # so the spike never moves past the platform it's attached to.
        self.min_x = platform_x
        self.max_x = platform_x + platform_width - size

        start_x = platform_x + (platform_width - size) // 2

        if facing == "up":
            # Spike rests on top of the platform, pointing upward.
            start_y = platform_y - size
        else:
            # Spike hangs from the bottom of the platform, pointing downward.
            start_y = platform_y + platform_height

        self.rect = self.image.get_rect(topleft=(start_x, start_y))
        self.direction = 1

    def _draw_triangle(self):
        size = self.size
        if self.facing == "up":
            points = [(0, size), (size, size), (size // 2, 0)]
        else:
            points = [(0, 0), (size, 0), (size // 2, size)]
        pygame.draw.polygon(self.image, self.color, points)

    def update(self):
        self.rect.x += self.speed * self.direction

        if self.rect.x <= self.min_x:
            self.rect.x = self.min_x
            self.direction = 1
        elif self.rect.x >= self.max_x:
            self.rect.x = self.max_x
            self.direction = -1


class Crown(pygame.sprite.Sprite):
    """The end-of-level goal. A small yellow square (half the size of
    the player) representing a crown to be collected."""

    def __init__(self, x, y, size=CROWN_SIZE, color=CROWN_COLOR):
        super().__init__()
        self.image = pygame.Surface((size, size))
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))


def load_level_1():
    """
    Returns a tuple of (platforms, hazards, crown) for level 1.
    Positions are hardcoded for the proof-of-concept stage.
    """
    platforms = pygame.sprite.Group()
    hazards = pygame.sprite.Group()

    # Ground
    ground = Platform(0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT)
    platforms.add(ground)

    # Boundary walls to keep the player from leaving the level.
    wall_left = Platform(0, 0, WALL_WIDTH, SCREEN_HEIGHT, color=WALL_COLOR)
    wall_right = Platform(
        SCREEN_WIDTH - WALL_WIDTH, 0, WALL_WIDTH, SCREEN_HEIGHT, color=WALL_COLOR
    )
    platforms.add(wall_left, wall_right)

    # Floating platforms
    platform_a = Platform(150, 450, 120, 20)
    platform_b = Platform(350, 350, 120, 20)
    platform_c = Platform(550, 250, 120, 20)
    platform_d = Platform(700, 200, 100, 20)
    platforms.add(platform_a, platform_b, platform_c, platform_d)

    # Hazards attached to specific platforms
    hazards.add(Hazard(150, 450, 120, 20, facing="up"))
    hazards.add(Hazard(350, 350, 120, 20, facing="down"))
    hazards.add(Hazard(550, 250, 120, 20, facing="up"))

    # Goal crown placed on the platform to the right of platform_c
    crown_x = platform_d.rect.x + (platform_d.rect.width - CROWN_SIZE) // 2
    crown_y = platform_d.rect.y - CROWN_SIZE
    crown = Crown(crown_x, crown_y)

    return platforms, hazards, crown
