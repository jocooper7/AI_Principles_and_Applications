import pygame
from constants import (
    PLAYER_WIDTH,
    PLAYER_HEIGHT,
    PLAYER_SPEED,
    JUMP_STRENGTH,
    GRAVITY,
    MAX_FALL_SPEED,
    MAX_LIFE,
    STARTING_LIFE,
    INVINCIBILITY_DURATION_MS,
    BLINK_INTERVAL_MS,
    KNOCKBACK_X,
    KNOCKBACK_Y,
    BLUE,
)


class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect(topleft=(x, y))

        # Remember the starting location so the player can be
        # returned here after running out of life points.
        self.start_x = x
        self.start_y = y

        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False

        self.max_life = MAX_LIFE
        self.life = STARTING_LIFE

        # Timestamp (ms, via pygame.time.get_ticks()) until which the
        # player is immune to further hazard damage.
        self.invincible_until = 0

        # Brief knockback impulse applied after taking hazard damage.
        # While active, this overrides normal horizontal input handling.
        self.knockback_until = 0

    def handle_input(self):
        keys = pygame.key.get_pressed()
        self.vel_x = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.vel_x = -PLAYER_SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.vel_x = PLAYER_SPEED

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_STRENGTH
            self.on_ground = False

    def apply_gravity(self):
        self.vel_y += GRAVITY
        if self.vel_y > MAX_FALL_SPEED:
            self.vel_y = MAX_FALL_SPEED

    def is_invincible(self):
        return pygame.time.get_ticks() < self.invincible_until

    def is_knocked_back(self):
        return pygame.time.get_ticks() < self.knockback_until

    def is_visible(self):
        """Returns False during alternating blink frames while the
        player is invincible, to visually signal the invincibility
        window. Always True otherwise."""
        if not self.is_invincible():
            return True
        return (pygame.time.get_ticks() // BLINK_INTERVAL_MS) % 2 == 0

    def take_damage(self, hazard_rect, amount=1):
        """Apply hazard damage, respecting the invincibility window,
        and apply a small knockback away from the hazard.
        Returns True if damage was actually applied."""
        if self.is_invincible():
            return False

        self.lose_life(amount)
        self.invincible_until = pygame.time.get_ticks() + INVINCIBILITY_DURATION_MS

        # Knock the player horizontally away from the hazard, and
        # give a small upward pop for a satisfying bounce-back feel.
        direction = -1 if self.rect.centerx < hazard_rect.centerx else 1
        self.vel_x = KNOCKBACK_X * direction
        self.vel_y = KNOCKBACK_Y
        self.knockback_until = pygame.time.get_ticks() + 200

        return True

    def lose_life(self, amount=1):
        """Reduce the player's life points. If life reaches 0,
        reset it to the starting value and respawn the player
        at their starting location."""
        self.life -= amount
        if self.life <= 0:
            self.life = self.max_life
            self.respawn()

    def respawn(self):
        """Return the player to their starting location and
        reset velocity/ground state."""
        self.rect.topleft = (self.start_x, self.start_y)
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.knockback_until = 0

    def update(self, platforms):
        if self.is_knocked_back():
            # Let the knockback impulse play out without overriding
            # horizontal velocity from normal input this frame.
            self.apply_gravity()
        else:
            self.handle_input()
            self.apply_gravity()

        # Horizontal movement + collision
        self.rect.x += self.vel_x
        self._collide(platforms, dx=self.vel_x, dy=0)

        # Vertical movement + collision
        self.rect.y += self.vel_y
        self.on_ground = False
        self._collide(platforms, dx=0, dy=self.vel_y)

    def _collide(self, platforms, dx, dy):
        for platform in pygame.sprite.spritecollide(self, platforms, False):
            if dx > 0:
                self.rect.right = platform.rect.left
            if dx < 0:
                self.rect.left = platform.rect.right
            if dy > 0:
                self.rect.bottom = platform.rect.top
                self.vel_y = 0
                self.on_ground = True
            if dy < 0:
                self.rect.top = platform.rect.bottom
                self.vel_y = 0
