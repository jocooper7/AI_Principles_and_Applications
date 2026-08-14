import pygame
from constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    TITLE,
    SKY_BLUE,
    UI_FONT_SIZE,
    UI_BIG_FONT_SIZE,
    UI_MARGIN,
    UI_TEXT_COLOR,
    LEVEL_TIME_LIMIT_SECONDS,
    INTRO_MESSAGE_DURATION_MS,
    INTRO_MESSAGE_TEXT,
    GOOD_JOB_TEXT,
    TIME_UP_TEXT,
    RETRY_PROMPT_TEXT,
    EXIT_PROMPT_TEXT,
)
from player import Player
from levels import load_level_1


# Game states
STATE_INTRO = "intro"
STATE_PLAYING = "playing"
STATE_WON = "won"
STATE_LOST = "lost"


def draw_centered_lines(screen, font, lines, color, center_y, line_spacing=10):
    """Draws a list of text lines centered horizontally, stacked
    vertically starting at center_y."""
    surfaces = [font.render(line, True, color) for line in lines]
    total_height = sum(s.get_height() for s in surfaces) + line_spacing * (
        len(surfaces) - 1
    )
    y = center_y - total_height // 2
    for surface in surfaces:
        rect = surface.get_rect(centerx=SCREEN_WIDTH // 2, top=y)
        screen.blit(surface, rect)
        y += surface.get_height() + line_spacing


def new_level():
    """Creates a fresh set of level entities."""
    platforms, hazards, crown = load_level_1()
    player = Player(100, 100)
    return platforms, hazards, crown, player


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, UI_FONT_SIZE)
    big_font = pygame.font.Font(None, UI_BIG_FONT_SIZE)

    total_score = 0

    platforms, hazards, crown, player = new_level()

    state = STATE_INTRO
    intro_started_at = pygame.time.get_ticks()

    level_start_ticks = pygame.time.get_ticks()
    time_remaining = LEVEL_TIME_LIMIT_SECONDS
    level_score = 0

    running = True
    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if state == STATE_PLAYING:
                    if event.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
                        player.jump()
                elif state == STATE_LOST:
                    if event.key == pygame.K_r:
                        # Retry: reset the level but keep total_score as-is.
                        platforms, hazards, crown, player = new_level()
                        level_start_ticks = now
                        time_remaining = LEVEL_TIME_LIMIT_SECONDS
                        state = STATE_INTRO
                        intro_started_at = now
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif state == STATE_WON:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        if state == STATE_INTRO:
            if now - intro_started_at >= INTRO_MESSAGE_DURATION_MS:
                state = STATE_PLAYING
                level_start_ticks = now

        elif state == STATE_PLAYING:
            elapsed_seconds = (now - level_start_ticks) // 1000
            time_remaining = max(0, LEVEL_TIME_LIMIT_SECONDS - elapsed_seconds)

            player.update(platforms)
            hazards.update()

            hazard_hits = pygame.sprite.spritecollide(player, hazards, False)
            if hazard_hits:
                player.take_damage(hazard_hits[0].rect)

            if player.rect.colliderect(crown.rect):
                level_score = time_remaining
                total_score += level_score
                state = STATE_WON

            elif time_remaining <= 0:
                state = STATE_LOST

        # ------------------------------------------------------------
        # Drawing
        # ------------------------------------------------------------
        screen.fill(SKY_BLUE)

        for platform in platforms:
            screen.blit(platform.image, platform.rect)
        for hazard in hazards:
            screen.blit(hazard.image, hazard.rect)
        screen.blit(crown.image, crown.rect)

        if state in (STATE_INTRO, STATE_PLAYING):
            if player.is_visible():
                screen.blit(player.image, player.rect)

        # Top-left: remaining life
        life_surface = font.render(
            f"Remaining Life: {player.life}", True, UI_TEXT_COLOR
        )
        screen.blit(life_surface, (UI_MARGIN, UI_MARGIN))

        # Top-center: timer, shown as ## seconds remaining
        timer_surface = font.render(f"{time_remaining:02d}", True, UI_TEXT_COLOR)
        timer_rect = timer_surface.get_rect(centerx=SCREEN_WIDTH // 2, top=UI_MARGIN)
        screen.blit(timer_surface, timer_rect)

        # Top-right: total score
        score_surface = font.render(f"Total Score: {total_score}", True, UI_TEXT_COLOR)
        score_rect = score_surface.get_rect(
            right=SCREEN_WIDTH - UI_MARGIN, top=UI_MARGIN
        )
        screen.blit(score_surface, score_rect)

        if state == STATE_INTRO:
            draw_centered_lines(
                screen,
                big_font,
                [INTRO_MESSAGE_TEXT],
                UI_TEXT_COLOR,
                SCREEN_HEIGHT // 2,
            )

        elif state == STATE_WON:
            draw_centered_lines(
                screen,
                big_font,
                [
                    GOOD_JOB_TEXT,
                    f"Level Score: {level_score}",
                    f"Total Score: {total_score}",
                    EXIT_PROMPT_TEXT,
                ],
                UI_TEXT_COLOR,
                SCREEN_HEIGHT // 2,
            )

        elif state == STATE_LOST:
            draw_centered_lines(
                screen,
                big_font,
                [TIME_UP_TEXT, RETRY_PROMPT_TEXT, EXIT_PROMPT_TEXT],
                UI_TEXT_COLOR,
                SCREEN_HEIGHT // 2,
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
