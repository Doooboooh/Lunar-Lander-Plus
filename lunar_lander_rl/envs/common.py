import gymnasium as gym
import numpy as np


VIEWPORT_W = 600
VIEWPORT_H = 400
FPS = 50


def make_base_env(render_mode: str | None = None, continuous: bool = False, **kwargs):
    if render_mode not in (None, "human", "rgb_array"):
        raise ValueError(f"Unsupported render_mode: {render_mode!r}")

    base_render_mode = "rgb_array" if render_mode is not None else None
    return gym.make(
        "LunarLander-v3",
        render_mode=base_render_mode,
        continuous=continuous,
        **kwargs,
    )


def state_to_world(env, x, y):
    unwrapped = env.unwrapped
    world_w = VIEWPORT_W / 30.0
    world_h = VIEWPORT_H / 30.0
    helipad_y = getattr(unwrapped, "helipad_y", world_h / 4.0)
    leg_down = 18.0 / 30.0
    world_x = float(x) * (world_w / 2.0) + (world_w / 2.0)
    world_y = float(y) * (world_h / 2.0) + helipad_y + leg_down
    return world_x, world_y


def state_to_pixel(env, x, y):
    x_px = int((float(x) + 1.0) * VIEWPORT_W / 2.0)
    helipad_y = getattr(env.unwrapped, "helipad_y", VIEWPORT_H / 30.0 / 4.0)
    leg_down = 18.0 / 30.0
    world_h = VIEWPORT_H / 30.0
    world_y = float(y) * (world_h / 2.0) + helipad_y + leg_down
    y_px = int(VIEWPORT_H - world_y * 30.0)
    return x_px, y_px


def draw_rect(frame, x1, y1, x2, y2, color):
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = color


def draw_circle(frame, cx, cy, radius, color):
    h, w = frame.shape[:2]
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius + 1))
    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius + 1))
    if x_max <= x_min or y_max <= y_min:
        return
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    frame[y_min:y_max, x_min:x_max][mask] = color


def draw_circle_outline(frame, cx, cy, radius, color, thickness=3):
    h, w = frame.shape[:2]
    x_min = max(0, int(cx - radius - thickness))
    x_max = min(w, int(cx + radius + thickness + 1))
    y_min = max(0, int(cy - radius - thickness))
    y_max = min(h, int(cy + radius + thickness + 1))
    if x_max <= x_min or y_max <= y_min:
        return
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    outer = (radius + thickness) ** 2
    inner = max(0, radius - thickness) ** 2
    mask = (dist2 <= outer) & (dist2 >= inner)
    frame[y_min:y_max, x_min:x_max][mask] = color


def draw_game_over(frame):
    import pygame

    if not pygame.font.get_init():
        pygame.font.init()

    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    shade = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 90))
    surface.blit(shade, (0, 0))

    font = pygame.font.Font(None, 76)
    text = font.render("GAME OVER", True, (255, 255, 255))
    shadow = font.render("GAME OVER", True, (20, 20, 20))
    rect = text.get_rect(center=(VIEWPORT_W // 2, VIEWPORT_H // 2))
    shadow_rect = shadow.get_rect(center=(VIEWPORT_W // 2 + 3, VIEWPORT_H // 2 + 3))
    surface.blit(shadow, shadow_rect)
    surface.blit(text, rect)

    frame[:] = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def present_frame(owner, frame):
    import pygame

    if owner._screen is None:
        pygame.init()
        pygame.display.init()
        owner._screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    if owner._clock is None:
        owner._clock = pygame.time.Clock()

    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    owner._screen.blit(surf, (0, 0))
    pygame.event.pump()
    owner._clock.tick(owner.metadata["render_fps"])
    pygame.display.flip()


def close_window(owner):
    if owner._screen is not None:
        import pygame

        pygame.display.quit()
        pygame.quit()
        owner._screen = None
        owner._clock = None
