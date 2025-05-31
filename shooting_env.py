import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import random
import pygame
import sys

class SimpleShootingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, discrete_action=True, num_angles=4, set_ManualMode=False):
        super().__init__()
        self.num_targets = 3
        self.bullet_cooldown = 30  # frames (1 second if 30 fps)
        self.frame = 0
        self.bullet_speed = 5.0
        self.tank_pos = np.array([0.0, 0.0])
        self.discrete_action = discrete_action
        self.manual_mode = set_ManualMode
        self.screen_size = 500
        self.scale = 2  # scale world coordinates to screen
        self.screen = None
        self.clock = None
        self.bullets = []
        self.num_angles = num_angles if discrete_action else None

        if self.discrete_action:
            self.angles = [(360 / self.num_angles) * i for i in range(self.num_angles)]
            self.action_space = spaces.Discrete(len(self.angles))
        else:
            self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([360.0]), dtype=np.float32)

        low = np.array([-100.0, -100.0, 0.0, 0.0, 0.0] * self.num_targets, dtype=np.float32)
        high = np.array([100.0, 100.0, 360.0, 200.0, 10.0] * self.num_targets, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.render_mode = "human"
        self.targets = []
        self.turret_angle = 0.0
        self.cooldown_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.targets = [self._spawn_target() for _ in range(self.num_targets)]
        self.turret_angle = 0.0
        self.cooldown_counter = 0
        self.frame = 0
        self.bullets = []
        if self.manual_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Simple Shooting Environment")
            self.clock = pygame.time.Clock()
        return self._get_obs(), {}

    def _spawn_target(self):
        radius = random.uniform(40.0, 80.0)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.02, 0.08)
        direction = random.choice([-1, 1])  # -1: CCW, 1: CW
        return {"radius": radius, "theta": angle, "speed": speed, "direction": direction}

    def _spawn_bullet(self):
        angle_rad = math.radians(self.turret_angle)
        bullet = {
            "x": 0.0,
            "y": 0.0,
            "dx": math.cos(angle_rad) * self.bullet_speed,
            "dy": math.sin(angle_rad) * self.bullet_speed,
        }
        self.bullets.append(bullet)

    def _update_bullets(self):
        for b in self.bullets:
            b["x"] += b["dx"]
            b["y"] += b["dy"]
        self.bullets = [b for b in self.bullets if abs(b["x"]) < 100 and abs(b["y"]) < 100]

    def _get_obs(self):
        obs = []
        for t in self.targets:
            x = t["radius"] * math.cos(t["theta"])
            y = t["radius"] * math.sin(t["theta"])
            distance = np.linalg.norm([x, y])
            angle = (math.degrees(t["theta"]) + 360) % 360
            obs.extend([x, y, angle, distance, t["speed"]])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.frame += 1
        reward = 0.0
        terminated = False

        if self.discrete_action:
            self.turret_angle = self.angles[action]
        else:
            self.turret_angle = action[0] % 360

        for t in self.targets:
            t["theta"] += t["direction"] * t["speed"]

        self._update_bullets()

        self.cooldown_counter += 1
        if self.cooldown_counter >= self.bullet_cooldown:
            self.cooldown_counter = 0
            self._spawn_bullet()

        reward += self._check_hits()
        obs = self._get_obs()
        if self.manual_mode:
            self.render()
        return obs, reward, terminated, False, {}

    def _check_hits(self):
        hits = 0
        hit_targets = set()
        for i, t in enumerate(self.targets):
            tx = t["radius"] * math.cos(t["theta"])
            ty = t["radius"] * math.sin(t["theta"])
            for b in self.bullets:
                if math.hypot(b["x"] - tx, b["y"] - ty) < 5.0:
                    hits += 1
                    hit_targets.add(i)
                    break
        for i in sorted(hit_targets, reverse=True):
            self.targets[i] = self._spawn_target()
        return hits

    def render(self):
        if not self.manual_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))
        center = self.screen_size // 2

        pygame.draw.circle(self.screen, (0, 0, 0), (center, center), 10)

        dx = math.cos(math.radians(self.turret_angle)) * 40
        dy = math.sin(math.radians(self.turret_angle)) * 40
        pygame.draw.line(self.screen, (0, 0, 255), (center, center), (center + dx, center + dy), 3)

        for t in self.targets:
            x = t["radius"] * math.cos(t["theta"]) * self.scale + center
            y = t["radius"] * math.sin(t["theta"]) * self.scale + center
            color = (255, 0, 0) if t["direction"] == 1 else (255, 128, 0)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)

        for b in self.bullets:
            bx = b["x"] * self.scale + center
            by = b["y"] * self.scale + center
            pygame.draw.circle(self.screen, (0, 128, 0), (int(bx), int(by)), 4)

        pygame.display.flip()
        self.clock.tick(30)

    def manual_step(self):
        if not self.manual_mode:
            raise ValueError("manual_mode must be enabled")

        mouse_x, mouse_y = pygame.mouse.get_pos()
        dx = mouse_x - self.screen_size // 2
        dy = mouse_y - self.screen_size // 2
        self.turret_angle = math.degrees(math.atan2(dy, dx)) % 360

        self.frame += 1
        for t in self.targets:
            t["theta"] += t["direction"] * t["speed"]

        self._update_bullets()

        self.cooldown_counter += 1
        reward = 0
        if self.cooldown_counter >= self.bullet_cooldown:
            self.cooldown_counter = 0
            self._spawn_bullet()

        reward += self._check_hits()
        self.render()
        return self._get_obs(), reward

    def close(self):
        if self.manual_mode:
            pygame.quit()

if __name__ == "__main__":
    env = SimpleShootingEnv(discrete_action=True, num_angles=8, set_ManualMode=True)
    env.reset()

    while True:
        obs, reward = env.manual_step()
        if reward > 0:
            print(f"Hit! Reward: {reward}")
