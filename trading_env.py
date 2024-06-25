# trading_env.py
# trading_env.py
import gym
from gym import spaces
import numpy as np
import pygame
from config import WIDTH, HEIGHT

class BreakoutEnv(gym.Env):
    def __init__(self, data_fetcher, width=800, height=600):
        super().__init__()
        self.width = width
        self.height = height
        self.data_fetcher = data_fetcher
        self.paddle_width = 100
        self.paddle_height = 10
        self.ball_radius = 8
        self.block_width = 60
        self.block_height = 20
        
        self.action_space = spaces.Discrete(3)  # Left, Right, Stay
        obs_size = len(self.data_fetcher.price_data) + len(self.data_fetcher.liquidation_levels)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.reset()

    def _get_observation(self):
        return np.array(list(self.data_fetcher.price_data) + self.data_fetcher.liquidation_levels, dtype=np.float32)

    def reset(self):
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height - 50
        self.ball_dx = 5 * (np.random.random() * 2 - 1)
        self.ball_dy = -5
        self.score = 0
        self.blocks = self._create_blocks()
        return self._get_observation()

    def _create_blocks(self):
        blocks = []
        price_data = list(self.data_fetcher.price_data)
        liquidation_levels = self.data_fetcher.liquidation_levels
        all_levels = price_data + liquidation_levels
        
        if not all_levels:
            return blocks  # Return empty list if no data

        min_price, max_price = min(all_levels), max(all_levels)
        
        for i, price in enumerate(price_data):
            normalized_y = (price - min_price) / (max_price - min_price) if max_price > min_price else 0.5
            y = int(normalized_y * (self.height // 2))
            blocks.append(pygame.Rect(i * (self.block_width + 5), y, self.block_width, self.block_height))
        
        return blocks

    def step(self, action):
        # Move paddle
        if action == 0:  # Left
            self.paddle_x = max(0, self.paddle_x - 10)
        elif action == 2:  # Right
            self.paddle_x = min(self.width - self.paddle_width, self.paddle_x + 10)
        
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_dx *= -1
        if self.ball_y <= 0:
            self.ball_dy *= -1
        
        # Ball collision with paddle
        if self.ball_y >= self.height - self.paddle_height - self.ball_radius and \
           self.paddle_x <= self.ball_x <= self.paddle_x + self.paddle_width:
            self.ball_dy *= -1
        
        # Ball collision with blocks
        for block in self.blocks[:]:
            if block.collidepoint(self.ball_x, self.ball_y):
                self.blocks.remove(block)
                self.ball_dy *= -1
                self.score += 1
        
        # Check for game over
        done = self.ball_y > self.height or len(self.blocks) == 0
        reward = 1 if len(self.blocks) == 0 else (-1 if done else 0)
        
        return self._get_observation(), reward, done, {"score": self.score}

    def _get_observation(self):
        screen = pygame.Surface((self.width, self.height))
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (self.paddle_x, self.height - self.paddle_height, self.paddle_width, self.paddle_height))
        pygame.draw.circle(screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        for block in self.blocks:
            pygame.draw.rect(screen, (255, 0, 0), block)
        return np.array(pygame.surfarray.array3d(screen))

    def render(self, mode='human'):
        if mode == 'human':
            screen = pygame.display.get_surface()
            if screen is None:
                pygame.init()
                screen = pygame.display.set_mode((self.width, self.height))
            
            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (255, 255, 255), (self.paddle_x, self.height - self.paddle_height, self.paddle_width, self.paddle_height))
            pygame.draw.circle(screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), self.ball_radius)
            for block in self.blocks:
                pygame.draw.rect(screen, (255, 0, 0), block)
            
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            
            pygame.display.flip()