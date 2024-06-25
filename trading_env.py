# trading_env.py
import gym
from gym import spaces
import numpy as np
import pygame
from config import WIDTH, HEIGHT
import asyncio

class BreakoutEnv(gym.Env):
    def __init__(self, data_fetcher, width=1200, height=800):
        super().__init__()
        self.width = width
        self.height = height
        self.data_fetcher = data_fetcher
        self.paddle_width = 100
        self.paddle_height = 10
        self.ball_radius = 8
        self.block_width = 60
        self.block_height = 20
        
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.long_position = 0
        self.short_position = 0
        self.entry_price_long = 0
        self.entry_price_short = 0
        self.position_size_long = 0
        self.position_size_short = 0
        self.orders = []
        self.trade_history = []
        
        self.action_space = spaces.Discrete(3)  # Left, Right, Stay
        obs_size = len(self.data_fetcher.price_data) + len(self.data_fetcher.liquidation_levels)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.reset()

    def _get_observation(self):
        obs = np.array(list(self.data_fetcher.price_data) + self.data_fetcher.liquidation_levels, dtype=np.float32)
        return obs

    def reset(self):
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height - 50
        self.ball_dx = 5 * (np.random.random() * 2 - 1)
        self.ball_dy = -5
        self.score = 0
        self.blocks = self._create_blocks()
        self.balance = self.initial_balance
        self.long_position = 0
        self.short_position = 0
        self.entry_price_long = 0
        self.entry_price_short = 0
        self.position_size_long = 0
        self.position_size_short = 0
        self.orders = []
        self.trade_history = []
        obs = self._get_observation()
        return obs

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
            
            # Calculate color based on price (green for higher prices, red for lower)
            color_value = int(255 * normalized_y)
            color = (255 - color_value, color_value, 0)  # (R, G, B)
            
            blocks.append({'rect': pygame.Rect(i * (self.block_width + 5), y, self.block_width, self.block_height),
                        'color': color})
        
        return blocks

    def _update_blocks(self):
        self.blocks = self._create_blocks()

    async def step(self, action):
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
            if block['rect'].collidepoint(self.ball_x, self.ball_y):
                self.blocks.remove(block)
                self.ball_dy *= -1
                self.score += 1
        
        # Trading logic
        current_price = self.data_fetcher.price_data[-1]
        
        if action == 0:  # Buy/Long
            self._open_long_position(current_price)
        elif action == 2:  # Sell/Short
            self._open_short_position(current_price)

        # Fetch new liquidation levels and update blocks in real-time
        await self.data_fetcher.fetch_liquidation_levels()
        self._update_blocks()

        # Update balance based on unrealized PNL
        unrealized_pnl_long = self._calculate_unrealized_pnl(self.long_position, self.entry_price_long, current_price)
        unrealized_pnl_short = self._calculate_unrealized_pnl(self.short_position, self.entry_price_short, current_price)
        unrealized_pnl = unrealized_pnl_long + unrealized_pnl_short
        self.balance = (self.initial_balance + sum(trade['pnl'] for trade in self.trade_history) + 
                        unrealized_pnl_long + unrealized_pnl_short)

        # Ensure the ball does not fall below the paddle
        if self.ball_y > self.height - self.paddle_height - self.ball_radius:
            self.ball_y = self.height - self.paddle_height - self.ball_radius
            self.ball_dy *= -1
        
        # Check for game over
        done = len(self.blocks) == 0
        reward = 1 if len(self.blocks) == 0 else 0
        
        info = {
            "score": self.score,
            "balance": self.balance,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_long": unrealized_pnl_long,
            "unrealized_pnl_short": unrealized_pnl_short,
            "position_long": self.long_position,
            "position_short": self.short_position,
            "position_size_long": abs(self.position_size_long),
            "position_size_short": abs(self.position_size_short),
            "entry_price_long": self.entry_price_long,
            "entry_price_short": self.entry_price_short,
            "current_price": current_price,
            "orders": self.orders,
            "trade_history": self.trade_history
        }
        
        obs = self._get_observation()
        return obs, reward, done, info
    
    def _open_long_position(self, price):
        self.long_position += 1
        self.entry_price_long = price
        self.position_size_long = self.balance * 0.1 / price  # Use 10% of balance for each trade
        order = {
            "type": "MARKET",
            "side": "BUY",
            "price": price,
            "size": self.position_size_long
        }
        self.orders.append(order)

    def _open_short_position(self, price):
        self.short_position -= 1
        self.entry_price_short = price
        self.position_size_short = self.balance * 0.1 / price  # Use 10% of balance for each trade
        order = {
            "type": "MARKET",
            "side": "SELL",
            "price": price,
            "size": self.position_size_short
        }
        self.orders.append(order)

    def _calculate_unrealized_pnl(self, position, entry_price, current_price):
        if position == 0:
            return 0
        return position * abs(self.balance * 0.1 / entry_price) * (current_price - entry_price)

    def get_render_array(self):
        screen = pygame.Surface((self.width, self.height))
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (self.paddle_x, self.height - self.paddle_height, self.paddle_width, self.paddle_height))
        pygame.draw.circle(screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        for block in self.blocks:
            pygame.draw.rect(screen, block['color'], block['rect'])
        return np.array(pygame.surfarray.array3d(screen))

    def render(self, mode='human'):
        if mode == 'human':
            screen = pygame.display.get_surface()
            if screen is None:
                pygame.init()
                screen = pygame.display.set_mode((self.width, self.height))
            
            render_array = self.get_render_array()
            pygame.surfarray.blit_array(screen, render_array)
            
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
            balance_text = font.render(f"Balance: ${self.balance:.2f}", True, (255, 255, 255))
            pnl_text = font.render(f"P/L: ${self.balance - self.initial_balance:.2f}", True, (255, 255, 255))
            position_long_text = font.render(f"Long Position: {self.long_position}", True, (255, 255, 255))
            position_short_text = font.render(f"Short Position: {self.short_position}", True, (255, 255, 255))
            position_size_long_text = font.render(f"Long Position Size: {self.position_size_long:.4f}", True, (255, 255, 255))
            position_size_short_text = font.render(f"Short Position Size: {self.position_size_short:.4f}", True, (255, 255, 255))
            entry_price_long_text = font.render(f"Long Entry Price: ${self.entry_price_long:.2f}", True, (255, 255, 255))
            entry_price_short_text = font.render(f"Short Entry Price: ${self.entry_price_short:.2f}", True, (255, 255, 255))
            current_price_text = font.render(f"Current Price: ${self.data_fetcher.price_data[-1]:.2f}", True, (255, 255, 255))
        
            screen.blit(score_text, (10, 10))
            screen.blit(balance_text, (10, 50))
            screen.blit(pnl_text, (10, 90))
            screen.blit(position_long_text, (10, 130))
            screen.blit(position_short_text, (10, 170))
            screen.blit(position_size_long_text, (10, 210))
            screen.blit(position_size_short_text, (10, 250))
            screen.blit(entry_price_long_text, (10, 290))
            screen.blit(entry_price_short_text, (10, 330))
            screen.blit(current_price_text, (10, 370))
            
            # Display last 5 trades
            for i, trade in enumerate(self.trade_history[-5:]):
                trade_text = font.render(f"Trade {i+1}: Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}, PNL: ${trade['pnl']:.2f}", True, (255, 255, 255))
                screen.blit(trade_text, (10, 410 + i * 40))
            
            pygame.display.flip()





