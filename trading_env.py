# trading_env.py
import gym
import numpy as np
import pygame
import logging
from typing import Tuple, Dict, Any
from model import AsyncFinancialDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data_fetcher: AsyncFinancialDataFetcher, width: int = 800, height: int = 600):
        super().__init__()
        self.data_fetcher = data_fetcher
        self.width = width
        self.height = height
        
        # Game objects
        self.paddle_width = 100
        self.paddle_height = 10
        self.paddle_x = width // 2 - self.paddle_width // 2
        self.ball_radius = 5
        self.ball_x = width // 2
        self.ball_y = height - 50
        
        # Trading state
        self.balance = 10000
        self.position = 0
        self.open_orders = []
        self.last_action = "None"
        self.trade_history = []
        self.open_orders = []
        
        # Block setup
        self.blocks = self._initialize_blocks()
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

    def _initialize_blocks(self):
        blocks = []
        price_range = np.linspace(self.data_fetcher.current_price * 0.9, 
                                  self.data_fetcher.current_price * 1.1, 
                                  num=50)
        for i, price in enumerate(price_range):
            block = {
                'price': price,
                'rect': pygame.Rect(i * 16, 0, 15, 20),
                'color': self._get_block_color(price),
                'status': 'active'
            }
            blocks.append(block)
        return blocks

    def _get_block_color(self, price: float) -> Tuple[int, int, int]:
        if price in self.data_fetcher.liquidation_levels:
            return (255, 0, 0)  # Red for liquidation levels
        return (0, 255, 0)  # Green for normal levels

    def reset(self) -> np.ndarray:
        self.balance = 10000
        self.position = 0
        self.open_orders = []
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_y = self.height - 50
        self.blocks = self._initialize_blocks()
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Move the ball based on price change
        if len(self.data_fetcher.price_history) > 1:
            price_change = self.data_fetcher.current_price - self.data_fetcher.price_history[-2]
            self.ball_y -= int(price_change * 100)  # Scale factor may need adjustment
            self.ball_y = max(0, min(self.ball_y, self.height))
        else:
            # If there's not enough price history, use a small random movement
            self.ball_y += np.random.randint(-5, 6)
            self.ball_y = max(0, min(self.ball_y, self.height))

        # Handle trader action
        reward = self._handle_action(action)

        # Check for block collisions
        self._check_block_collisions()

        # Update game state
        done = self.balance <= 0 or self.ball_y >= self.height

        return self._get_observation(), reward, done, {}
    
    def calculate_pnl(self):
        if self.position == 0:
            return 0
        current_value = self.position * self.data_fetcher.current_price
        cost_basis = sum(trade['price'] * trade['amount'] for trade in self.trade_history if trade['action'] == 'BUY')
        return current_value - cost_basis

    def _handle_action(self, action: int) -> float:
        if action == 1:  # Buy
            if self.balance > 0:
                amount = min(self.balance, self.data_fetcher.current_price)
                self.position += amount / self.data_fetcher.current_price
                self.balance -= amount
                self._place_order('BUY', amount)
                self.last_action = "BUY"
        elif action == 2:  # Sell
            if self.position > 0:
                amount = self.position * self.data_fetcher.current_price
                self.balance += amount
                self._place_order('SELL', amount)
                self.position = 0
                self.last_action = "SELL"
        else:
            self.last_action = "HOLD"

        return self._calculate_reward()

    def _place_order(self, order_type: str, amount: float):
        order = {
            'type': order_type,
            'price': self.data_fetcher.current_price,
            'amount': amount,
            'pnl': 0  # This will be updated when the order is executed
        }
        self.open_orders.append(order)
        self.trade_history.append(order)

    def _check_block_collisions(self):
        for block in self.blocks:
            if block['rect'].collidepoint(self.ball_x, self.ball_y):
                if block['status'] == 'active':
                    block['status'] = 'hit'
                    block['color'] = (255, 165, 0)  # Orange for hit blocks
                elif block['status'] == 'hit' and self.open_orders:
                    self._execute_trade(block)

    def _execute_trade(self, block):
        for order in self.open_orders:
            pnl = (block['price'] - order['price']) * order['amount']
            if pnl > 0:
                self.balance += pnl
                block['status'] = 'destroyed'
                block['color'] = (0, 0, 255)  # Blue for destroyed blocks
            self.open_orders.remove(order)

    def _calculate_reward(self) -> float:
        portfolio_value = self.balance + self.position * self.data_fetcher.current_price
        return portfolio_value - 10000  # Reward is change in portfolio value

    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.balance,
            self.position,
            self.data_fetcher.current_price,
            self.data_fetcher.volume,
            self.data_fetcher.adx,
            len(self.open_orders)
        ], dtype=np.float32)

    def render(self, mode='human'):
        if mode == 'human':
            # Implement PyGame rendering here
            pass
        elif mode == 'rgb_array':
            # Implement rendering to RGB array here
            pass

    def close(self):
        pygame.quit()