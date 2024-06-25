# trading_env.py
import gym
from gym import spaces
import numpy as np
from collections import deque

class TradingEnv(gym.Env):
    def __init__(self, initial_balance=10000, max_steps=1000):
        super().__init__()
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.data = deque(maxlen=self.max_steps)
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        current_price = self.data[-1] if self.data else 100  # Default price if no data

        if action == 0:  # Buy
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += current_price

        done = self.current_step >= self.max_steps
        reward = self._calculate_reward()
        info = {"balance": self.balance, "position": self.position}
        
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.array([
            self.balance,
            self.position,
            self.data[-1] if self.data else 100,  # Current price
            np.mean(self.data) if self.data else 100,  # Average price
            np.std(self.data) if len(self.data) > 1 else 0  # Price volatility
        ], dtype=np.float32)

    def _calculate_reward(self):
        return self.balance + self.position * self.data[-1] if self.data else self.balance

    def update_data(self, price):
        self.data.append(price)