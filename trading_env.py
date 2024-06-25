# trading_env.py
import gym
from gym import spaces
import numpy as np
import pygame
import asyncio

import gym
from gym import spaces
import numpy as np
import pygame
import asyncio


class BreakoutEnv(gym.Env):
    def __init__(self, data_fetcher, width=1200, height=800):
        super().__init__()
        self.width = width
        self.height = height
        self.data_fetcher = data_fetcher

        # Game objects
        self.paddle_width = 100
        self.paddle_height = 10
        self.ball_radius = 8
        self.block_width = 60
        self.block_height = 20
        self.ball_speed = 5

        # Trading variables
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

        # Gym spaces
        self.action_space = spaces.Discrete(3)  # Left, Right, Stay
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data_fetcher.max_data_points + 2,),
            dtype=np.float32
        )

        # Initialize game state
        self.blocks = []
        self.score = 0
        self.paddle_x = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_dx = 0
        self.ball_dy = 0

    async def reset(self):
        # Reset game objects
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height - 50
        self.ball_dx = self.ball_speed * (np.random.random() * 2 - 1)
        self.ball_dy = -self.ball_speed
        self.score = 0

        # Reset trading variables
        self.balance = self.initial_balance
        self.long_position = 0
        self.short_position = 0
        self.entry_price_long = 0
        self.entry_price_short = 0
        self.position_size_long = 0
        self.position_size_short = 0
        self.orders = []
        self.trade_history = []

        # Create initial blocks
        self.blocks = await self._create_blocks()

        return await self._get_observation()

    async def _create_blocks(self):
        blocks = []
        state = await self.data_fetcher.get_current_state()
        if state is None:
            return blocks

        price_data = state[:-2].tolist()
        liquidation_levels = state[-2:].tolist()
        all_levels = price_data + liquidation_levels

        if not all_levels:
            return blocks

        min_price, max_price = min(all_levels), max(all_levels)

        for i, price in enumerate(price_data):
            normalized_y = (price - min_price) / (max_price -
                                                  min_price) if max_price > min_price else 0.5
            y = int(normalized_y * (self.height // 2))

            color_value = int(255 * normalized_y)
            color = (255 - color_value, color_value, 0)  # (R, G, B)

            blocks.append({
                'rect': pygame.Rect(i * (self.block_width + 5), y, self.block_width, self.block_height),
                'color': color
            })

        return blocks

    async def _get_observation(self):
        state = await self.data_fetcher.get_current_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return state.numpy()

    async def step(self, action):
        # Move paddle
        if action == 0:  # Left
            self.paddle_x = max(0, self.paddle_x - 10)
        elif action == 2:  # Right
            self.paddle_x = min(
                self.width - self.paddle_width, self.paddle_x + 10)

        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_dx *= -1
        if self.ball_y <= 0:
            self.ball_dy *= -1

        # Ball collision with paddle
        if (self.ball_y >= self.height - self.paddle_height - self.ball_radius and
                self.paddle_x <= self.ball_x <= self.paddle_x + self.paddle_width):
            self.ball_dy *= -1

        # Ball collision with blocks
        ball_rect = pygame.Rect(
            self.ball_x - self.ball_radius,
            self.ball_y - self.ball_radius,
            self.ball_radius * 2,
            self.ball_radius * 2
        )
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                self.blocks.remove(block)
                self.ball_dy *= -1
                self.score += 1
                break  # Exit loop after hitting a block to prevent multiple collisions

        # Trading logic
        state = await self.data_fetcher.get_current_state()
        if state is not None:
            current_price = state[-1].item()  # Get the most recent price

            if action == 0:  # Buy/Long
                self._open_long_position(current_price)
            elif action == 2:  # Sell/Short
                self._open_short_position(current_price)

            # Update blocks
            self.blocks = await self._create_blocks()

            # Update balance based on unrealized PNL
            unrealized_pnl_long = self._calculate_unrealized_pnl(
                self.long_position, self.entry_price_long, current_price)
            unrealized_pnl_short = self._calculate_unrealized_pnl(
                self.short_position, self.entry_price_short, current_price)
            unrealized_pnl = unrealized_pnl_long + unrealized_pnl_short
            self.balance = (self.initial_balance +
                            sum(trade['pnl'] for trade in self.trade_history) +
                            unrealized_pnl)

            # Ensure the ball does not fall below the paddle
            if self.ball_y > self.height - self.paddle_height - self.ball_radius:
                self.ball_y = self.height - self.paddle_height - self.ball_radius
                self.ball_dy *= -1

            # Check for game over
            done = len(self.blocks) == 0 or self.ball_y >= self.height
            reward = 1 if len(self.blocks) == 0 else - \
                1 if self.ball_y >= self.height else 0

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

            obs = await self._get_observation()
            return obs, reward, done, info
        else:
            # Handle the case where we don't have data
            return await self._get_observation(), 0, False, {"score": self.score}

    def _open_long_position(self, price):
        if self.balance > 0:  # Only open a position if we have balance
            self.long_position += 1
            self.entry_price_long = price
            # Use 10% of balance or all remaining balance
            self.position_size_long = min(
                self.balance * 0.1, self.balance) / price
            order = {
                "type": "MARKET",
                "side": "BUY",
                "price": price,
                "size": self.position_size_long
            }
            self.orders.append(order)
            self.balance -= self.position_size_long * price  # Deduct the cost from balance

    def _open_short_position(self, price):
        if self.balance > 0:  # Only open a position if we have balance
            self.short_position -= 1
            self.entry_price_short = price
            # Use 10% of balance or all remaining balance
            self.position_size_short = min(
                self.balance * 0.1, self.balance) / price
            order = {
                "type": "MARKET",
                "side": "SELL",
                "price": price,
                "size": self.position_size_short
            }
            self.orders.append(order)
            # For shorts, we don't deduct from balance immediately

    def _close_positions(self, current_price):
        if self.long_position != 0:
            pnl = self._calculate_unrealized_pnl(
                self.long_position, self.entry_price_long, current_price)
            self.balance += pnl
            self.trade_history.append({
                "entry_price": self.entry_price_long,
                "exit_price": current_price,
                "pnl": pnl
            })
            self.long_position = 0
            self.entry_price_long = 0
            self.position_size_long = 0

        if self.short_position != 0:
            pnl = self._calculate_unrealized_pnl(
                self.short_position, self.entry_price_short, current_price)
            self.balance += pnl
            self.trade_history.append({
                "entry_price": self.entry_price_short,
                "exit_price": current_price,
                "pnl": pnl
            })
            self.short_position = 0
            self.entry_price_short = 0
            self.position_size_short = 0

    def _calculate_unrealized_pnl(self, position, entry_price, current_price):
        if position == 0 or entry_price == 0:
            return 0
        position_value = abs(position * self.balance * 0.1)
        if position > 0:  # Long position
            return position * (current_price - entry_price)
        else:  # Short position
            return position * (entry_price - current_price)

    def get_render_array(self):
        screen = pygame.Surface((self.width, self.height))
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (self.paddle_x, self.height -
                         self.paddle_height, self.paddle_width, self.paddle_height))
        pygame.draw.circle(screen, (255, 255, 255), (int(
            self.ball_x), int(self.ball_y)), self.ball_radius)
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
            texts = [
                f"Score: {self.score}",
                f"Balance: ${self.balance:.2f}",
                f"P/L: ${self.balance - self.initial_balance:.2f}",
                f"Long Position: {self.long_position}",
                f"Short Position: {self.short_position}",
                f"Long Position Size: {self.position_size_long:.4f}",
                f"Short Position Size: {self.position_size_short:.4f}",
                f"Long Entry Price: ${self.entry_price_long:.2f}",
                f"Short Entry Price: ${self.entry_price_short:.2f}",
                f"Current Price: ${
                    self.data_fetcher.price_data[-1] if self.data_fetcher.price_data else 0:.2f}"
            ]

            for i, text in enumerate(texts):
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10 + i * 40))

            # Display last 5 trades
            for i, trade in enumerate(self.trade_history[-5:]):
                trade_text = font.render(
                    f"Trade {i+1}: Entry: ${trade['entry_price']:.2f}, "
                    f"Exit: ${trade['exit_price']:.2f}, PNL: ${
                        trade['pnl']:.2f}",
                    True, (255, 255, 255)
                )
                screen.blit(trade_text, (10, 410 + i * 40))

            # Display open positions and orders
            open_positions_text = font.render(
                "Open Positions", True, (255, 255, 255))
            screen.blit(open_positions_text,
                        (self.width - 300, self.height - 200))

            if self.long_position != 0:
                long_text = font.render(
                    f"Long: {self.long_position} @ ${self.entry_price_long:.2f}", True, (0, 255, 0))
                screen.blit(long_text, (self.width - 300, self.height - 160))

            if self.short_position != 0:
                short_text = font.render(f"Short: {abs(
                    self.short_position)} @ ${self.entry_price_short:.2f}", True, (255, 0, 0))
                screen.blit(short_text, (self.width - 300, self.height - 120))

            orders_text = font.render("Open Orders", True, (255, 255, 255))
            screen.blit(orders_text, (self.width - 300, self.height - 80))

            # Display last 3 orders
            for i, order in enumerate(self.orders[-3:]):
                order_text = font.render(f"{order['side']} {
                                         order['size']:.4f} @ ${order['price']:.2f}", True, (255, 255, 0))
                screen.blit(order_text, (self.width - 300,
                            self.height - 40 + i * 40))

            pygame.display.flip()
