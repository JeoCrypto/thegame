import pygame
import random
import numpy as np
import gym
from gym import spaces
import websockets
import json
import threading
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import asyncio

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Advanced Trading Bot Simulation')

# Colors
BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)

# Global variables for real-time data
price_data = []
open_interest_data = []

# WebSocket for real-time data
def on_message(ws, message):
    global price_data, open_interest_data
    json_message = json.loads(message)
    price = float(json_message['p'])
    price_data.append(price)
    # Simulate fetching open interest data
    open_interest = float(json_message['q'])  # Just for example
    open_interest_data.append(open_interest)
    if len(price_data) > 1000:
        price_data.pop(0)
        open_interest_data.pop(0)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### WebSocket closed ###")

def on_open(ws):
    print("### WebSocket opened ###")

async def start_websocket():
    socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(socket) as ws:
        while True:
            message = await ws.recv()
            json_message = json.loads(message)
            price = float(json_message['p'])
            price_data.append(price)
            # Simulate fetching open interest data
            open_interest = float(json_message['q'])  # Just for example
            open_interest_data.append(open_interest)
            if len(price_data) > 1000:
                price_data.pop(0)
                open_interest_data.pop(0)

# Start WebSocket in a separate thread
ws_thread = threading.Thread(target=lambda: asyncio.run(start_websocket()))
ws_thread.start()



# MCTS Node and MCTS classes (from previous implementation)
class Node:
    def __init__(self, prior, to_play, state):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = state

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self.ucb_score(child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, child):
        prior_score = child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = -child.value()
        else:
            value_score = 0
        return value_score + prior_score

    def expand(self, model, game, state, to_play):
        action_probs, value = model.predict(state)
        valid_moves = game.get_valid_moves(state)
        action_probs = action_probs * valid_moves
        action_probs /= np.sum(action_probs)
        for a, prob in enumerate(action_probs):
            if prob > 0:
                self.children[a] = Node(prior=prob, to_play=-to_play, state=None)

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, state, to_play, num_simulations=50):
        root = Node(0, to_play, state)
        root.expand(self.model, self.game, state, to_play)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            next_state, _ = self.game.get_next_state(state, to_play, action)
            value = self.game.get_reward_for_player(next_state, to_play)

            if value is None:
                node.expand(self.model, self.game, next_state, -to_play)
                value, _ = self.model.predict(next_state)

            self.backpropagate(search_path, value, to_play)

        return root

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

# Trading Environment
class TradingEnv(gym.Env):
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(1000,), dtype=np.float32)
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.price_history = []

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.price_history = price_data[-1000:]
        return np.array(self.price_history)

    def step(self, action):
        self.current_step += 1
        current_price = price_data[-1]
        self.price_history.append(current_price)
        self.price_history.pop(0)

        if action == 0:  # Buy
            self.position += 1
            self.balance -= current_price
        elif action == 1:  # Sell
            self.position -= 1
            self.balance += current_price

        reward = self.balance + self.position * current_price
        done = self.current_step >= len(price_data) - 1

        obs = np.array(self.price_history)
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def get_valid_moves(self, state):
        return np.ones(3)  # All moves are always valid in this simplified environment

    def get_next_state(self, state, player, action):
        new_state = state.copy()
        if action == 0:  # Buy
            new_state[-1] += 1
        elif action == 1:  # Sell
            new_state[-1] -= 1
        return new_state, -player

    def get_reward_for_player(self, state, player):
        return None  # Reward is calculated in the step function

# Neural Network Model
class TradingNN(nn.Module):
    def __init__(self, input_size, action_size):
        super(TradingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.forward(state)
        return policy.detach().numpy().flatten(), value.item()

# Visualization
def draw_chart(screen, data, color):
    if len(data) < 2:
        return
    scaled_data = [(i, HEIGHT - (d - min(data)) / (max(data) - min(data)) * HEIGHT) for i, d in enumerate(data)]
    pygame.draw.lines(screen, color, False, scaled_data, 2)

def draw_liquidation_levels(screen, levels):
    for level in levels:
        y = HEIGHT - (level - min(price_data)) / (max(price_data) - min(price_data)) * HEIGHT
        pygame.draw.line(screen, RED, (0, y), (WIDTH, y), 2)

def calculate_liquidation_levels(price_data, open_interest_data):
    oi_threshold = np.mean(open_interest_data) + np.std(open_interest_data)
    high_oi_levels = [price for price, oi in zip(price_data, open_interest_data) if oi > oi_threshold]
    return high_oi_levels

# Main game loop
def main():
    env = TradingEnv()
    model = TradingNN(1000, 3)
    mcts = MCTS(env, model, {'num_simulations': 50})
    ppo_model = PPO('MlpPolicy', env, verbose=1)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        # Update and draw price chart
        draw_chart(screen, price_data, GREEN)

        # Calculate and draw liquidation levels
        liquidation_levels = calculate_liquidation_levels(price_data, open_interest_data)
        draw_liquidation_levels(screen, liquidation_levels)

        # Run MCTS and PPO
        if len(price_data) >= 1000:
            state = env.reset()
            root = mcts.run(state, 1)
            action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

            # PPO action
            ppo_action, _ = ppo_model.predict(state)

            # Take action in environment
            obs, reward, done, _ = env.step(action)

            # Train PPO model
            ppo_model.learn(total_timesteps=1)

        # Display metrics
        font = pygame.font.Font(None, 36)
        metrics_text = f"Balance: {env.balance:.2f}, Position: {env.position}"
        text = font.render(metrics_text, True, WHITE)
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()