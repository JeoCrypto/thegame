import pygame
import random
import numpy as np
import gym
from gym import spaces
import websocket
import json
import threading
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
import time
import hmac
import hashlib
from urllib.parse import urlencode
import logging
import requests
import asyncio
import aiohttp

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Advanced Trading Bot Simulation')

# Colors
BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)

# Global variables for real-time data
price_data = deque(maxlen=1000)
open_interest_data = deque(maxlen=1000)

# Logger setup
logger = logging.getLogger(__name__)

# Binance API credentials
API_KEY = "uKQtYQWg6wERwMP2gZmXGVoxW2IGx2iPomvUXLVd8hA0awdFyifgBjaQcezWaLiS"
API_SECRET = "p6xRauRC2NmD3JMzrHKFDnUKln2xxQ8pyISDDnrShK4smI8vIsqgaWGfeDRCqvHh"
BASE_URL = "https://fapi.binance.com"

class BinanceClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json;charset=utf-8", "X-MBX-APIKEY": self.api_key}
        )

    def _hashing(self, query_string):
        return hmac.new(self.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

    def _dispatch_request(self, http_method):
        return {
            "GET": self.session.get,
            "DELETE": self.session.delete,
            "PUT": self.session.put,
            "POST": self.session.post,
        }.get(http_method, "GET")

    def _send_signed_request(self, http_method, url_path, payload=None):
        if payload is None:
            payload = {}
        query_string = urlencode(payload)
        query_string = query_string.replace("%27", "%22")
        if query_string:
            query_string = "{}&timestamp={}".format(query_string, self._get_timestamp())
        else:
            query_string = "timestamp={}".format(self._get_timestamp())

        url = self.base_url + url_path + "?" + query_string + "&signature=" + self._hashing(query_string)
        logger.debug(f"{http_method} {url}")
        params = {"url": url, "params": {}}
        response = self._dispatch_request(http_method)(**params)
        if response.headers['Content-Type'] != 'application/json':
            raise ValueError(f"Unexpected content type: {response.headers['Content-Type']}, response: {response.text}")
        return response.json()

    def _send_public_request(self, url_path, payload=None):
        if payload is None:
            payload = {}
        query_string = urlencode(payload, True)
        url = self.base_url + url_path
        if query_string:
            url = url + "?" + query_string
        logger.debug(f"{url}")
        response = self._dispatch_request("GET")(url=url)
        if response.headers['Content-Type'] != 'application/json':
            raise ValueError(f"Unexpected content type: {response.headers['Content-Type']}, response: {response.text}")
        return response.json()

    def _get_timestamp(self):
        server_time = self._send_public_request('/fapi/v1/time')
        return server_time['serverTime']

    def get_precision(self, symbol):
        exchange_info = self._send_public_request('/fapi/v1/exchangeInfo')
        symbol_info = next((x for x in exchange_info['symbols'] if x['symbol'] == symbol), None)

        if symbol_info is None:
            raise ValueError(f"Symbol '{symbol}' not found in exchange information.")

        price_precision = symbol_info['pricePrecision']
        qty_precision = symbol_info['quantityPrecision']

        return int(price_precision), int(qty_precision)

    async def get_precision_and_tick_size(self, symbol):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/fapi/v1/exchangeInfo') as response:
                exchange_info = await response.json()

        symbol_info = next((x for x in exchange_info['symbols'] if x['symbol'] == symbol), None)

        if symbol_info is None:
            raise ValueError(f"Symbol '{symbol}' not found in exchange information.")

        price_precision = symbol_info['pricePrecision']
        qty_precision = symbol_info['quantityPrecision']
        tick_size = Decimal(next(f['tickSize'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'))
        max_price = Decimal(next(f['maxPrice'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'))

        return int(price_precision), int(qty_precision), tick_size, max_price

    def get_listen_key(self):
        response = self._send_signed_request('POST', '/fapi/v1/listenKey')
        return response['listenKey']

    def update_listen_key(self):
        self._send_signed_request('PUT', '/fapi/v1/listenKey')

    def delete_listen_key(self):
        self._send_signed_request('DELETE', '/fapi/v1/listenKey')

    def get_account_balance(self, asset: str):
        account_data = self._send_signed_request('GET', '/fapi/v2/account')

        if 'assets' not in account_data:
            return 0

        amount = [x for x in account_data['assets'] if x.get('asset') == asset][0]['walletBalance']
        return float(amount)

    def get_open_positions(self, symbol: str):
        positions = self._send_signed_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
        return positions

    def get_open_orders(self, symbol: str):
        orders = self._send_signed_request('GET', '/fapi/v1/openOrders', {'symbol': symbol})
        return orders

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self._send_signed_request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': leverage})
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")

    def set_margin_type(self, symbol: str, margin_type: str):
        try:
            self._send_signed_request('POST', '/fapi/v1/marginType', {'symbol': symbol, 'marginType': margin_type})
        except Exception as e:
            logger.error(f"Error setting margin type: {e}")

    def set_hedge_mode(self, position_mode: bool):
        try:
            self._send_signed_request('POST', '/fapi/v1/positionSide/dual', {'dualSidePosition': position_mode})
        except Exception as e:
            logger.error(f"Error setting hedge mode: {e}")

    def get_notional_brackets(self, symbol: str):
        try:
            result = self._send_signed_request('GET', '/fapi/v1/leverageBracket', {'symbol': symbol})
            return result[0]['brackets']
        except Exception as e:
            logger.error(f"Error getting notional brackets: {e}")

    async def renew_listen_key(self, listen_key: str):
        try:
            await self._send_signed_request('PUT', '/fapi/v1/listenKey', {'listenKey': listen_key})
        except Exception as e:
            logger.error(f"Error renewing listen key: {e}")

    async def keep_alive_listen_key(self):
        try:
            listen_key = await self.get_listen_key()
            while True:
                await asyncio.sleep(1800)  # 30 minutes
                await self.renew_listen_key(listen_key)
        except Exception as e:
            logger.error(f"Error keeping listen key alive: {e}")

# WebSocket for real-time data
def on_message(ws, message):
    global price_data, open_interest_data
    try:
        json_message = json.loads(message)
        price = float(json_message['p'])
        open_interest = float(json_message['q'])  # Just for example
        price_data.append(price)
        open_interest_data.append(open_interest)
    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### WebSocket closed ###")

def on_open(ws):
    print("### WebSocket opened ###")

def start_websocket():
    while True:
        try:
            socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
            ws = websocket.WebSocketApp(socket,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
            ws.run_forever()
        except Exception as e:
            print(f"WebSocket error: {e}")
            time.sleep(5)  # Wait before trying to reconnect

# Start WebSocket in a separate thread
ws_thread = threading.Thread(target=start_websocket)
ws_thread.daemon = True
ws_thread.start()

# MCTS Node and MCTS classes (refactored)
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
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select_child(self):
        best_score = -np.inf
        best_action, best_child = None, None

        for action, child in self.children.items():
            score = self.ucb_score(child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, child):
        prior_score = child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
        value_score = -child.value() if child.visit_count > 0 else 0
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
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.price_history = list(price_data)
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
    binance_client = BinanceClient(API_KEY, API_SECRET, BASE_URL)
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
        draw_chart(screen, list(price_data), GREEN)

        # Calculate and draw liquidation levels
        liquidation_levels = calculate_liquidation_levels(list(price_data), list(open_interest_data))
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

            # Execute trade with Binance API
            if action == 0:
                binance_client.set_leverage("BTCUSDT", 10)
                print("Buy action taken")
            elif action == 1:
                binance_client.set_leverage("BTCUSDT", 10)
                print("Sell action taken")
            else:
                print("Hold action taken")

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