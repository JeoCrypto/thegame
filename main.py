# main.py
import asyncio
import websockets
import json
import torch
import torch.optim as optim
from collections import deque

from config import SYMBOL, LEVERAGE
from binance_client import BinanceClient
from trading_env import TradingEnv
from model import TradingModel
from visualization import init_pygame, update_display

async def main():
    binance_client = BinanceClient()
    env = TradingEnv()
    model = TradingModel(input_size=5, hidden_size=64, output_size=3)
    optimizer = optim.Adam(model.parameters())
    
    screen = init_pygame()
    price_data = deque(maxlen=1000)

    await binance_client.set_leverage(SYMBOL, LEVERAGE)

    async def handle_websocket_message(message):
        data = json.loads(message)
        price = float(data['p'])
        price_data.append(price)
        env.update_data(price)

    async with websockets.connect(f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@trade") as websocket:
        episode = 0
        while True:
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                message = await websocket.recv()
                await handle_websocket_message(message)

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = model(state_tensor)
                action = torch.argmax(action_probs).item()

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Train the model
                optimizer.zero_grad()
                loss = -torch.log(action_probs[0][action]) * reward
                loss.backward()
                optimizer.step()

                state = next_state

                update_display(screen, list(price_data), info['balance'], info['position'], episode)

            print(f"Episode {episode} finished with reward: {episode_reward}")
            episode += 1

if __name__ == "__main__":
    asyncio.run(main())
    