import asyncio
import pygame
from trading_env import BreakoutEnv
from model import AdvancedTradingModel, AsyncFinancialDataFetcher, initialize_model_and_data
import torch
import torch.optim as optim
from visualization import init_pygame, update_display, improved_update_display
from config import SYMBOL, WIDTH, HEIGHT, API_KEY, API_SECRET, BASE_URL
import logging
from binance_client import BinanceClient
import cProfile
import pstats
import io
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_model(model, optimizer, episode, path='model_checkpoint.pth'):
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved at episode {episode}")


def load_model(model, optimizer, path='model_checkpoint.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path)

        # Check if the saved model has the temperature parameter
        if 'temperature' not in checkpoint['model_state_dict']:
            print("Old model version detected. Initializing temperature parameter.")
            checkpoint['model_state_dict']['temperature'] = torch.ones(1) * 1.0

        model.load_state_dict(checkpoint['model_state_dict'])

        # Instead of loading the optimizer state, we'll reinitialize it
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        start_episode = checkpoint['episode']
        print(f"Model loaded from episode {start_episode}")
        return start_episode, optimizer
    else:
        print("No saved model found. Starting from scratch.")
        return 0, optimizer


async def main():
    binance_client = None
    try:
        binance_client = BinanceClient(API_KEY, API_SECRET, BASE_URL)
        logger.info("Binance client initialized.")

        model, data_fetcher, fetch_task = await initialize_model_and_data(SYMBOL, binance_client)
        logger.info("Model initialized.")

        optimizer = optim.RMSprop(
            model.parameters(), lr=0.00025, alpha=0.99, eps=1e-08)

        start_episode, optimizer = load_model(model, optimizer)

        screen = init_pygame()
        clock = pygame.time.Clock()

        env = BreakoutEnv(data_fetcher, width=WIDTH, height=HEIGHT)

        num_episodes = 1000
        save_interval = 50

        for episode in range(start_episode, num_episodes):
            state = await env.reset()
            done = False
            total_reward = 0

            logger.info(f"Starting episode {episode + 1}")

            while not done:
                try:
                    action = model.get_action(state, env)
                    next_state, reward, done, info = await env.step(action)

                    reward = max(min(reward, 1), -1)  # Reward clipping

                    total_reward += reward

                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    log_probs = model(state_tensor)

                    loss = -log_probs[0][action] * reward

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN or Inf loss detected. State: {
                                       state}, Action: {action}, Reward: {reward}")
                        continue

                    loss.backward()

                    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

                    optimizer.step()

                    state = next_state

                    improved_update_display(
                        screen, env.paddle_x, env.ball_x, env.ball_y, env.blocks, info)

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            logger.info(
                                "Pygame QUIT event received. Shutting down.")
                            return

                    clock.tick(60)

                except Exception as e:
                    logger.error(f"Error during episode: {e}")
                    break

        logger.info(f"Episode {
                    episode + 1} finished with score: {info['score']} and reward: {total_reward}")

        # Add a small delay between episodes to allow for screen updates
        await asyncio.sleep(0.1)

        if (episode + 1) % save_interval == 0:
            save_model(model, optimizer, episode + 1,
                       path=f"model_checkpoint.pth")

        logger.info("Training completed. Shutting down.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        pygame.quit()
        if fetch_task:
            fetch_task.cancel()
        if binance_client:
            await binance_client.close()

if __name__ == "__main__":
    asyncio.run(main())
