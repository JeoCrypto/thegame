import asyncio
import pygame
from trading_env import BreakoutEnv
from model import initialize_model_and_data, AdvancedTradingModel
import torch
import torch.optim as optim
from visualization import init_pygame, update_display
from config import SYMBOL, WIDTH, HEIGHT, API_KEY, API_SECRET, BASE_URL
import logging
from binance_client import BinanceClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Initialize Binance Client
    binance_client = BinanceClient(API_KEY, API_SECRET, BASE_URL)
    logger.info("Binance client initialized.")

    try:
        model, data_fetcher, fetch_task = await initialize_model_and_data(SYMBOL, binance_client)
        logger.info(f"Model initialized. Input size: {model.fc1.in_features}")
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        return

    # Initialize Pygame
    screen = init_pygame()
    clock = pygame.time.Clock()

    env = BreakoutEnv(data_fetcher, width=WIDTH, height=HEIGHT)
    optimizer = torch.optim.Adam(model.parameters())

    logger.info(f"Environment initialized. Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        logger.info(f"Starting episode {episode + 1}")
        logger.debug(f"Initial state shape: {state.shape}")

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logger.debug(f"State tensor shape: {state_tensor.shape}")

            # Reinitialize the model if the input size changes
            expected_size = len(data_fetcher.price_data) + len(data_fetcher.liquidation_levels)
            if state_tensor.shape[1] != expected_size:
                logger.warning(f"Unexpected state shape: {state_tensor.shape}, expected: (1, {expected_size})")
                logger.warning(f"Price data length: {len(data_fetcher.price_data)}")
                logger.warning(f"Liquidation levels length: {len(data_fetcher.liquidation_levels)}")

                # Reinitialize the model
                model = AdvancedTradingModel(input_size=expected_size, hidden_size=64, output_size=3)
                optimizer = torch.optim.Adam(model.parameters())
                logger.info(f"Reinitialized model with input size: {expected_size}")
                continue

            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()

            next_state, reward, done, info = await env.step(action)
            total_reward += reward
            logger.debug(f"Action: {action}, Reward: {reward}, Done: {done}")

            # Train the model
            optimizer.zero_grad()
            loss = -torch.log(action_probs[0][action]) * reward
            loss.backward()
            optimizer.step()

            state = next_state

            # Update the display
            update_display(screen, env.paddle_x, env.ball_x, env.ball_y, env.blocks, info)

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Pygame QUIT event received. Shutting down.")
                    pygame.quit()
                    fetch_task.cancel()
                    await binance_client.close()
                    return

            # Control the frame rate
            clock.tick(60)

        logger.info(f"Episode {episode + 1} finished with score: {info['score']} and reward: {total_reward}")

    logger.info("Training completed. Shutting down.")
    pygame.quit()
    fetch_task.cancel()
    await binance_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")








