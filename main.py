import asyncio
import pygame
from trading_env import BreakoutEnv
from model import initialize_model_and_data
import torch
import torch.optim as optim
from visualization import init_pygame, update_display
from config import SYMBOL, WIDTH, HEIGHT
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    try:
        model, data_fetcher, fetch_task = await initialize_model_and_data(SYMBOL)
    except ValueError as e:
        print(f"Initialization failed: {e}")
        return

    # Initialize Pygame
    screen = init_pygame()
    clock = pygame.time.Clock()

    env = BreakoutEnv(data_fetcher, width=WIDTH, height=HEIGHT)
    optimizer = torch.optim.Adam(model.parameters())

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Train the model
            optimizer.zero_grad()
            loss = -torch.log(action_probs[0][action]) * reward
            loss.backward()
            optimizer.step()

            state = next_state

            # Update the display
            update_display(screen, env.paddle_x, env.ball_x, env.ball_y, env.blocks, info['score'])

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    fetch_task.cancel()
                    return

            # Control the frame rate
            clock.tick(60)

        print(f"Episode {episode + 1} finished with score: {info['score']} and reward: {total_reward}")

    pygame.quit()
    fetch_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())