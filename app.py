import asyncio
import logging
import os
import torch
import torch.optim as optim
from model import AsyncFinancialDataFetcher, AdvancedTradingModel
from visualization import GameVisualizer
from trading_env import TradingEnv
from config import API_KEY, API_SECRET, BASE_URL, SYMBOL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoWolfGame:
    def __init__(self):
        self.data_fetcher = None
        self.env = None
        self.visualizer = None
        self.model = None
        self.optimizer = None

    async def initialize(self):
        self.data_fetcher = AsyncFinancialDataFetcher(SYMBOL, API_KEY, API_SECRET, BASE_URL)
        await self.data_fetcher.initialize()
        
        self.env = TradingEnv(self.data_fetcher)
        self.visualizer = GameVisualizer(self.env.width, self.env.height)
        
        # Initialize the AI model
        input_size = self.env.observation_space.shape[0]
        hidden_size = 128
        num_layers = 2
        output_size = self.env.action_space.n
        self.model = AdvancedTradingModel(input_size, hidden_size, num_layers, output_size)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.99, eps=1e-08)

    def load_model(self, path='checkpoints/model_checkpoint.pth'):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            if 'model_architecture' in checkpoint:
                saved_input_size = checkpoint['model_architecture']['input_size']
                current_input_size = self.env.observation_space.shape[0]

                if saved_input_size != current_input_size:
                    logger.warning(f"Input size mismatch: saved {saved_input_size}, current {current_input_size}. Creating new model.")
                    self.create_new_model(current_input_size)
                    return 0
                
                self.model = AdvancedTradingModel(
                    input_size=checkpoint['model_architecture']['input_size'],
                    hidden_size=checkpoint['model_architecture']['hidden_size'],
                    num_layers=checkpoint['model_architecture']['num_layers'],
                    output_size=checkpoint['model_architecture']['output_size']
                )
            else:
                logger.warning("Old checkpoint format detected. Creating new model.")
                self.create_new_model(self.env.observation_space.shape[0])
                return 0

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.99, eps=1e-08)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            logger.info(f"Model loaded from episode {start_episode}")
            return start_episode
        else:
            logger.info("No saved model found. Starting from scratch.")
            self.create_new_model(self.env.observation_space.shape[0])
            return 0

    def create_new_model(self, input_size):
        hidden_size = 128
        num_layers = 2
        output_size = self.env.action_space.n
        self.model = AdvancedTradingModel(input_size, hidden_size, num_layers, output_size)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.99, eps=1e-08)
        logger.info(f"Created new model with input size: {input_size}")

    def save_model(self, episode, path='checkpoints/model_checkpoint.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_architecture': {
            'input_size': self.model.lstm.input_size,
            'hidden_size': self.model.lstm.hidden_size,
            'num_layers': self.model.lstm.num_layers,
            'output_size': self.model.fc.out_features
            }
            }, path)
        logger.info(f"Model saved at episode {episode}")
        
    async def cleanup(self):
        if self.data_fetcher:
            await self.data_fetcher.close()
        if self.visualizer:
            self.visualizer.close()
        logger.info("Cleanup completed.")

    async def run(self):
        await self.initialize()
        start_episode = self.load_model()
        
        data_stream_task = asyncio.create_task(self.data_fetcher.start_data_stream())

        num_episodes = 1000
        save_interval = 50

        try:
            for episode in range(start_episode, num_episodes):
                state = self.env.reset()
                total_reward = 0
                done = False

                while not done:
                    # Check if data_stream_task has raised an exception
                    if data_stream_task.done():
                        data_stream_task.result()  # This will raise the exception if there was one

                    action = self.model.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    total_reward += reward
                    state = next_state

                    self.visualizer.render(self.env)

                    if self.visualizer.check_quit():
                        return

                logger.info(f"Episode {episode + 1} finished with reward: {total_reward}")
                
                if (episode + 1) % save_interval == 0:
                    self.save_model(episode + 1)

        except asyncio.CancelledError:
            logger.info("Game was cancelled")
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
        finally:
            # Cancel the data stream task if it's still running
            if not data_stream_task.done():
                data_stream_task.cancel()
                try:
                    await data_stream_task
                except asyncio.CancelledError:
                    pass
            await self.cleanup()

async def main():
    game = CryptoWolfGame()
    try:
        await game.run()
    except asyncio.CancelledError:
        logger.info("Game was cancelled")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        await game.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Game was interrupted by user")