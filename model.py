import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import time
from collections import deque
from binance_client import BinanceClient
from config import API_KEY, API_SECRET, BASE_URL, SYMBOL
import logging

logger = logging.getLogger(__name__)


class AdvancedTradingModel(nn.Module):
    def __init__(self, max_input_size, hidden_size, output_size):
        super(AdvancedTradingModel, self).__init__()
        self.max_input_size = max_input_size
        self.fc1 = nn.Linear(max_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        # Input normalization
        x = (x - x.mean()) / (x.std() + 1e-8)

        # Pad or truncate input to match max_input_size
        if x.size(1) < self.max_input_size:
            x = F.pad(x, (0, self.max_input_size - x.size(1)))
        elif x.size(1) > self.max_input_size:
            x = x[:, :self.max_input_size]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Sanity check
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning(f"NaN or Inf detected in forward pass. Input: {x}")
            x = torch.where(torch.isnan(x) | torch.isinf(x),
                            torch.zeros_like(x), x)

        return F.log_softmax(x / self.temperature.clamp(min=1e-8), dim=-1)

    def get_action(self, state, env):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            log_probs = self(state_tensor)
            action_probs = torch.exp(log_probs)

            # Apply constraints
            if env.long_position >= 5:  # Limit long positions
                action_probs[0][0] = 0  # Disable buy action
            if env.short_position <= -5:  # Limit short positions
                action_probs[0][2] = 0  # Disable sell action

            # Renormalize probabilities
            action_probs = action_probs / (action_probs.sum() + 1e-8)

            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                logger.warning(
                    f"NaN or Inf detected in action probabilities. Using uniform distribution.")
                action_probs = torch.ones_like(
                    action_probs) / action_probs.size(-1)

            action = torch.multinomial(action_probs, 1).item()
        return action


class AsyncFinancialDataFetcher:
    def __init__(self, binance_client, symbol=SYMBOL, max_data_points=1000, update_interval=1):
        self.symbol = symbol
        self.max_data_points = max_data_points
        self.price_data = deque(maxlen=max_data_points)
        self.liquidation_levels = []
        self.binance_client = binance_client
        self.update_interval = update_interval
        self.lock = asyncio.Lock()

    async def start(self):
        await asyncio.gather(
            self.fetch_real_time_data(),
            self.update_liquidation_levels()
        )

    async def fetch_real_time_data(self):
        try:
            async for msg in self.binance_client.create_websocket_connection(f"{self.symbol.lower()}@trade"):
                price = float(msg['p'])
                async with self.lock:
                    self.price_data.append(price)
                logger.debug(f"Received price: {price}. Total prices: {
                             len(self.price_data)}")
        except Exception as e:
            logger.error(f"Error in fetch_real_time_data: {e}")
            raise

    async def update_liquidation_levels(self):
        while True:
            try:
                open_interest = await self.binance_client.get_open_interest(self.symbol)
                funding_rate = await self.binance_client.get_funding_rate(self.symbol)

                async with self.lock:
                    if self.price_data:
                        current_price = self.price_data[-1]
                        self.liquidation_levels = [
                            current_price * (1 + funding_rate * 2),
                            current_price * (1 - funding_rate * 2)
                        ]
                        logger.info(f"Liquidation levels updated: {
                                    self.liquidation_levels}")
                    else:
                        logger.warning(
                            "No price data available to set liquidation levels")
            except Exception as e:
                logger.error(f"Error updating liquidation levels: {e}")

            await asyncio.sleep(self.update_interval)

    async def get_current_state(self):
        async with self.lock:
            if not self.price_data or not self.liquidation_levels:
                return None
            # Ensure we only return up to max_data_points
            price_data = list(self.price_data)[-self.max_data_points:]
            return torch.tensor(price_data + self.liquidation_levels, dtype=torch.float32)


async def initialize_model_and_data(symbol=SYMBOL, binance_client=None, max_retries=5, retry_delay=5, timeout=30):
    if binance_client is None:
        binance_client = BinanceClient(API_KEY, API_SECRET, BASE_URL)

    for attempt in range(max_retries):
        logger.info(f"Initialization attempt {attempt + 1}/{max_retries}")
        data_fetcher = AsyncFinancialDataFetcher(binance_client, symbol)
        fetch_task = asyncio.create_task(data_fetcher.start())

        start_time = time.time()
        try:
            while len(data_fetcher.price_data) < 100:
                if time.time() - start_time > timeout:
                    raise asyncio.TimeoutError(
                        "Timeout while waiting for initial data")
                await asyncio.sleep(0.1)

            logger.info(
                f"Received {len(data_fetcher.price_data)} price points")

            # Set max_input_size to be larger than the expected input size
            max_input_size = data_fetcher.max_data_points + 10  # Add some buffer

            model = AdvancedTradingModel(
                max_input_size=max_input_size, hidden_size=64, output_size=3)
            logger.info(f"Successfully initialized model with max input size: {
                        max_input_size}")
            return model, data_fetcher, fetch_task

        except asyncio.TimeoutError:
            logger.warning("Timeout while waiting for initial data")
            fetch_task.cancel()
        except Exception as e:
            logger.error(f"Error during data fetching: {e}")
            fetch_task.cancel()

        logger.warning(
            f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
        await asyncio.sleep(retry_delay)

    raise ValueError("Failed to fetch initial data after multiple attempts")






