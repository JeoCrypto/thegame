# model.py
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
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedTradingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FinancialDataFetcher:
    def __init__(self, symbol=SYMBOL, max_data_points=1000):
        self.symbol = symbol
        self.price_data = deque(maxlen=max_data_points)
        self.liquidation_levels = []
        self.binance_client = BinanceClient(API_KEY, API_SECRET, BASE_URL)

    def get_current_state(self):
        if not self.price_data or not self.liquidation_levels:
            return None
        return torch.tensor(list(self.price_data) + self.liquidation_levels, dtype=torch.float32)

    async def fetch_real_time_data(self):
        logger.info(f"Starting to fetch real-time data for {self.symbol}")
        async for msg in self.binance_client.create_websocket_connection(f"{self.symbol.lower()}@trade"):
            price = float(msg['p'])
            self.price_data.append(price)
            logger.debug(f"Received price: {price}. Total prices: {len(self.price_data)}")

    async def fetch_liquidation_levels(self):
        logger.info("Fetching liquidation levels")
        try:
            open_interest = await self.binance_client.get_open_interest(self.symbol)
            funding_rate = await self.binance_client.get_funding_rate(self.symbol)
            
            current_price = self.price_data[-1]
            self.liquidation_levels = [
                current_price * (1 + funding_rate * 2),
                current_price * (1 - funding_rate * 2)
            ]
            logger.info(f"Liquidation levels set: {self.liquidation_levels}")
        except Exception as e:
            logger.error(f"Error fetching liquidation levels: {e}")

async def initialize_model_and_data(symbol=SYMBOL, max_retries=5, retry_delay=5, timeout=30):
    for attempt in range(max_retries):
        logger.info(f"Initialization attempt {attempt + 1}/{max_retries}")
        data_fetcher = FinancialDataFetcher(symbol)
        fetch_task = asyncio.create_task(data_fetcher.fetch_real_time_data())
        
        start_time = time.time()
        try:
            # Wait for some initial data (e.g., 100 price points)
            while len(data_fetcher.price_data) < 100:
                if time.time() - start_time > timeout:
                    raise asyncio.TimeoutError("Timeout while waiting for initial data")
                await asyncio.sleep(0.1)
                if len(data_fetcher.price_data) > 0:
                    logger.info(f"Received {len(data_fetcher.price_data)} price points")

            # Fetch liquidation levels after getting some price data
            await data_fetcher.fetch_liquidation_levels()

            input_size = len(data_fetcher.price_data) + len(data_fetcher.liquidation_levels)
            model = AdvancedTradingModel(
                input_size=input_size, 
                hidden_size=64, 
                output_size=3
            )
            logger.info(f"Successfully initialized with input size: {input_size}")
            return model, data_fetcher, fetch_task

        except asyncio.TimeoutError:
            logger.warning("Timeout while waiting for initial data")
            fetch_task.cancel()
        except Exception as e:
            logger.error(f"Error during data fetching: {e}")
            fetch_task.cancel()
        
        logger.warning(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
        await asyncio.sleep(retry_delay)
    
    raise ValueError("Failed to fetch initial data after multiple attempts")