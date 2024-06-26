import asyncio
import logging
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from binance_client import BinanceClient
from config import API_KEY, API_SECRET, BASE_URL, SYMBOL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AdvancedTradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(0) if x.dim() == 1 else x  # Ensure input is 3D: (batch, seq, features)
        _, (hn, _) = self.lstm(x.float())
        out = self.fc(hn[-1])
        return out

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            q_values = self.forward(state)
            return q_values.argmax().item()
        
class AsyncFinancialDataFetcher:
    def __init__(self, symbol: str, api_key: str, api_secret: str, base_url: str):
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.client = None
        self.current_price = None
        self.volume = 0
        self.adx = 0
        self.liquidation_levels: List[float] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []

    async def initialize(self):
        self.client = BinanceClient(self.api_key, self.api_secret, self.base_url)
        await self.client.__aenter__()
        # Fetch initial price
        self.current_price = await self.client.get_current_price(self.symbol)
        self.price_history.append(self.current_price)
        logger.info(f"Initial price: {self.current_price}")
        await self.update_liquidation_levels()

    async def process_price(self, price: float):
        self.current_price = price
        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        logger.info(f"New price: {price}")
        await self.calculate_adx()
        await self.update_liquidation_levels()

    async def start_data_stream(self):
        try:
            async for price in self.client.create_websocket_connection(self.symbol):
                await self.process_price(price)
        except Exception as e:
            logger.error(f"Error in data stream: {e}")
        finally:
            await self.close()

    async def calculate_adx(self):
        if len(self.price_history) < 14:
            return
        df = pd.DataFrame({'close': self.price_history})
        df['high'] = df['close'].rolling(2).max()
        df['low'] = df['close'].rolling(2).min()
        df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                             df['high'] - df['high'].shift(1), 0)
        df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             df['low'].shift(1) - df['low'], 0)
        df['TR'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['+DI14'] = 100 * (df['+DM'].rolling(14).sum() / df['TR'].rolling(14).sum())
        df['-DI14'] = 100 * (df['-DM'].rolling(14).sum() / df['TR'].rolling(14).sum())
        df['DX'] = 100 * abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14'])
        self.adx = df['DX'].rolling(14).mean().iloc[-1]

    async def update_liquidation_levels(self):
        if self.current_price is None or self.current_price == 0:
            logger.warning("Current price is not set. Unable to calculate liquidation levels.")
            return
        try:
            open_interest = await self.client.get_open_interest(self.symbol)
            funding_rate = await self.client.get_funding_rate(self.symbol)
            
            logger.info(f"Current price: {self.current_price}")
            logger.info(f"Open interest: {open_interest}")
            logger.info(f"Funding rate: {funding_rate}")

            if funding_rate == 0:
                logger.warning("Funding rate is zero. Using default value of 0.0001")
                funding_rate = 0.0001

            self.liquidation_levels = [
                self.current_price * (1 + funding_rate * 2),
                self.current_price * (1 - funding_rate * 2)
            ]
            logger.info(f"Updated liquidation levels: {self.liquidation_levels}")
        except Exception as e:
            logger.error(f"Error updating liquidation levels: {e}")

    def get_game_speed(self) -> float:
        volume_ma = np.mean(self.volume_history) if self.volume_history else 1
        adx_factor = min(self.adx / 100, 1) if self.adx else 0.5
        speed = (volume_ma * adx_factor) / 1000
        return max(0.1, min(speed, 2.0))

    async def close(self):
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None