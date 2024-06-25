# binance_client.py
import logging
import hmac
import logging
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
import asyncio
import websockets
import json

API_KEY = "uKQtYQWg6wERwMP2gZmXGVoxW2IGx2iPomvUXLVd8hA0awdFyifgBjaQcezWaLiS"
API_SECRET = "p6xRauRC2NmD3JMzrHKFDnUKln2xxQ8pyISDDnrShK4smI8vIsqgaWGfeDRCqvHh"
BASE_URL = "https://fapi.binance.com"

logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    async def _get_server_time(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/fapi/v1/time") as response:
                if response.status == 200:
                    data = await response.json()
                    return data['serverTime']
                else:
                    raise Exception(f"Failed to get server time: {
                                    response.status}")

    async def _request(self, method, path, params=None):
        url = f"{self.base_url}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}

        if params is None:
            params = {}

        # Add server time to params
        server_time = await self._get_server_time()
        params['timestamp'] = server_time

        query_string = urlencode(params)
        signature = hmac.new(self.api_secret.encode(
            'utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API request failed with status {response.status}: {await response.text()}")

    async def get_exchange_info(self, symbol):
        path = "/fapi/v1/exchangeInfo"
        data = await self._request("GET", path)
        symbol_info = next(
            (s for s in data['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            raise ValueError(f"Symbol {symbol} not found")
        return symbol_info

    async def get_account_balance(self):
        path = "/fapi/v2/account"
        params = {"timestamp": int(asyncio.get_event_loop().time() * 1000)}
        data = await self._request("GET", path, params)
        return {asset['asset']: float(asset['balance']) for asset in data['assets']}

    async def set_leverage(self, symbol, leverage):
        path = "/fapi/v1/leverage"
        params = {
            "symbol": symbol,
            "leverage": leverage,
            "timestamp": int(asyncio.get_event_loop().time() * 1000)
        }
        await self._request("POST", path, params)

    async def place_order(self, symbol, side, quantity, price=None, order_type="MARKET"):
        path = "/fapi/v1/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timestamp": int(asyncio.get_event_loop().time() * 1000)
        }
        if price:
            params["price"] = price
        return await self._request("POST", path, params)

    async def create_websocket_connection(self, stream):
        ws_url = f"wss://fstream.binance.com/ws/{stream}"
        async for websocket in websockets.connect(ws_url):
            try:
                while True:
                    message = await websocket.recv()
                    yield json.loads(message)
            except websockets.ConnectionClosed:
                continue

    async def get_open_interest(self, symbol):
        logger.info(f"Fetching open interest for {symbol}")
        path = "/fapi/v1/openInterest"
        params = {"symbol": symbol}
        data = await self._request("GET", path, params)
        logger.info(f"Open interest data received: {data}")
        return float(data['openInterest'])

    async def get_funding_rate(self, symbol):
        logger.info(f"Fetching funding rate for {symbol}")
        path = "/fapi/v1/fundingRate"
        params = {"symbol": symbol}
        data = await self._request("GET", path, params)
        logger.info(f"Funding rate data received: {data}")
        return float(data[0]['fundingRate'])