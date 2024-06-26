import logging
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
import asyncio
import websockets
import json

logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _get_server_time(self):
        async with self.session.get(f"{self.base_url}/fapi/v1/time") as response:
            if response.status == 200:
                data = await response.json()
                return data['serverTime']
            else:
                raise Exception(f"Failed to get server time: {response.status}")

    async def _request(self, method, path, params=None):
        url = f"{self.base_url}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}

        if params is None:
            params = {}

        server_time = await self._get_server_time()
        params['timestamp'] = server_time

        query_string = urlencode(params)
        signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature

        async with self.session.request(method, url, params=params, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"API request failed with status {response.status}: {await response.text()}")

    async def get_current_price(self, symbol):
        path = "/fapi/v1/ticker/price"
        params = {"symbol": symbol}
        data = await self._request("GET", path, params)
        return float(data['price'])

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

    async def create_websocket_connection(self, symbol):
        ws_url = f"wss://fstream.binance.com/ws/{symbol.lower()}@trade"
        while True:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"WebSocket connection established for {symbol}")
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        price = float(data['p'])
                        yield price
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                await asyncio.sleep(1)

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("BinanceClient session closed.")