# binance_client.py
import logging
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
import asyncio
from decimal import Decimal

from config import API_KEY, API_SECRET, BASE_URL

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.base_url = BASE_URL

    async def _request(self, method, path, params=None):
        url = f"{self.base_url}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params:
            query_string = urlencode(params)
            signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
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
        symbol_info = next((s for s in data['symbols'] if s['symbol'] == symbol), None)
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