import hmac
import time
import hashlib
import requests
from urllib.parse import urlencode
import logging
import asyncio
import aiohttp
from decimal import Decimal

logger = logging.getLogger(__name__)

API_KEY = "uKQtYQWg6wERwMP2gZmXGVoxW2IGx2iPomvUXLVd8hA0awdFyifgBjaQcezWaLiS"
API_SECRET = "p6xRauRC2NmD3JMzrHKFDnUKln2xxQ8pyISDDnrShK4smI8vIsqgaWGfeDRCqvHh"
BASE_URL = "https://fapi.binance.com"

class BinanceClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json;charset=utf-8", "X-MBX-APIKEY": self.api_key}
        )
    
    def _hashing(self, query_string):
        return hmac.new(self.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

    def _dispatch_request(self, http_method):
        return {
            "GET": self.session.get,
            "DELETE": self.session.delete,
            "PUT": self.session.put,
            "POST": self.session.post,
        }.get(http_method, "GET")

    def _send_signed_request(self, http_method, url_path, payload=None):
        if payload is None:
            payload = {}
        query_string = urlencode(payload)
        query_string = query_string.replace("%27", "%22")
        if query_string:
            query_string = "{}&timestamp={}".format(query_string, self._get_timestamp())
        else:
            query_string = "timestamp={}".format(self._get_timestamp())

        url = self.base_url + url_path + "?" + query_string + "&signature=" + self._hashing(query_string)
        logger.debug(f"{http_method} {url}")
        params = {"url": url, "params": {}}
        response = self._dispatch_request(http_method)(**params)
        if response.headers['Content-Type'] != 'application/json':
            raise ValueError(f"Unexpected content type: {response.headers['Content-Type']}, response: {response.text}")
        return response.json()

    def _send_public_request(self, url_path, payload=None):
        if payload is None:
            payload = {}
        query_string = urlencode(payload, True)
        url = self.base_url + url_path
        if query_string:
            url = url + "?" + query_string
        logger.debug(f"{url}")
        response = self._dispatch_request("GET")(url=url)
        if response.headers['Content-Type'] != 'application/json':
            raise ValueError(f"Unexpected content type: {response.headers['Content-Type']}, response: {response.text}")
        return response.json()

    def _get_timestamp(self):
        server_time = self._send_public_request('/fapi/v1/time')
        return server_time['serverTime']

    def get_precision(self, symbol):
        exchange_info = self._send_public_request('/fapi/v1/exchangeInfo')
        symbol_info = next((x for x in exchange_info['symbols'] if x['symbol'] == symbol), None)

        if symbol_info is None:
            raise ValueError(f"Symbol '{symbol}' not found in exchange information.")

        price_precision = symbol_info['pricePrecision']
        qty_precision = symbol_info['quantityPrecision']

        return int(price_precision), int(qty_precision)

    async def get_precision_and_tick_size(self, symbol):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/fapi/v1/exchangeInfo') as response:
                exchange_info = await response.json()

        symbol_info = next((x for x in exchange_info['symbols'] if x['symbol'] == symbol), None)

        if symbol_info is None:
            raise ValueError(f"Symbol '{symbol}' not found in exchange information.")

        price_precision = symbol_info['pricePrecision']
        qty_precision = symbol_info['quantityPrecision']
        tick_size = Decimal(next(f['tickSize'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'))
        max_price = Decimal(next(f['maxPrice'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'))

        return int(price_precision), int(qty_precision), tick_size, max_price

    def get_listen_key(self):
        response = self._send_signed_request('POST', '/fapi/v1/listenKey')
        return response['listenKey']

    def update_listen_key(self):
        self._send_signed_request('PUT', '/fapi/v1/listenKey')

    def delete_listen_key(self):
        self._send_signed_request('DELETE', '/fapi/v1/listenKey')

    def get_account_balance(self, asset: str):
        account_data = self._send_signed_request('GET', '/fapi/v2/account')

        if 'assets' not in account_data:
            return 0

        amount = [x for x in account_data['assets'] if x.get('asset') == asset][0]['walletBalance']
        return float(amount)

    def get_open_positions(self, symbol: str):
        positions = self._send_signed_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
        return positions

    def get_open_orders(self, symbol: str):
        orders = self._send_signed_request('GET', '/fapi/v1/openOrders', {'symbol': symbol})
        return orders

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self._send_signed_request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': leverage})
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")

    def set_margin_type(self, symbol: str, margin_type: str):
        try:
            self._send_signed_request('POST', '/fapi/v1/marginType', {'symbol': symbol, 'marginType': margin_type})
        except Exception as e:
            logger.error(f"Error setting margin type: {e}")

    def set_hedge_mode(self, position_mode: bool):
        try:
            self._send_signed_request('POST', '/fapi/v1/positionSide/dual', {'dualSidePosition': position_mode})
        except Exception as e:
            logger.error(f"Error setting hedge mode: {e}")

    def get_notional_brackets(self, symbol: str):
        try:
            result = self._send_signed_request('GET', '/fapi/v1/leverageBracket', {'symbol': symbol})
            return result[0]['brackets']
        except Exception as e:
            logger.error(f"Error getting notional brackets: {e}")

    async def renew_listen_key(self, listen_key: str):
        try:
            await self._send_signed_request('PUT', '/fapi/v1/listenKey', {'listenKey': listen_key})
        except Exception as e:
            logger.error(f"Error renewing listen key: {e}")

    async def keep_alive_listen_key(self):
        try:
            listen_key = await self.get_listen_key()
            while True:
                await asyncio.sleep(1800)  # 30 minutes
                await self.renew_listen_key(listen_key)
        except Exception as e:
            logger.error(f"Error keeping listen key alive: {e}")

# Example usage
if __name__ == "__main__":
    binance_client = BinanceClient(API_KEY, API_SECRET, BASE_URL)
    
    # Get account balance for a specific asset
    balance = binance_client.get_account_balance("BTC")
    print(f"BTC Balance: {balance}")

    # Get open positions for a symbol
    positions = binance_client.get_open_positions("BTCUSDT")
    print(f"Open Positions: {positions}")

    # Start the async loop for keeping listen key alive
    asyncio.run(binance_client.keep_alive_listen_key())