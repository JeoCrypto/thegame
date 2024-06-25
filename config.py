# config.py
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = "uKQtYQWg6wERwMP2gZmXGVoxW2IGx2iPomvUXLVd8hA0awdFyifgBjaQcezWaLiS"
API_SECRET = "p6xRauRC2NmD3JMzrHKFDnUKln2xxQ8pyISDDnrShK4smI8vIsqgaWGfeDRCqvHh"
BASE_URL = "https://fapi.binance.com"

SYMBOL = "BTCUSDT"
LEVERAGE = 10

# Screen dimensions
WIDTH, HEIGHT = 1200, 800

# Colors
BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)

# Game object dimensions
PADDLE_HEIGHT = 10
PADDLE_WIDTH = 100
BALL_RADIUS = 8