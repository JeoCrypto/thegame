# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API credentials
API_KEY = os.getenv("BINANCE_API_KEY", "uKQtYQWg6wERwMP2gZmXGVoxW2IGx2iPomvUXLVd8hA0awdFyifgBjaQcezWaLiS")
API_SECRET = os.getenv("BINANCE_API_SECRET", "p6xRauRC2NmD3JMzrHKFDnUKln2xxQ8pyISDDnrShK4smI8vIsqgaWGfeDRCqvHh")
BASE_URL = "https://fapi.binance.com"

# Trading settings
SYMBOL = "BTCUSDT"
LEVERAGE = 10

# Screen dimensions for game visualization
WIDTH, HEIGHT = 1200, 800

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)  # Added ORANGE color

# Game object dimensions
PADDLE_HEIGHT = 10
PADDLE_WIDTH = 100
BALL_RADIUS = 8
