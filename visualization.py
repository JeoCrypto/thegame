import pygame
import numpy as np
from trading_env import TradingEnv
from config import PADDLE_HEIGHT, PADDLE_WIDTH, BALL_RADIUS, HEIGHT, WIDTH, BLACK, WHITE, RED, GREEN, BLUE

class GameVisualizer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CryptoWolf - AI Breakout")
        self.font = pygame.font.Font(None, 24)

    def render(self, env: TradingEnv):
        self.screen.fill(BLACK)  # Black background
        
        # Draw blocks
        for block in env.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])

        # Draw paddle
        paddle_color = GREEN if env.position > 0 else RED if env.position < 0 else WHITE
        pygame.draw.rect(self.screen, paddle_color, 
                         (env.paddle_x, self.height - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))

        # Draw ball
        pygame.draw.circle(self.screen, WHITE, (int(env.ball_x), int(env.ball_y)), BALL_RADIUS)

        # Draw game info
        info_text = [
            f"Balance: ${env.balance:.2f}",
            f"Position: {env.position:.4f}",
            f"Price: ${env.data_fetcher.current_price:.2f}",
            f"ADX: {env.data_fetcher.adx:.2f}",
            f"Game Speed: {env.data_fetcher.get_game_speed():.2f}x",
            f"PnL: ${env.calculate_pnl():.2f}",
            f"Open Orders: {len(env.open_orders)}",
            f"Last Action: {env.last_action}"
        ]
        for i, text in enumerate(info_text):
            surf = self.font.render(text, True, WHITE)
            self.screen.blit(surf, (10, 10 + i * 30))

        # Draw recent trades
        trade_text = "Recent Trades:"
        self.screen.blit(self.font.render(trade_text, True, WHITE), (10, 300))
        for i, trade in enumerate(env.trade_history[-5:]):
            trade_info = f"{trade['action']} @ ${trade['price']:.2f} - PnL: ${trade['pnl']:.2f}"
            color = GREEN if trade['pnl'] > 0 else RED
            self.screen.blit(self.font.render(trade_info, True, color), (10, 330 + i * 30))

        pygame.display.flip()

    def check_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def close(self):
        pygame.quit()