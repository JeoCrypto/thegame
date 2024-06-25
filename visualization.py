# visualization.py

# visualization.py
import pygame
from config import PADDLE_HEIGHT, PADDLE_WIDTH, BALL_RADIUS, HEIGHT, WIDTH, BLACK, WHITE, RED, GREEN, BLUE

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('CryptoWolf - AI Breakout')
    return screen

def draw_paddle(screen, paddle_x):
    pygame.draw.rect(screen, WHITE, (paddle_x, HEIGHT - 20, 100, 10))

def draw_ball(screen, ball_x, ball_y):
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), 8)

def draw_blocks(screen, blocks):
    for block in blocks:
        pygame.draw.rect(screen, RED, block)

def draw_score(screen, score):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))

def update_display(screen, paddle_x, ball_x, ball_y, blocks, info):
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (paddle_x, HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), BALL_RADIUS)
    
    for block in blocks:
        pygame.draw.rect(screen, block['color'], block['rect'])
    
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {info['score']}", True, WHITE)
    balance_text = font.render(f"Balance: ${info['balance']:.2f}", True, WHITE)
    pnl_text = font.render(f"Unrealized P/L: ${info.get('unrealized_pnl', 0.0):.2f}", True, WHITE)
    position_long_text = font.render(f"Long Position: {info['position_long']}", True, WHITE)
    position_short_text = font.render(f"Short Position: {info['position_short']}", True, WHITE)
    position_size_long_text = font.render(f"Long Position Size: {info['position_size_long']:.4f}", True, WHITE)
    position_size_short_text = font.render(f"Short Position Size: {info['position_size_short']:.4f}", True, WHITE)
    entry_price_long_text = font.render(f"Long Entry Price: ${info['entry_price_long']:.2f}", True, WHITE)
    entry_price_short_text = font.render(f"Short Entry Price: ${info['entry_price_short']:.2f}", True, WHITE)
    current_price_text = font.render(f"Current Price: ${info['current_price']:.2f}", True, WHITE)
    
    screen.blit(score_text, (10, 10))
    screen.blit(balance_text, (10, 50))
    screen.blit(pnl_text, (10, 90))
    screen.blit(position_long_text, (10, 130))
    screen.blit(position_short_text, (10, 170))
    screen.blit(position_size_long_text, (10, 210))
    screen.blit(position_size_short_text, (10, 250))
    screen.blit(entry_price_long_text, (10, 290))
    screen.blit(entry_price_short_text, (10, 330))
    screen.blit(current_price_text, (10, 370))
    
    # Display last 5 trades
    for i, trade in enumerate(info['trade_history'][-5:]):
        trade_text = font.render(f"Trade {i+1}: Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}, PNL: ${trade['pnl']:.2f}", True, WHITE)
        screen.blit(trade_text, (10, 410 + i * 40))
    
    pygame.display.flip()

