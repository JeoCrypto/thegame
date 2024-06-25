# visualization.py

# visualization.py
import pygame
from config import WIDTH, HEIGHT, BLACK, WHITE, RED, GREEN, BLUE

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('AI Breakout Training')
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

def update_display(screen, paddle_x, ball_x, ball_y, blocks, score):
    screen.fill(BLACK)
    draw_paddle(screen, paddle_x)
    draw_ball(screen, ball_x, ball_y)
    draw_blocks(screen, blocks)
    draw_score(screen, score)
    pygame.display.flip()