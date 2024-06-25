# visualization.py
import pygame
from config import WIDTH, HEIGHT, BLACK, WHITE, GREEN, RED

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Trading Bot Simulation')
    return screen

def draw_chart(screen, data, color):
    if len(data) < 2:
        return
    scaled_data = [(i, HEIGHT - (d - min(data)) / (max(data) - min(data)) * HEIGHT) for i, d in enumerate(data)]
    pygame.draw.lines(screen, color, False, scaled_data, 2)

def draw_text(screen, text, position, color=WHITE, font_size=36):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def update_display(screen, price_data, balance, position, episode):
    screen.fill(BLACK)
    draw_chart(screen, price_data, GREEN)
    draw_text(screen, f"Episode: {episode}", (20, 20))
    draw_text(screen, f"Balance: {balance:.2f}", (20, 60))
    draw_text(screen, f"Position: {position}", (20, 100))
    pygame.display.flip()