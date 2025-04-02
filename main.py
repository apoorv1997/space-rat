import sys
import random
from bot import Bot
import pygame
from ship import Ship
import numpy as np
import matplotlib.pyplot as plt

pygame.init()

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
GRID_SIZE = 30
CELL_SIZE = 20
MARGIN = 50
INFO_PANEL_HEIGHT = 150

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)

def draw_ship(screen, ship):
    """Draw the ship layout"""
    for i in range(ship.dimension):
        for j in range(ship.dimension):
            color = BLACK if ship.grid[i, j] == 1 else WHITE
            pygame.draw.rect(screen, color, 
                            (MARGIN + j * CELL_SIZE, 
                             MARGIN + i * CELL_SIZE, 
                             CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, 
                            (MARGIN + j * CELL_SIZE, 
                             MARGIN + i * CELL_SIZE, 
                             CELL_SIZE, CELL_SIZE), 1)

def draw_bot(screen, bot):
    """Draw the bot and its possible locations"""
    if not bot.localized:
        for (i, j) in bot.possible_locations:
            pygame.draw.rect(screen, CYAN, 
                           (MARGIN + j * CELL_SIZE + 5, 
                            MARGIN + i * CELL_SIZE + 5, 
                            CELL_SIZE - 10, CELL_SIZE - 10))
    
    if bot.current_location:
        i, j = bot.current_location
        pygame.draw.rect(screen, BLUE, 
                        (MARGIN + j * CELL_SIZE + 2, 
                         MARGIN + i * CELL_SIZE + 2, 
                         CELL_SIZE - 4, CELL_SIZE - 4))
        
        for idx, (pi, pj) in enumerate(bot.path):
            alpha = min(255, 50 + idx * 2)
            path_color = (0, 0, 255, alpha)
            path_surface = pygame.Surface((CELL_SIZE - 6, CELL_SIZE - 6), pygame.SRCALPHA)
            path_surface.fill(path_color)
            screen.blit(path_surface, 
                       (MARGIN + pj * CELL_SIZE + 3, 
                        MARGIN + pi * CELL_SIZE + 3))

def draw_rat(screen, rat_location, detected=False):
    """Draw the rat"""
    if rat_location:
        i, j = rat_location
        color = RED if detected else PURPLE
        pygame.draw.circle(screen, color, 
                          (MARGIN + j * CELL_SIZE + CELL_SIZE // 2, 
                           MARGIN + i * CELL_SIZE + CELL_SIZE // 2), 
                          CELL_SIZE // 3)

def draw_probability_heatmap(screen, bot):
    """Draw a heatmap of rat probabilities"""
    if not bot.localized:
        return
    
    max_prob = max(bot.rat_probabilities.values()) if bot.rat_probabilities else 1
    for (i, j), prob in bot.rat_probabilities.items():
        if prob > 0:
            intensity = int(255 * (prob / max_prob))
            color = (255, 255 - intensity, 0)
            pygame.draw.rect(screen, color, 
                           (MARGIN + j * CELL_SIZE + 8, 
                            MARGIN + i * CELL_SIZE + 8, 
                            CELL_SIZE - 16, CELL_SIZE - 16))

def draw_info_panel(screen, bot, font, alpha):
    """Draw the information panel at the bottom"""
    pygame.draw.rect(screen, WHITE, (0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 100, SCREEN_WIDTH, INFO_PANEL_HEIGHT))
    
    phase_text = font.render(f"Phase: {bot.phase}", True, BLACK)
    screen.blit(phase_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 120))
    
    action_text = font.render(f"Last Action: {bot.last_action}", True, BLACK)
    screen.blit(action_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 150))
    
    counts_text = font.render(
        f"Movements: {bot.actions_taken['movements']} | "
        f"Sensing: {bot.actions_taken['sensing']} | "
        f"Detection: {bot.actions_taken['detection']}", 
        True, BLACK)
    screen.blit(counts_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 180))
    
    alpha_text = font.render(f"Alpha (sensitivity): {alpha:.2f}", True, BLACK)
    screen.blit(alpha_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 210))
    
    legend = [
        ("Bot", BLUE),
        ("Rat", PURPLE),
        ("Caught Rat", RED),
        ("Possible Locations", CYAN),
        ("Rat Probability", ORANGE)
    ]
    
    for idx, (text, color) in enumerate(legend):
        pygame.draw.rect(screen, color, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 120 + idx * 25, 15, 15))
        legend_text = font.render(text, True, BLACK)
        screen.blit(legend_text, (SCREEN_WIDTH - 130, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 120 + idx * 25))

def main():
    """Main game loop"""
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Space Rats Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    alpha = 0.1
    ship = Ship()
    rat_location = random.choice(ship.open_cells)
    
    bot = Bot(ship, alpha=alpha)
    bot.rat_location = rat_location
    
    running = True
    paused = False
    speed = 5
    last_update_time = 0
    caught = False

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(20, speed + 1)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)
                elif event.key == pygame.K_r:
                    ship = Ship()
                    rat_location = random.choice(ship.open_cells)
                    bot = Bot(ship, alpha=alpha)
                    bot.rat_location = rat_location
                    caught = False
        
        if not paused and not caught and current_time - last_update_time > 1000 / speed:
            if not bot.localized:
                bot.localize_bot()
            else:
                caught = bot.track_rat()
                if caught:
                    print(f"Bot caught rat with {bot.actions_taken['movements']} movements!")
            
            last_update_time = current_time
        
        screen.fill(GRAY)
        draw_ship(screen, ship)
        
        if bot.localized:
            draw_probability_heatmap(screen, bot)
        
        draw_bot(screen, bot)
        draw_rat(screen, rat_location, bot.current_location == rat_location)
        draw_info_panel(screen, bot, font, alpha)
        
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "UP/DOWN - Increase/Decrease speed",
            "R - Reset simulation"
        ]
        
        for idx, text in enumerate(controls):
            control_text = font.render(text, True, BLACK)
            screen.blit(control_text, (SCREEN_WIDTH - 200, 20 + idx * 25))
        
        if caught:
            victory_text = font.render("RAT CAUGHT! Press R to reset", True, RED)
            screen.blit(victory_text, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 50))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()