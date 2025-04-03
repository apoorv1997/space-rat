import os
import pygame


# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
GRID_SIZE = 30
CELL_SIZE = 20
MARGIN = 50
INFO_PANEL_HEIGHT = 150

# Colors
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

class Visualizer:
    """A class for visualizing the ship grid and the robot's actions"""

    def __init__(self, ship, cell_size=30, env=None):
        self.ship = ship
        self.cell_size = cell_size
        self.width = ship.dimension * cell_size
        self.height = ship.dimension * cell_size
        self.env = env
        self.bot = env.bot if env is not None else None

        # Initializing the pygame window
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Ship Grid Visualization")

        # Load image assets for fire and robot emojis.
        # assets_path = os.path.join(os.path.dirname(__file__), 'assets')
        # self.fire_image = pygame.image.load(os.path.join(assets_path, "fire.png")).convert_alpha()
        # self.robot_image = pygame.image.load(os.path.join(assets_path, "walle.png")).convert_alpha()

        # Scale images to size relative to cell size
        # desired_size = int(self.cell_size * 0.6)
        # self.fire_image = pygame.transform.smoothscale(self.fire_image, (desired_size, desired_size))
        # self.robot_image = pygame.transform.smoothscale(self.robot_image, (desired_size, desired_size))


    def draw_ship(screen, ship):
        """Draw the ship layout"""
        for i in range(ship.dimension):
            for j in range(ship.dimension):
                color = BLACK if ship.grid[i, j] == 1 else WHITE
                pygame.draw.rect(screen, color, (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, GRAY, (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        pygame.display.flip()  # update display


    def draw_static_grid(self):
        """
        Display static representation of ship grid,
        overlaying robot's path (blue dots) and every cell that was ignited (fire image)
        """
        self.draw_grid()
        if self.env is not None:
            # Overlay ignited cells
            red_fire_image = self.fire_image.copy()

            # for cell in self.ship.history_fire_cells:
            #     # Draw fire image if cell was ignited.
            #     row, col = cell.row, cell.col
            #     cell_obj = self.ship.get_cell(row, col)
            #     if not cell_obj.on_fire:
            #         ew, eh = red_fire_image.get_size()
            #         ex = col * self.cell_size + (self.cell_size - ew) // 2
            #         ey = row * self.cell_size + (self.cell_size - eh) // 2
            #         self.screen.blit(red_fire_image, (ex, ey))
            # Draw robot's path as small blue circles
            for cell in self.env.bot_path:
                center = (cell.col * self.cell_size + self.cell_size // 2,
                          cell.row * self.cell_size + self.cell_size // 2)
                pygame.draw.circle(self.screen, (0, 0, 255), center, self.cell_size // 8)
            # Draw robot's final position
            if self.env.bot:
                row, col = self.env.bot.cell.row, self.env.bot.cell.col
                ew, eh = self.robot_image.get_size()
                ex = col * self.cell_size + (self.cell_size - ew) // 2
                ey = row * self.cell_size + (self.cell_size - eh) // 2
                self.screen.blit(self.robot_image, (ex, ey))
        pygame.display.flip()  # update display

        while True:
            # wait for user to close the window
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                break
        pygame.quit()

    def draw_bot(screen, bot):
        """Draw the bot and its possible locations"""
        # Draw possible locations during localization
        if not bot.localized:
            for (i, j) in bot.possible_locations:
                pygame.draw.rect(screen, CYAN, (MARGIN + j * CELL_SIZE + 5, MARGIN + i * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))
        
        # Draw current location
        if bot.current_location:
            i, j = bot.current_location
            pygame.draw.rect(screen, BLUE, (MARGIN + j * CELL_SIZE + 2, MARGIN + i * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4))
            
            # Draw path
            for idx, (pi, pj) in enumerate(bot.path):
                alpha = min(255, 50 + idx * 2)  # Fade out older path segments
                path_color = (0, 0, 255, alpha)
                path_surface = pygame.Surface((CELL_SIZE - 6, CELL_SIZE - 6), pygame.SRCALPHA)
                path_surface.fill(path_color)
                screen.blit(path_surface, (MARGIN + pj * CELL_SIZE + 3, MARGIN + pi * CELL_SIZE + 3))


    def draw_rat(screen, rat_location, detected=False):
        """Draw the rat"""
        if rat_location:
            i, j = rat_location
            color = RED if detected else PURPLE
            pygame.draw.circle(screen, color, (MARGIN + j * CELL_SIZE + CELL_SIZE // 2, MARGIN + i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)


    def draw_probability_heatmap(screen, bot):
        """Draw a heatmap of rat probabilities"""
        if not bot.localized:
            return
        
        max_prob = max(bot.rat_probabilities.values()) if bot.rat_probabilities else 1
        for (i, j), prob in bot.rat_probabilities.items():
            if prob > 0:
                intensity = int(255 * (prob / max_prob))
                color = (255, 255 - intensity, 0)  # Yellow to red gradient
                pygame.draw.rect(screen, color, (MARGIN + j * CELL_SIZE + 8, MARGIN + i * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16))      
    

    def draw_info_panel(screen, bot, font, alpha):
        """Draw the information panel at the bottom"""
        pygame.draw.rect(screen, WHITE, (0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT))
        
        # Draw phase information
        phase_text = font.render(f"Phase: {bot.phase}", True, BLACK)
        screen.blit(phase_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 20))
        
        # Draw last action
        action_text = font.render(f"Last Action: {bot.last_action}", True, BLACK)
        screen.blit(action_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 50))
        
        # Draw action counts
        counts_text = font.render(
            f"Movements: {bot.actions_taken['movements']} | "
            f"Sensing: {bot.actions_taken['sensing']} | "
            f"Detection: {bot.actions_taken['detection']}", 
            True, BLACK)
        screen.blit(counts_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 80))
        
        # Draw alpha value
        alpha_text = font.render(f"Alpha (sensitivity): {alpha:.2f}", True, BLACK)
        screen.blit(alpha_text, (20, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 110))
        
        # Draw legend
        legend = [
            ("Bot", BLUE),
            ("Rat", PURPLE),
            ("Caught Rat", RED),
            ("Possible Locations", CYAN),
            ("Rat Probability", ORANGE)
        ]
        
        for idx, (text, color) in enumerate(legend):
            pygame.draw.rect(screen, color, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 20 + idx * 25, 15, 15))
            legend_text = font.render(text, True, BLACK)
            screen.blit(legend_text, (SCREEN_WIDTH - 130, SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 20 + idx * 25))