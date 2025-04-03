from collections import defaultdict
import numpy as np
from heapq import heappush, heappop
import math
import random
from directions import Direction

class Bot:
    def __init__(self, ship, alpha=0.1):
        self.ship = ship
        self.alpha = alpha
        self.possible_locations = ship.open_cells.copy()
        self.rat_probabilities = defaultdict(float)
        self.initialize_rat_probabilities()
        
        # Ensure bot and rat start at different locations
        self.current_location = random.choice(self.ship.open_cells)
        self.rat_location = random.choice(self.ship.open_cells)
        while self.manhattan_distance(self.current_location, self.rat_location) < 5:
            self.rat_location = random.choice(self.ship.open_cells)
            
        self.actions_taken = {
            'movements': 0,
            'sensing': 0,
            'detection': 0
        }
        self.localized = False
        self.phase = "Localizing"
        self.last_action = "Initialized"
        self.last_ping = False
        self.path = [self.current_location]
        self.action_counter = 0
        self.visited = set([self.current_location])
        self.last_position = None
        
    def initialize_rat_probabilities(self):
        """Initialize uniform probability distribution for rat location"""
        prob = 1.0 / len(self.ship.open_cells)
        for cell in self.ship.open_cells:
            self.rat_probabilities[cell] = prob
    
    # def count_blocked_neighbors(self, cell):
    #     """Count how many of the 8 neighboring cells are blocked"""
    #     x, y = cell
    #     blocked = 0
    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             if dx == 0 and dy == 0:
    #                 continue
    #             nx, ny = x + dx, y + dy
    #             if (nx < 0 or nx >= self.ship.dimension or 
    #                 ny < 0 or ny >= self.ship.dimension or 
    #                 self.ship.grid[nx, ny] == 1):
    #                 blocked += 1
    #     return blocked
    
    def sense_blocked_neighbors(self):
        """Sense action - count blocked neighbors in current cell"""
        self.actions_taken['sensing'] += 1
        count = self.ship.get_cell(self.current_location.row, self.current_location.col).count_blocked_neighbors()
        self.last_action = f"Sensed {count} blocked neighbors"
        return count
    
    def detect_rat(self):
        """Space rat detector action - returns True if ping received"""
        self.actions_taken['detection'] += 1
        if self.current_location == self.rat_location:
            self.last_ping = True
            self.last_action = "Detected rat in same cell!"
            return True
        
        d = self.manhattan_distance(self.current_location, self.rat_location)
        ping_prob = math.exp(-self.alpha * (d - 1))
        ping_received = np.random.random() < ping_prob
        self.last_ping = ping_received
        self.last_action = f"Rat detector {'pinged' if ping_received else 'no ping'} (distance: {d})"
        return ping_received
    
    def manhattan_distance(self, cell1, cell2):
        """Calculate Manhattan distance between two cells"""
        return abs(cell1.row - cell2.row) + abs(cell1.col - cell2.col)
    
    def attempt_move(self, direction):
        """Attempt to move in one of 8 directions"""
        self.actions_taken['movements'] += 1
        x, y = self.current_location.row, self.current_location.col
        
        if direction == Direction.NORTH:
            new_x, new_y = x - 1, y
        elif direction == Direction.NORTHEAST:
            new_x, new_y = x - 1, y + 1
        elif direction == Direction.EAST:
            new_x, new_y = x, y + 1
        elif direction == Direction.SOUTHEAST:
            new_x, new_y = x + 1, y + 1
        elif direction == Direction.SOUTH:
            new_x, new_y = x + 1, y
        elif direction == Direction.SOUTHWEST:
            new_x, new_y = x + 1, y - 1
        elif direction == Direction.WEST:
            new_x, new_y = x, y - 1
        elif direction == Direction.NORTHWEST:
            new_x, new_y = x - 1, y - 1
        
        # Check if move is valid
        if (0 <= new_x < self.ship.dimension and 
            0 <= new_y < self.ship.dimension and 
            self.ship.get_cell(new_x, new_y).is_open()):
            self.current_location = (new_x, new_y)  # This is the critical update
            self.path.append(self.current_location)
            self.visited.add(self.current_location)
            return True
        return False
    
    def most_common_open_direction(self, possible_locations):
        """Find direction most commonly open among possible locations"""
        direction_counts = {d: 0 for d in Direction}
        
        for cell in possible_locations:
            x, y = cell.row, cell.col
            if x > 0 and self.ship.get_cell(x-1, y).is_open():
                direction_counts[Direction.NORTH] += 1
            if x > 0 and y < self.ship.dimension-1 and self.ship.get_cell(x-1, y+1).is_open():
                direction_counts[Direction.NORTHEAST] += 1
            if y < self.ship.dimension-1 and self.ship.get_cell(x, y+1).is_open():
                direction_counts[Direction.EAST] += 1
            if x < self.ship.dimension-1 and y < self.ship.dimension-1 and self.ship.get_cell(x+1, y+1).is_open():
                direction_counts[Direction.SOUTHEAST] += 1
            if x < self.ship.dimension-1 and self.ship.get_cell(x+1, y).is_open():
                direction_counts[Direction.SOUTH] += 1
            if x < self.ship.dimension-1 and y > 0 and self.ship.get_cell(x+1, y-1).is_open():
                direction_counts[Direction.SOUTHWEST] += 1
            if y > 0 and self.ship.get_cell(x, y-1).is_open():
                direction_counts[Direction.WEST] += 1
            if x > 0 and y > 0 and self.ship.get_cell(x-1, y-1).is_open():
                direction_counts[Direction.NORTHWEST] += 1
        
        max_count = max(direction_counts.values())
        best_directions = [d for d, count in direction_counts.items() if count == max_count]
        return random.choice(best_directions)
    
    def update_location_knowledge(self, sensed_blocked):
        """Update possible locations based on sensed blocked neighbors"""
        new_possible = []
        for cell in self.possible_locations:
            if cell.count_blocked_neighbors() == sensed_blocked:
                new_possible.append(cell)
            # if self.count_blocked_neighbors(cell) == sensed_blocked:
            #     new_possible.append(cell)
        
        if new_possible:
            self.possible_locations = new_possible
        else:
            self.last_action = "Warning: No cells matched sensed blocked count!"
    
    def update_location_after_move(self, direction, move_success):
        """Update possible locations after attempted move"""
        new_possible = []
        for cell in self.possible_locations:
            can_move = False
            if direction == Direction.NORTH:
                can_move = cell.row > 0 and self.ship.get_cell(cell.row, cell.col).is_open()
            elif direction == Direction.NORTHEAST:
                can_move = (cell.row > 0 and cell.col < self.ship.dimension-1 and 
                           self.ship.get_cell(cell.row-1, cell.col+1).is_open())
            elif direction == Direction.EAST:
                can_move = cell.col < self.ship.dimension-1 and self.ship.get_cell(cell.row, cell.col+1).is_open()
            elif direction == Direction.SOUTHEAST:
                can_move = (cell.row < self.ship.dimension-1 and cell.col < self.ship.dimension-1 and 
                           self.ship.get_cell(cell.row+1, cell.col+1).is_open())
            elif direction == Direction.SOUTH:
                can_move = cell.row < self.ship.dimension-1 and self.ship.get_cell(cell.row+1, cell.col).is_open()
            elif direction == Direction.SOUTHWEST:
                can_move = (cell.row < self.ship.dimension-1 and cell.col > 0 and 
                           self.ship.get_cell(cell.row+1, cell.col-1).is_open())
            elif direction == Direction.WEST:
                can_move = cell.col > 0 and self.ship.get_cell(cell.row, cell.col-1).is_open()
            elif direction == Direction.NORTHWEST:
                can_move = cell.row > 0 and cell.col > 0 and self.ship.get_cell(cell.row-1, cell.col-1).is_open()
            
            if (move_success and can_move) or (not move_success and not can_move):
                new_possible.append(cell)
        
        if new_possible:
            self.possible_locations = new_possible
        else:
            self.last_action = "Warning: No possible locations after move!"
    
    def update_rat_probabilities(self, ping_received):
        """Improved probability update with exploration incentive"""
        total_prob = 0.0
        new_probs = defaultdict(float)
        
        for cell in self.rat_probabilities:
            d = self.manhattan_distance(self.current_location, cell)
            ping_prob = math.exp(-self.alpha * (d - 1))
            
            if ping_received:
                new_probs[cell] = self.rat_probabilities[cell] * ping_prob
            else:
                new_probs[cell] = self.rat_probabilities[cell] * (1 - ping_prob)
            
            if cell not in self.visited:
                new_probs[cell] *= 1.2
            
            total_prob += new_probs[cell]
        
        if total_prob > 0:
            for cell in new_probs:
                new_probs[cell] /= total_prob
        self.rat_probabilities = new_probs
    
    def get_most_probable_rat_location(self):
        """Return cell with highest probability of containing the rat"""
        return max(self.rat_probabilities.items(), key=lambda x: x[1])[0]
    
    def find_shortest_path(self, start, target):
        """A* pathfinding algorithm to find shortest path to target"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heappop(frontier)

            if current == target:
                break

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.ship.dimension and 
                    0 <= neighbor[1] < self.ship.dimension and 
                    self.ship.grid[neighbor[0], neighbor[1]] == 0):
                    new_cost = cost_so_far[current] + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, target)
                        heappush(frontier, (priority, neighbor))
                        came_from[neighbor] = current

        path = []
        current = target
        while current != start:
            path.append(current)
            current = came_from.get(current, None)
            if current is None:
                return []
        path.reverse()
        return path
    
    def move_directly_toward_target(self, target):
        """Move directly toward target without pathfinding"""
        dx = target[0] - self.current_location[0]
        dy = target[1] - self.current_location[1]
        
        primary_direction = None
        if abs(dx) > abs(dy):
            primary_direction = Direction.SOUTH if dx > 0 else Direction.NORTH
        else:
            primary_direction = Direction.EAST if dy > 0 else Direction.WEST
        
        if self.attempt_move(primary_direction):
            self.last_action = f"Moved {primary_direction.name} toward rat (direct)"
            return True
        
        diagonal_directions = []
        if dx > 0 and dy > 0:
            diagonal_directions = [Direction.SOUTHEAST, Direction.SOUTH, Direction.EAST]
        elif dx > 0 and dy < 0:
            diagonal_directions = [Direction.SOUTHWEST, Direction.SOUTH, Direction.WEST]
        elif dx < 0 and dy > 0:
            diagonal_directions = [Direction.NORTHEAST, Direction.NORTH, Direction.EAST]
        elif dx < 0 and dy < 0:
            diagonal_directions = [Direction.NORTHWEST, Direction.NORTH, Direction.WEST]
        
        for direction in diagonal_directions:
            if self.attempt_move(direction):
                self.last_action = f"Moved {direction.name} toward rat (direct fallback)"
                return True
        
        for direction in Direction:
            if self.attempt_move(direction):
                self.last_action = f"Moved {direction.name} (random fallback)"
                return True
        
        self.last_action = "Failed to move - completely stuck!"
        return False
    
    def move_toward_rat(self):
        """Improved movement using pathfinding and probability heatmap"""
        target = self.get_most_probable_rat_location()
        
        path = self.find_shortest_path(self.current_location, target)
        
        if path and len(path) > 0:
            next_cell = path[0]
            
            if len(self.path) > 1 and next_cell == self.path[-2]:
                if len(path) > 1:
                    next_cell = path[1]
                else:
                    return self.move_directly_toward_target(target)
            
            dx = next_cell[0] - self.current_location[0]
            dy = next_cell[1] - self.current_location[1]
            
            direction = None
            if dx == -1 and dy == 0:
                direction = Direction.NORTH
            elif dx == -1 and dy == 1:
                direction = Direction.NORTHEAST
            elif dx == 0 and dy == 1:
                direction = Direction.EAST
            elif dx == 1 and dy == 1:
                direction = Direction.SOUTHEAST
            elif dx == 1 and dy == 0:
                direction = Direction.SOUTH
            elif dx == 1 and dy == -1:
                direction = Direction.SOUTHWEST
            elif dx == 0 and dy == -1:
                direction = Direction.WEST
            elif dx == -1 and dy == -1:
                direction = Direction.NORTHWEST
            
            if direction is not None and self.attempt_move(direction):
                self.last_action = f"Moved {direction.name} toward rat"
                return True
        
        return self.move_directly_toward_target(target)
    
    def localize_bot(self):
        """Phase 1: Determine bot's current location"""
        if len(self.possible_locations) == 1:
            # Only update the possible locations, don't reset current_location
            self.possible_locations = [self.current_location]
            self.localized = True
            self.phase = "Tracking Rat"
            self.last_action = "Localization complete!"
            return True
        
        if self.action_counter % 2 == 0:
            sensed_blocked = self.sense_blocked_neighbors()
            self.update_location_knowledge(sensed_blocked)
            self.last_action = f"Sensed {sensed_blocked} blocked neighbors, {len(self.possible_locations)} possible locations"
        else:
            direction = self.most_common_open_direction(self.possible_locations)
            move_success = self.attempt_move(direction)
            self.update_location_after_move(direction, move_success)
            self.last_action = f"Moved {direction.name} - {'success' if move_success else 'failed'}, {len(self.possible_locations)} possible locations"
        
        self.action_counter += 1
        
        if len(self.possible_locations) == 1:
            # Only update the possible locations, don't reset current_location
            self.possible_locations = [self.current_location]
            self.localized = True
            self.phase = "Tracking Rat"
            self.last_action = "Localization complete!"
            return True
        
        if self.action_counter > 100 and len(self.possible_locations) > 1:
            # Choose the cell closest to our current path
            best_cell = min(self.possible_locations,
                        key=lambda cell: self.manhattan_distance(cell, self.current_location))
            self.possible_locations = [best_cell]
            self.localized = True
            self.phase = "Tracking Rat"
            self.last_action = "Localization complete (failsafe)"
            return True
        
        return False
    
    def track_rat(self):
        """Improved tracking with optimized action sequence"""
        if self.current_location == self.rat_location:
            self.phase = "Rat Caught!"
            self.last_action = "Successfully caught the rat!"
            return True
        
        if self.action_counter % 3 != 0:
            if self.move_toward_rat():
                self.visited.add(self.current_location)
        else:
            ping_received = self.detect_rat()
            self.update_rat_probabilities(ping_received)
            self.last_action = f"Detector {'pinged' if ping_received else 'no ping'}"
        
        self.action_counter += 1
        return self.current_location == self.rat_location

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