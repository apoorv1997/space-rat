from collections import defaultdict
import random

from cell import Cell
from directions import Direction


class InterestingBot:
    def __init__(self, ship, alpha=0.5):
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

    def manhattan_distance(self, cell1, cell2):
        """Calculate Manhattan distance between two cells"""
        return abs(cell1.row - cell2.row) + abs(cell1.col - cell2.col)

    def sense_blocked_neighbors(self):
        """Sense action - count blocked neighbors in current cell"""
        self.actions_taken['sensing'] += 1
        count = self.ship.get_cell(self.current_location.row, self.current_location.col).count_blocked_neighbors()
        self.last_action = f"Sensed {count} blocked neighbors"
        return count
    
    def update_belief_state(self):
        """Bayesian update of rat location probabilities incorporating movement model"""
        # Create temporary copy of probabilities
        new_probs = defaultdict(float)
        total = 0.0
        
        # Markov transition - rat could have moved to adjacent cells
        for cell, prob in self.rat_probabilities.items():
            if prob < 0.001:  # Skip very low probabilities
                continue
                
            # Rat can stay or move to adjacent cells
            neighbors = self.get_valid_neighbors(cell)
            spread_prob = prob / (len(neighbors) + 1)  # +1 for staying
            
            # Distribute probability
            new_probs[cell] += spread_prob  # Staying probability
            for neighbor in neighbors:
                new_probs[neighbor] += spread_prob  # Moving probability
        
        # Normalize
        total_prob = sum(new_probs.values())
        if total_prob > 0:
            self.rat_probabilities = {cell: p/total_prob for cell, p in new_probs.items()}

    def get_valid_neighbors(self, cell):
        """Get all accessible neighboring cells"""
        x, y = cell
        neighbors = []
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.ship.dimension and 
                0 <= ny < self.ship.dimension and 
                self.ship.grid[nx, ny] == 0):
                neighbors.append((nx, ny))
        return neighbors
    
    def value_iteration(self, horizon=5):
        """MDP-based decision making with finite horizon"""
        # Initialize value function
        values = {cell: 0 for cell in self.ship.open_cells}
        
        for _ in range(horizon):
            new_values = {}
            for cell in self.ship.open_cells:
                # Q-values for each possible action
                q_values = []
                for direction in Direction:
                    # Get successor states
                    successor = self.get_successor(cell, direction)
                    if successor == cell:  # Didn't move
                        reward = -0.1  # Small penalty for not moving
                    else:
                        # Reward based on probability of finding rat
                        reward = self.rat_probabilities.get(successor, 0) * 10
                    
                    # Value is immediate reward + discounted future value
                    q_values.append(reward + 0.9 * values[successor])
                
                # Value is max over all possible actions
                new_values[cell] = max(q_values) if q_values else 0
            
            values = new_values
        
        # Return optimal policy (direction with highest Q-value)
        return self.get_optimal_action(self.current_location, values)
    
    def get_successor(self, cell, direction):
        """Get the resulting cell from taking an action"""
        x, y = cell.row, cell.col
        if direction == Direction.NORTH:
            new_x, new_y = x - 1, y
        elif direction == Direction.NORTHEAST:
            new_x, new_y = x - 1, y + 1
        # ... (other directions same as before)
        
        if (0 <= new_x < self.ship.dimension and 
            0 <= new_y < self.ship.dimension and 
            self.ship.get_cell(new_x, new_y).is_open()):
            return Cell(new_x, new_y)
        return cell
    

    def get_optimal_action(self, cell, values):
        """Choose best action based on value function"""
        best_value = -float('inf')
        best_direction = None
        
        for direction in Direction:
            successor = self.get_successor(cell, direction)
            value = values[successor]
            if value > best_value:
                best_value = value
                best_direction = direction
                
        return best_direction
    
    def track_rat(self):
        """Enhanced tracking using MDP and Bayesian updates"""
        if self.current_location == self.rat_location:
            self.phase = "Rat Caught!"
            self.last_action = "Successfully caught the rat!"
            return True
        
        # Bayesian belief update
        if self.action_counter % 3 == 0:
            ping_received = self.detect_rat()
            self.update_rat_probabilities(ping_received)
            self.update_belief_state()  # Markov transition update
            self.last_action = f"Detector {'pinged' if ping_received else 'no ping'}"
        else:
            # MDP-based decision making
            optimal_direction = self.value_iteration()
            if optimal_direction and self.attempt_move(optimal_direction):
                self.last_action = f"Moved {optimal_direction.name} (MDP-optimized)"
            else:
                # Fallback to probability-guided movement
                self.probability_guided_movement()
        
        self.action_counter += 1
        return self.current_location == self.rat_location
    
    def update_location_knowledge(self, sensed_blocked):
        """Update possible locations based on sensed blocked neighbors count"""
        new_possible = []
        current_blocked = self.ship.get_cell(self.current_location.row, self.current_location.col).count_blocked_neighbors()
        
        for cell in self.possible_locations:
            # Only keep cells that match both:
            # 1. The sensed blocked count AND
            # 2. Could have resulted in our current observation
            if (self.ship.get_cell(cell.row, cell.col).count_blocked_neighbors() == sensed_blocked and
                self.ship.get_cell(cell.row, cell.col).count_blocked_neighbors() == current_blocked):
                new_possible.append(cell)
        
        if new_possible:
            self.possible_locations = new_possible
        else:
            # Fallback - keep previous possibilities if none match (shouldn't happen)
            self.last_action = "Warning: No cells matched sensing data!"
    
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
            self.last_position = self.current_location
            self.current_location = Cell(new_x, new_y)
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
    
    def localize_bot(self):
        """Phase 1: Determine bot's current location"""
        if len(self.possible_locations) == 1:
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
            self.possible_locations = [self.current_location]
            self.localized = True
            self.phase = "Tracking Rat"
            self.last_action = "Localization complete!"
            return True
        
        if self.action_counter > 100 and len(self.possible_locations) > 1:
            best_cell = min(self.possible_locations,
                          key=lambda cell: self.manhattan_distance(cell, self.current_location))
            self.possible_locations = [best_cell]
            self.localized = True
            self.phase = "Tracking Rat"
            self.last_action = "Localization complete (failsafe)"
            return True
        
        return False
    

    def probability_guided_movement(self):
        """Movement based on probability heatmap when pathfinding fails"""
        target = self.get_most_probable_rat_location()
        dx = target[0] - self.current_location[0]
        dy = target[1] - self.current_location[1]
        
        # Create priority list of directions based on probability gradient
        direction_priority = []
        if dx != 0 or dy != 0:
            if abs(dx) > abs(dy):
                primary_axis = [Direction.SOUTH if dx > 0 else Direction.NORTH]
                secondary_axis = [Direction.EAST if dy > 0 else Direction.WEST]
            else:
                primary_axis = [Direction.EAST if dy > 0 else Direction.WEST]
                secondary_axis = [Direction.SOUTH if dx > 0 else Direction.NORTH]
            
            # Combine with diagonals
            diagonal_directions = []
            if dx > 0 and dy > 0:
                diagonal_directions = [Direction.SOUTHEAST]
            elif dx > 0 and dy < 0:
                diagonal_directions = [Direction.SOUTHWEST]
            elif dx < 0 and dy > 0:
                diagonal_directions = [Direction.NORTHEAST]
            elif dx < 0 and dy < 0:
                diagonal_directions = [Direction.NORTHWEST]
            
            direction_priority = diagonal_directions + primary_axis + secondary_axis
        
        # Try all possible directions in order of priority
        for direction in direction_priority:
            if self.attempt_move(direction):
                self.last_action = f"Moved {direction.name} toward rat (probability guided)"
                return True
        
        # If all else fails, try any possible move
        for direction in Direction:
            if self.attempt_move(direction):
                self.last_action = f"Moved {direction.name} (random fallback)"
                return True
        
        self.last_action = "Failed to move - completely stuck!"
        return False
    

    def get_optimal_action(self, cell, values):
        """Choose best action based on value function"""
        best_value = -float('inf')
        best_direction = None
        
        for direction in Direction:
            successor = self.get_successor(cell, direction)
            value = values[successor]
            if value > best_value:
                best_value = value
                best_direction = direction
                
        return best_direction
    

    def get_successor(self, cell, direction):
        """Get the resulting cell from taking an action"""
        x, y = cell.row, cell.col
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
        
        if (0 <= new_x < self.ship.dimension and 
            0 <= new_y < self.ship.dimension and 
            self.ship.get_cell(new_x, new_y).is_open()):
            return Cell(new_x, new_y)
        return cell
    

    def value_iteration(self, horizon=3):
        """MDP-based decision making with finite horizon"""
        # Initialize value function
        values = {cell: 0 for cell in self.ship.open_cells}
        
        for _ in range(horizon):
            new_values = {}
            for cell in self.ship.open_cells:
                # Q-values for each possible action
                q_values = []
                for direction in Direction:
                    # Get successor states
                    successor = self.get_successor(cell, direction)
                    if successor == cell:  # Didn't move
                        reward = -0.1  # Small penalty for not moving
                    else:
                        # Reward based on probability of finding rat
                        reward = self.rat_probabilities.get(successor, 0) * 10
                    
                    # Value is immediate reward + discounted future value
                    q_values.append(reward + 0.9 * values[successor])
                
                # Value is max over all possible actions
                new_values[cell] = max(q_values) if q_values else 0
            
            values = new_values
        
        # Return optimal policy (direction with highest Q-value)
        return self.get_optimal_action(self.current_location, values)