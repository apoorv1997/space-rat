import random
import numpy as np
import heapq
from bot import Bot
# from botcontroller import BotController


class Environment:
    """
    The Environment class handles time-stepping the simulation and updating the state of the ship.
    """
    def __init__(self, ship, alpha=0.3):
        self.ship = ship
        self.alpha = alpha
        self.size = 30
        self.possible_bot_locations = [(i, j) for i in range(self.size) for j in range(self.size) if ship[i][j] == 0]
        self.possible_rat_locations = self.possible_bot_locations.copy()
        self.belief_rat = {loc: 1 / len(self.possible_rat_locations) for loc in self.possible_rat_locations}
        self.bot_pos = np.random.choice(self.possible_bot_locations)
        self.rat_pos = np.random.choice(self.possible_rat_locations)
        self.movements = 0
        self.sensing_actions = 0
        self.detector_actions = 0


    def tick(self):
        """Algorithm to run the simulation to catch the rat"""
        # Phase 1: Identify the bot location
        while len(self.possible_bot_locations) > 1:
            blocked_neighbors = self.blocked_neighbors(*self.bot_pos)
            self.possible_bot_locations = [loc for loc in self.possible_bot_locations if self.blocked_neighbors(*loc) == blocked_neighbors]
            dx, dy = np.random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            self.move(dx, dy)

        print(f"Location Identified: {self.bot_pos}")

        # Phase 2: Track and Catch the Rat
        while not self.rat_detector():
            ping = self.rat_detector()
            self.update_rat_belief(ping)
            most_likely_rat_pos = max(self.belief_rat, key=self.belief_rat.get)
            path = self.a_star(self.bot_pos, most_likely_rat_pos)
            if path:
                next_pos = path[0]
                dx, dy = next_pos[0] - self.bot_pos[0], next_pos[1] - self.bot_pos[1]
                self.move(dx, dy)

        print(f"Rat caught at {self.bot_pos} in {self.movements} moves!")
        print(f"Sensing actions: {self.sensing_actions}, Detector actions: {self.detector_actions}")


    def blocked_neighbors(self, x, y):
        self.sensing_actions += 1
        return sum(self.ship[nx][ny] == 1 for nx, ny in self.ship.get_neighbors(x, y))

    def move(self, dx, dy):
        x, y = self.bot_pos
        if self.ship[x+dx][y+dy] == 0:
            self.bot_pos = (x+dx, y+dy)
            self.movements += 1
            return True
        return False
    
    def rat_detector(self):
        """ Uses the space rat detector to determine proximity probability. """
        self.detector_actions += 1
        bx, by = self.bot_pos
        rx, ry = self.rat_pos
        distance = abs(bx - rx) + abs(by - ry)  # Manhattan Distance
        probability = np.exp(-self.alpha * (distance - 1))
        return random.random() < probability
    
    def update_bot_location(self, blocked_neighbors):
        self.possible_bot_locations = [loc for loc in self.possible_bot_locations
                                       if self.blocked_neighbors(*loc) == blocked_neighbors]
        

    def update_rat_belief(self, ping):
        for loc in self.belief_rat.keys():
            distance = abs(self.bot_pos[0] - loc[0]) + abs(self.bot_pos[1] - loc[1])
            prob_ping_given_loc = np.exp(-self.alpha * (distance - 1)) if ping else 1 - np.exp(-self.alpha * (distance - 1))
            self.belief_rat[loc] *= prob_ping_given_loc
        total_prob = sum(self.belief_rat.values())
        for loc in self.belief_rat.keys():
            self.belief_rat[loc] /= total_prob

    def a_star(self, start, goal):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.ship.get_neighbors(*current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current
        return []
    
    def reset(self):
        """
        Reset the environment to its initial state
        """
        for cell in self.ship.get_on_fire_cells():
            self.ship.extinguish_cell(cell)
        self.bot.cell = self.initial_bot_cell
        self.bot_path = [self.bot.cell]
        self.ship.ignite_cell(self.initial_fire_cell)
        self.ship.history_fire_cells.clear()