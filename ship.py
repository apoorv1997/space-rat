import random
import numpy as np
from cell import Cell


class Ship:
    """
    A class representing a ship in a grid. The ship is represented as a 2D grid of cell objects,
    where each cell can be open, closed, or ignited.
    """

    def __init__(self, dimension=30):
        self.dimension = dimension
        self.initial_cell = None
        self.grid = np.zeros((dimension, dimension), dtype=int)
        self._initialize_walls()
        self.bot_position = self._get_random_position()
        self.rat_position = self._get_random_position()



    def _initialize_walls(self):
        """ Set up the outer walls of the ship as blocked cells (-1). """
        self.grid[0, :] = -1
        self.grid[:, 0] = -1
        self.grid[-1, :] = -1
        self.grid[:, -1] = -1


    def _get_random_position(self):
        """ Get a random position inside the ship that is not blocked. """
        while True:
            x, y = random.randint(1, self.dimension-2), random.randint(1, self.dimension-2)
            if self.grid[x, y] == 0:
                return (x, y)
    
    def blocked_neighbors(self, position):
        """ Returns the number of blocked neighboring cells around a given position. """
        x, y = position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                     (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
        return sum(1 for nx, ny in neighbors if self.grid[nx, ny] == -1)

    def rat_detector(self):
        """ Uses the space rat detector to determine proximity probability. """
        bx, by = self.bot_position
        rx, ry = self.rat_position
        distance = abs(bx - rx) + abs(by - ry)  # Manhattan Distance
        probability = np.exp(-self.alpha * (distance - 1))
        return random.random() < probability
    
    def get_cell(self, row, col):
        """Return cell object at given coordinates"""
        if self.cell_in_bounds(row, col):
            return self.grid[row][col]
        raise ValueError("Invalid cell coordinates")
    

    def cell_in_bounds(self, row, col):
        """Check if cell coordinates are within grid bounds"""
        return 0 <= row < self.dimension and 0 <= col < self.dimension