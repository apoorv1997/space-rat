import random
import numpy as np


class Ship:
    def __init__(self, dimension=30):
        self.dimension = dimension
        self.grid = self.generate_ship_layout()
        self.open_cells = self.get_open_cells()
        
    def generate_ship_layout(self):
        """Generate a grid with blocked edges and some interior walls"""
        grid = np.zeros((self.dimension, self.dimension), dtype=int)
        # Block outer edges
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        # Add some interior walls (simpler pattern for better navigation)
        for i in range(5, self.dimension-5, 5):
            for j in range(5, self.dimension-5, 5):
                if np.random.random() < 0.4:  # 40% chance of wall
                    grid[i, j] = 1
                    if np.random.random() < 0.5 and i+1 < self.dimension:
                        grid[i+1, j] = 1
        
        return grid
    
    def get_open_cells(self):
        """Return list of (x,y) coordinates of all open cells"""
        return [(i, j) for i in range(self.dimension) 
                       for j in range(self.dimension) 
                       if self.grid[i, j] == 0]