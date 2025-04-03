import random
import numpy as np

from cell import Cell


class Ship:
    def __init__(self, dimension=30):
        self.dimension = dimension
        self.initial_cell = None
        self.grid = [
            [Cell(row, col) for col in range(dimension)]
            for row in range(dimension)
        ]
        self.set_cell_neighbors()
        self.generate_ship_layout()
        self.open_cells = self.get_open_cells()

    
    def cell_in_bounds(self, row, col):
        """Check if cell coordinates are within grid bounds"""
        return 0 <= row < self.dimension and 0 <= col < self.dimension
    
    def get_cell(self, row, col):
        """Return cell object at given coordinates"""
        if self.cell_in_bounds(row, col):
            return self.grid[row][col]
        raise ValueError("Invalid cell coordinates")
    

    def get_dead_end_cells(self):
        """Return list of open cells that have exactly one open neighbor"""
        dead_ends = []
        for row in self.grid:
            for cell in row:
                if cell.is_dead_end():
                    dead_ends.append(cell)
        return dead_ends
    

    def set_cell_neighbors(self):
        """Set valid neighbors for each cell in the grid"""
        for row in self.grid:
            for cell in row:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = cell.row + dr, cell.col + dc
                    if self.cell_in_bounds(nr, nc):
                        cell.neighbors.append(self.get_cell(nr, nc))
    
    def count_open_neighbors(self):
        """Return number of open neighbors for a cell"""
        return sum(neighbor.open for neighbor in self.neighbors)
    
    def generate_ship_layout(self):
        """Generate a grid with blocked edges and some interior walls"""
        # grid = np.zeros((self.dimension, self.dimension), dtype=int)
        # Block outer edges
        # grid[0, :] = 1
        # grid[-1, :] = 1
        # grid[:, 0] = 1
        # grid[:, -1] = 1

        start_row = random.randint(1, self.dimension - 2)
        start_col = random.randint(1, self.dimension - 2)
        start_cell = self.get_cell(start_row, start_col)
        start_cell.open_cell()
        self.initial_cell = start_cell

        # Initialize frontier with closed neighbors of the starting cell
        frontier = set(start_cell.get_closed_neighbors())
        while frontier:
            cell = random.choice(list(frontier))
            if cell.is_frontier():
                cell.open_cell()
            frontier.remove(cell)
            # Update frontier: add neighbors of the opened cell that are now frontier candidates.
            for neighbor in cell.get_closed_neighbors():
                if neighbor.is_frontier():
                    frontier.add(neighbor)

        # Sanity Check
        open_count = sum(cell.is_open() for row in self.grid for cell in row)
        total = self.dimension * self.dimension
        percent_open = open_count / total * 100
        dead_end_cells = self.get_dead_end_cells()
        initial_dead_end_count = len(dead_end_cells)
        # randomly open half of the dead ends
        while len(dead_end_cells) > initial_dead_end_count // 2:
            cell = random.choice(dead_end_cells)
            closed_neighbors = cell.get_closed_neighbors()
            if closed_neighbors:
                random.choice(closed_neighbors).open_cell()
            dead_end_cells = self.get_dead_end_cells()
        # return grid
    
    # def get_open_cells(self):
    #     """Return list of (x,y) coordinates of all open cells"""
    #     return [(i, j) for i in range(self.dimension) 
    #                    for j in range(self.dimension) 
    #                    if self.grid[i, j] == 0]

    def get_open_cells(self):
        """Return list of open cells"""
        return [cell for row in self.grid for cell in row if cell.open]