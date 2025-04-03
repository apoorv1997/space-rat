class Cell:
    """
    Class representing a cell in the ship grid.
    """

    def __init__(self, row: int, col: int):
        self.neighbors = []
        self.row = row
        self.col = col
        self.open = False

    def reset_cell(self):
        """Reset cell to initial state"""
        self.open = False

    def count_blocked_neighbors(self):
        """Return number of blocked neighbors for a cell"""
        return sum(not neighbor.open for neighbor in self.neighbors)
    
    def count_open_neighbors(self):
        """Return number of open neighbors for a cell"""
        return sum(neighbor.open for neighbor in self.neighbors)

    def get_closed_neighbors(self):
        """Return list of closed neighbors"""
        return [neighbor for neighbor in self.neighbors if not neighbor.open]

    def get_open_neighbors(self):
        """Return list of open neighbors"""
        return [neighbor for neighbor in self.neighbors if neighbor.open]

    def is_dead_end(self):
        """Return True if the cell is open and has exactly one open neighbor"""
        return self.open and self.count_open_neighbors() == 1
    
    def is_frontier(self):
        """Return True if the cell is not open and has exactly one open neighbor"""
        return not self.open and self.count_open_neighbors() == 1
    
    def get_viable_adjacent_cells(self):
        """Return list of open neighbors that are not on fire"""
        return [neighbor for neighbor in self.neighbors if neighbor.open]

    def open_cell(self):
        """Open the cell"""
        self.open = True

        
    def is_open(self):
        """Return True if the cell is open"""
        return self.open