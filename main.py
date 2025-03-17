from environment import Environment
from ship import Ship
from visualizer import Visualizer


def main():
    ship_dimension = 30  # dimension sets the nxn ship grid size
    alpha = 0.1  
    cell_size = 30  
    realtime = True
    tick_interval = 0.5
    ship = Ship(ship_dimension)
    env = Environment(ship, alpha)
    visualizer = Visualizer(ship, cell_size, env)
    visualizer.draw_grid(ship, cell_size, env)

    
if __name__ == "__main__":
    main()