import random
from bot import Bot
# from botcontroller import BotController


class Environment:
    """
    The Environment class handles time-stepping the simulation and updating the state of the ship.
    """
    def __init__(self, ship, alpha=0.3):
        self.ship = ship
        self.alpha = alpha
        self.space_rat = None
        self.bot = None


    def tick(self, bot_direction):
        """
        """
        self.bot.move(bot_direction) # make the move
        self.bot_path.append(self.bot.cell) # record the path
        curr_bot_cell = self.bot.cell


        if curr_bot_cell is self.space_rat:
            for cell in self.ship.get_on_fire_cells():
                self.ship.extinguish_cell(cell)
            return "success"

        return "ongoing"


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