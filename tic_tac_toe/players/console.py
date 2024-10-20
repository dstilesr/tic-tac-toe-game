import random
from typing import List

from ..schemas import PLAYS
from .base import BasePlayer


class ConsolePlayer(BasePlayer):
    """
    Human player that interacts via the terminal.
    """

    def __init__(self, mark: PLAYS, random_seed: int = 12345):
        super().__init__(mark)
        self.rng = random.Random(random_seed)

    def make_move(
            self,
            reward: float,
            state: str, available_moves: List[int]) -> int:
        """
        Make a random move on the board.
        :param reward:
        :param state:
        :param available_moves:
        :return:
        """
        selected = None
        while selected not in available_moves:
            player_input = input(
                "Enter next move (Available: %s) ->  "
                % ", ".join(map(str, available_moves))
            )
            try:
                selected = int(player_input)
                if selected not in available_moves:
                    print("Invalid input! Please enter an integer in the list!")
            except ValueError:
                print("Invalid input! Please enter an integer in the list!")

        return selected

    def end_game(self, reward: float, state: str):
        """
        Dummy method in this case.
        :param reward:
        :param state:
        :return:
        """
        pass
