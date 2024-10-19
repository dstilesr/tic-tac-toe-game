import random
from typing import List

from ..schemas import PLAYS
from .base import BasePlayer


class RandomPlayer(BasePlayer):
    """
    Player that randomly chooses one of the available moves.
    """

    def __init__(self, mark: PLAYS, random_seed: int = 12345):
        super().__init__(mark)
        self.rng = random.Random(random_seed)

    def make_move(self, reward: float, state: str, available_moves: List[int]) -> int:
        """
        Make a random move on the board.
        :param reward:
        :param state:
        :param available_moves:
        :return:
        """
        return self.rng.choice(available_moves)

    def end_game(self, reward: float, state: str):
        """
        Dummy method in this case.
        :param reward:
        :param state:
        :return:
        """
        pass
