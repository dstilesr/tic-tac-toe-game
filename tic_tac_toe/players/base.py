from typing import List
from abc import ABC, abstractmethod, ABCMeta

from ..schemas import PLAYS


class BasePlayer(ABC):
    """
    Interface for a player in the game.
    """

    def __init__(self, mark: PLAYS) -> None:
        self.__mark = mark

        rival = "O" if self.mark == "X" else "X"
        self.mapping = {
            self.mark: "1",
            rival: "2",
            "-": "0"
        }

    @property
    def mark(self) -> PLAYS:
        """
        Which 'mark' the player plays.
        """
        return self.__mark

    @mark.setter
    def mark(self, mark: PLAYS):
        """
        Setter for mark property.
        :param mark:
        """
        if mark not in ("X", "O"):
            raise ValueError("Invalid mark '%s'" % mark)

        rival = "O" if mark == "X" else "X"
        self.mapping = {
            mark: "1",
            rival: "2",
            "-": "0"
        }
        self.__mark = mark

    def translate_board(self, state: str) -> str:
        """
        Translate the state of the board to a string of 0, 1, 2 characters. 0
        represents an empty cell, 1 a cell occupied by the player, and 2 a
        cell occupied by the rival player.
        :param state:
        :return:
        """
        return "".join(
            self.mapping[s] for s in state
        )

    @abstractmethod
    def make_move(
            self,
            reward: float,
            state: str,
            available_moves: List[int]) -> int:
        """
        Make a move in the game.
        :param reward: Reward from previous state-action.
        :param state: Current state of the game.
        :param available_moves: Available moves in the game.
        :return: Index (between 0 and 8) of the cell to occupy.
        """
        pass

    @abstractmethod
    def end_game(self, reward: float, state: str):
        """
        Register end of the game.
        :param reward: Final reward.
        :param state: Final state of the board.
        :return:
        """
        pass
