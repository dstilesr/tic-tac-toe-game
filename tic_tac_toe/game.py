from typing import Literal, List, Optional, Tuple, Dict

from .players import BasePlayer
from .schemas import GameSettings, PLAYS


class Game:
    """
    TicTacToe game environment.
    """
    WINS: List[Tuple[int, int, int]] = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  #: Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  #: Columns
        (0, 4, 8), (2, 4, 6)  #: Diagonals
    ]
    EMPTY_MARK: str = "-"

    def __init__(
            self,
            settings: GameSettings,
            x_player: BasePlayer,
            o_player: BasePlayer):
        """
        :param settings:
        :param x_player:
        :param o_player:
        """
        self.__state = self.EMPTY_MARK * 9
        self.__next_turn: PLAYS = "X"

        assert x_player.mark == "X" and o_player.mark == "O", "Wrong player setting!"
        self.__players = {
            "X": x_player,
            "O": o_player,
        }
        self.__settings = settings

    @property
    def players(self) -> Dict[PLAYS, BasePlayer]:
        """
        Dictionary mapping mark -> player.
        """
        return self.__players

    @property
    def state(self) -> str:
        """
        State of the game. This is a string of length 9 with each character
        corresponding to a position on the board. A '-' represents an empty
        cell.
        """
        return self.__state

    @property
    def next_turn(self) -> PLAYS:
        """
        Which player has the next turn in the game. 1 for X, 2 for O.
        """
        return self.__next_turn

    @property
    def settings(self) -> GameSettings:
        """
        Game reward settings.
        """
        return self.__settings

    def empty_cells(self, state: str) -> List[int]:
        """
        Return the empty cell indices for a given game state.
        :param state:
        :return:
        """
        return [
            i for i, cell in enumerate(state)
            if cell == self.EMPTY_MARK
        ]

    def check_winner(self, state: str) -> Optional[Literal["-", "X", "O"]]:
        """
        Determine if the game is over and the winner. Return None if the game
        is not over, '-' for a draw, and 'X' or 'O' to indicate the winner.
        :param state:
        :return:
        """
        board_full = self.EMPTY_MARK not in state
        for play in ("X", "O"):
            won = any(
                all(state[i] == play for i in w)
                for w in self.WINS
            )
            if won:
                return play

        if board_full:
            return "-"

        return None

    def make_move(self) -> Optional[Literal["-", "X", "O"]]:
        """
        A player makes a move in the game.
        :return: None if the game is not over. Otherwise, see return for
            'check_winner'.
        """
        available = self.empty_cells(self.state)
        other = "O" if self.next_turn == "X" else "X"
        move = self.players[self.next_turn].make_move(
            self.settings.step_reward,
            self.state,
            available
        )

        # Update state
        state = list(self.state)
        state[move] = self.next_turn
        self.__state = ",".join(state)

        win = self.check_winner(self.state)
        assert win != other, "Cannot win if the other player has made a move!"

        if win == self.next_turn:
            self.players[self.next_turn].end_game(
                self.settings.win_reward,
                self.state
            )
            self.players[other].end_game(
                self.settings.lose_reward,
                self.state
            )
        elif win == "-":
            self.players[self.next_turn].end_game(
                self.settings.draw_reward,
                self.state
            )
            self.players[other].end_game(
                self.settings.draw_reward,
                self.state
            )

        self.__next_turn = other
        return win
