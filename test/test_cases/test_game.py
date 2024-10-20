from unittest import TestCase

from tic_tac_toe.game import Game
from tic_tac_toe.schemas import GameSettings
from tic_tac_toe.players.random import RandomPlayer


class TestGame(TestCase):
    """
    Tests for the 'Game' class.
    """

    def setUp(self):
        self.game_settings = GameSettings()
        self.x_player = RandomPlayer("X", 1234)
        self.o_player = RandomPlayer("O", 1234)
        self.game = Game(self.game_settings, self.x_player, self.o_player)

    def test_check_win(self):
        """
        Test that the check win method works correctly.
        :return:
        """
        # Unfinished
        board = (
            "X--"
            "O-X"
            "OX-"
        )
        self.assertIsNone(
            self.game.check_winner(board),
            "Marked unfinished game as done."
        )
        board = (
            "---"
            "---"
            "---"
        )
        self.assertIsNone(
            self.game.check_winner(board),
            "Marked empty game as done"
        )

        # Draw
        board = (
            "OOX"
            "XXO"
            "OOX"
        )
        self.assertEqual(
            self.game.check_winner(board),
            "-",
            "Did not identify a draw."
        )

        # Wins
        board = (
            "XOX"
            "-X-"
            "OOX"
        )
        self.assertEqual(
            self.game.check_winner(board),
            "X",
            "Did not identify a win."
        )

        board = (
            "XXX"
            "-O-"
            "OOX"
        )
        self.assertEqual(
            self.game.check_winner(board),
            "X",
            "Did not identify a win."
        )
        board = (
            "XOX"
            "XO-"
            "OOX"
        )
        self.assertEqual(
            self.game.check_winner(board),
            "O",
            "Did not identify a win."
        )

    def test_moves(self):
        """
        Check that game moves are implemented correctly.
        :return:
        """
        self.assertEqual(
            "X",
            self.game.next_turn,
            "Starting player should be 'X'."
        )

        out = self.game.make_move()
        self.assertIsNone(out, "Ended with just 1 move!")
        self.assertEqual(
            len(self.game.empty_cells(self.game.state)),
            8,
            "Incorrect number of empty cells."
        )
        self.assertEqual(
            "O",
            self.game.next_turn,
            "Did not update next turn."
        )

        out = self.game.make_move()
        self.assertIsNone(out, "Ended with just 2 moves!")
        self.assertEqual(
            len(self.game.empty_cells(self.game.state)),
            7,
            "Incorrect number of empty cells."
        )
        self.assertEqual(
            "X",
            self.game.next_turn,
            "Did not update next turn."
        )

    def test_empty_cells(self):
        """
        Test that the game can correctly identify empty cells.
        """
        board = "-" * 9
        self.assertEqual(
            list(range(9)),
            self.game.empty_cells(board),
            "Did not identify empty cells."
        )

        board = (
            "XOX"
            "--O"
            "XO-"
        )
        self.assertEqual(
            [3, 4, 8],
            self.game.empty_cells(board),
            "Did not identify empty cells."
        )
