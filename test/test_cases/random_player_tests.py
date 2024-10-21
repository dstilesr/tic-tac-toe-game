from unittest import TestCase

from tic_tac_toe.players.random import RandomPlayer


class RandomPlayerTest(TestCase):
    """
    Tests for RandomPlayer class.
    """

    def test_translate(self):
        """
        Test that the board is translated correctly.
        """
        board = (
            "XOX"
            "OOX"
            "X--"
        )
        x_play = RandomPlayer("X")
        self.assertEqual(
            x_play.translate_board(board),
            "121221100",
            "Failed to translate board."
        )

        x_play.mark = "O"
        self.assertEqual(
            x_play.translate_board(board),
            "212112200",
            "Failed to translate board."
        )

        o_play = RandomPlayer("O")
        self.assertEqual(
            o_play.translate_board(board),
            "212112200",
            "Failed to translate board."
        )

    def test_move_selection(self):
        """
        Test that moves are selected correctly.
        """
        x_play = RandomPlayer("X")
        self.assertIn(
            x_play.make_move(0.0, "", [1, 2, 3]),
            [1, 2, 3],
            "Selected invalid move."
        )
        self.assertEqual(
            x_play.make_move(0.0, "", [1]),
            1,
            "Selected invalid move."
        )
