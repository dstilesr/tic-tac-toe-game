import unittest
import numpy as np

from tic_tac_toe.players.schemas import TDSettings
from tic_tac_toe.players.q_learn import QLearnPlayer
from tic_tac_toe.players.learned_base import StateActions


class QLearnPlayerTest(unittest.TestCase):
    """
    Tests for QLearn Player Class.
    """

    def setUp(self):
        self.ql_settings = TDSettings(
            epsilon_greedy=False,
            epsilon=0.15,
            default_q=2.0,
            discount_rate=0.9,
            step_size=1.0
        )

    def test_state_init(self):
        """
        Test that states for a new value are initialized correctly.
        """
        player = QLearnPlayer(mark="X", settings=self.ql_settings)
        self.assertDictEqual(
            player.agent_q_vals,
            {},
            "Initialized with non-empty values dictionary!"
        )

        state = "0" * 9
        player.check_visited_state(
            state,
            list(range(len(state)))
        )

        self.assertEqual(
            len(player.agent_q_vals),
            1,
            "Did not add a new state!"
        )

        qs = player.agent_q_vals[state]
        defaults =  np.array([self.ql_settings.default_q] * len(state))
        self.assertTrue(
            np.allclose(defaults, qs["q_vals"]),
            "Did not initialize q values to default!"
        )

        # Check known state
        state = "0" * 9
        visited = player.check_visited_state(
            state,
            list(range(len(state)))
        )
        self.assertTrue(visited, "Did not identify known state!")

        # Add second state
        state = "1" * 9
        visited = player.check_visited_state(
            state,
            list(range(len(state)))
        )
        self.assertFalse(visited, "Did not identify unknown state!")
        self.assertEqual(
            len(player.agent_q_vals),
            2,
            "Did not add a new state!"
        )

    def test_choose_action_greedy(self):
        """
        Test that a greedy action can be chosen reliably.
        """
        qs = {
            "---": StateActions(
                actions=np.array([0, 1, 2, 3], dtype=np.int16),
                q_vals=np.array([0.1, 0.2, 0.1, -2.0], dtype=np.float32),
            )
        }
        player = QLearnPlayer(
            mark="X",
            settings=self.ql_settings,
            agent_q_vals=qs
        )

        for _ in range(30):
            self.assertEqual(
                player.select_action("---"),
                int(np.argmax(qs["---"]["q_vals"])),
                "Did not select greedy action!"
            )
