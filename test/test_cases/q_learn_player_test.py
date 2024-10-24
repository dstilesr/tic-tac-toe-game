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
            epsilon=0.1,
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
            "000": StateActions(
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
                player.select_action("000"),
                int(np.argmax(qs["000"]["q_vals"])),
                "Did not select greedy action!"
            )

    def test_episode_end(self):
        """
        Test that the final update is performed correctly on ending a game.
        """
        qs = {
            "000": StateActions(
                actions=np.array([0, 1, 2, 3], dtype=np.int16),
                q_vals=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        }
        sets = TDSettings(
            epsilon_greedy=False,
            epsilon=0.15,
            default_q=2.0,
            discount_rate=0.9,
            step_size=0.9
        )
        player = QLearnPlayer(
            mark="X",
            settings=sets,
            agent_q_vals=qs
        )
        player.make_move(0.0, "---", [0, 1, 2, 3])
        player.end_game(2.0, "XOX")

        good = np.allclose(
            np.array([0.0, 1.9, 0.0, 0.0]),
            player.agent_q_vals["000"]["q_vals"],
        )
        self.assertTrue(good, "Did not correctly update values on episode end!")

    def test_step(self):
        """
        Test that the final update is performed correctly on ending a game.
        """
        qs = {
            "000": StateActions(
                actions=np.array([0, 1, 2, 3], dtype=np.int16),
                q_vals=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            ),
            "100": StateActions(
                actions=np.array([0, 1, 2, 3], dtype=np.int16),
                q_vals=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        }
        sets = TDSettings(
            epsilon_greedy=False,
            epsilon=0.15,
            default_q=2.0,
            discount_rate=0.8,
            step_size=0.5
        )
        player = QLearnPlayer(
            mark="X",
            settings=sets,
            agent_q_vals=qs
        )
        player.make_move(0.0, "---", [0, 1, 2, 3])
        player.make_move(3.0, "X--", [0, 1, 2, 3])

        good = np.allclose(
            np.array([0.0, 2.4, 0.0, 0.0]),
            player.agent_q_vals["000"]["q_vals"],
        )
        self.assertTrue(good, "Did not correctly update values on move!")

    def test_end_game(self):
        """
        Test that values are updated correctly when the game ends.
        :return:
        """
        qs = {
            "000": StateActions(
                actions=np.array([0, 1, 2, 3], dtype=np.int16),
                q_vals=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            ),
            "100": StateActions(
                actions=np.array([0, 1, 2, 3], dtype=np.int16),
                q_vals=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        }
        sets = TDSettings(
            epsilon_greedy=False,
            epsilon=0.15,
            default_q=2.0,
            discount_rate=0.8,
            step_size=0.5
        )
        agent = QLearnPlayer(mark="X", settings=sets, agent_q_vals=qs)
        agent.make_move(0.0, "---", [0, 1, 2, 3])
        agent.end_game(10.0, "X--")

        self.assertTrue(
            np.allclose(np.array([0.0, 5.5, 0.0, 0.0]), agent.agent_q_vals["000"]["q_vals"]),
            "Did not correctly update values on game end!"
        )

        self.assertIsNone(agent.prev_state, "Did not clear previous state!")
        self.assertIsNone(agent.prev_action, "Did not clear previous action!")

    def test_egreedy_prob_compute(self):
        """
        Test that epsilon-greedy probabilities are computed correctly.
        :return:
        """
        vals = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        agent = QLearnPlayer(mark="X", settings=self.ql_settings)
        probs = agent.get_egreedy_probs(vals)
        self.assertTrue(
            np.allclose(probs, np.array([.025, .925, .025, .025])),
            "Did not compute probabilities correctly!"
        )

