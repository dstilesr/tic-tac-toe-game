import numpy as np
from typing import Dict, Optional, List

from ..schemas import PLAYS
from .schemas import TDSettings
from .learned_base import BaseLearnedPlayer, StateActions


class QLearnPlayer(BaseLearnedPlayer):
    """
    Player that learns via Q-Learning.
    """

    def __init__(
            self,
            mark: PLAYS,
            settings: TDSettings,
            agent_q_vals: Optional[Dict[str, StateActions]] = None):
        """
        :param mark:
        :param settings:
        :param agent_q_vals:
        """
        super().__init__(
            mark=mark,
            settings=settings,
            agent_q_vals=agent_q_vals
        )
        self._prev_state = None
        self._prev_action = None

    def update(self, state: str, reward: float = 0.0):
        """
        Update q-val for previous state-action given the new state and reward.
        :param state:
        :param reward:
        """
        prev_qs = self.agent_q_vals[self._prev_state]
        prev_idx = np.argmax(prev_qs["actions"] == self._prev_action)

        next_q = np.max(self.agent_q_vals[state]["q_vals"])
        prev_val = prev_qs["q_vals"][prev_idx]

        td_err = reward + self.gamma * next_q - prev_val
        prev_qs["q_vals"][prev_idx] = prev_val + self.alpha * td_err

    def make_move(
            self,
            reward: float,
            state: str,
            available_moves: List[int]) -> int:
        """
        Make a move and update the q-values for the previous state-action
        pair.
        :param reward:
        :param state:
        :param available_moves:
        :return:
        """
        self.check_visited_state(state, available_moves)
        if self._prev_state is not None:
            self.update(state, reward)

        next_action = self.select_action(state)
        self._prev_state = state
        self._prev_action = next_action
        return next_action

    def end_game(self, reward: float, state: str):
        """
        Perform final q-learning update and set previous state and action to
        None for a future game.
        :param reward:
        :param state: Terminal board state.
        :return:
        """
        prev_qs = self.agent_q_vals[self._prev_state]
        prev_idx = np.argmax(prev_qs["actions"] == self._prev_action)
        prev_val = prev_qs["q_vals"][prev_idx]

        td_err = reward - prev_val
        prev_qs["q_vals"][prev_idx] = prev_val + self.alpha * td_err
