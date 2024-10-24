import numpy as np
from typing import Dict, Optional, List

from ..schemas import PLAYS
from .schemas import TDSettings
from .learned_base import BaseLearnedPlayer, StateActions


class ESarsaPlayer(BaseLearnedPlayer):
    """
    Player that learns via Expected Sarsa.
    """

    def __init__(
            self,
            mark: PLAYS,
            settings: TDSettings,
            agent_q_vals: Optional[Dict[str, StateActions]] = None,
            freeze: bool = False,):
        """
        :param mark:
        :param settings:
        :param agent_q_vals:
        :param freeze:
        """
        super().__init__(
            mark=mark,
            settings=settings,
            agent_q_vals=agent_q_vals,
            freeze=freeze
        )
        self._prev_state = None
        self._prev_action = None

    def get_egreedy_probs(self, qs: np.ndarray) -> np.ndarray:
        """
        Get the epsilon-greedy probability distribution for the given
        array of action values.
        :param qs:
        :return:
        """
        probs = (
            np.ones_like(qs, dtype=np.float32) * self.epsilon / qs.shape[0]
        )
        probs[np.argmax(qs)] += (1.0 - self.epsilon)
        return probs

    def update(self, mapped_state: str, reward: float = 0.0):
        """
        Update q-val for previous state-action given the new state and reward.
        :param mapped_state: Translated state.
        :param reward:
        """
        if self.frozen:
            return

        prev_qs = self.agent_q_vals[self._prev_state]
        prev_idx = np.argmax(prev_qs["actions"] == self._prev_action)

        curr_qs = self.agent_q_vals[mapped_state]["q_vals"]
        next_q = np.sum(curr_qs * self.get_egreedy_probs(curr_qs))

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
        mapped_state = self.translate_board(state)
        self.check_visited_state(mapped_state, available_moves)
        if self._prev_state is not None:
            self.update(mapped_state, reward)

        next_action = self.select_action(mapped_state)
        self._prev_state = mapped_state
        self._prev_action = next_action
        return next_action

    def end_game(self, reward: float, state: str):
        """
        Perform final update and set previous state and action to None for a
        future game.
        :param reward:
        :param state: Terminal board state.
        :return:
        """
        prev_qs = self.agent_q_vals[self._prev_state]
        prev_idx = np.argmax(prev_qs["actions"] == self._prev_action)
        prev_val = prev_qs["q_vals"][prev_idx]

        td_err = reward - prev_val
        prev_qs["q_vals"][prev_idx] = prev_val + self.alpha * td_err

        self._prev_action = None
        self._prev_state = None