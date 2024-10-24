import numpy as np
from typing import Dict, Optional, List

from ..schemas import PLAYS
from .schemas import TDSettings
from .learned_base import BaseTDPlayer, StateActions


class QLearnPlayer(BaseTDPlayer):
    """
    Player that learns via Q-Learning.
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
        self.prev_state = None
        self.prev_action = None

    def update(self, mapped_state: str, reward: float = 0.0):
        """
        Update q-val for previous state-action given the new state and reward.
        :param mapped_state: Translated state.
        :param reward:
        """
        if self.frozen:
            return

        prev_qs = self.agent_q_vals[self.prev_state]
        prev_idx = np.argmax(prev_qs["actions"] == self.prev_action)

        next_q = np.max(self.agent_q_vals[mapped_state]["q_vals"])
        prev_val = prev_qs["q_vals"][prev_idx]

        td_err = reward + self.gamma * next_q - prev_val
        prev_qs["q_vals"][prev_idx] = prev_val + self.alpha * td_err
