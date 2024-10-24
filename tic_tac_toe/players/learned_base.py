import numpy as np
from abc import ABCMeta, abstractmethod
from typing import TypedDict, Dict, Optional, List, Any

from ..schemas import PLAYS
from .base import BasePlayer
from .schemas import TDSettings, TabularPolicy


class StateActions(TypedDict):
    """
    State actions represented within the agent.
    """
    actions: np.ndarray[Any, np.int16]  #: 1d array of action labels
    q_vals: np.ndarray[Any, np.float32]  #: 1d array of q-values for actions


class BaseLearnedPlayer(BasePlayer, metaclass=ABCMeta):
    """
    Base for players with learned policies.
    """

    @classmethod
    def from_policy(
            cls,
            policy: Optional[TabularPolicy],
            mark: PLAYS,
            settings: TDSettings) -> "BaseLearnedPlayer":
        """
        Instantiate from a serialized policy.
        :param policy:
        :param mark:
        :param settings:
        :return:
        """
        agent_policy = {}
        # Translate policy format
        if policy is not None:
            for state, qs in policy.states.items():
                actions = []
                state_q = []
                for a, q in qs.items():
                    actions.append(int(a))
                    state_q.append(q)

                agent_policy[state] = StateActions(
                    actions=np.array(actions, dtype=np.int16),
                    q_vals=np.array(state_q, dtype=np.float32)
                )

        return cls(mark=mark, settings=settings, agent_q_vals=agent_policy)

    def __init__(
            self,
            mark: PLAYS,
            settings: TDSettings,
            agent_q_vals: Optional[Dict[str, StateActions]] = None,
            freeze: bool = False):
        """
        :param mark:
        :param settings:
        :param agent_q_vals:
        :param freeze: Freeze the agent's weights or not.
        """
        super().__init__(mark)
        self.__epsilon = settings.epsilon
        self.__gamma = settings.discount_rate
        self.__step_size = settings.step_size
        self.__e_greedy = settings.epsilon_greedy
        self.__default_q = settings.default_q
        self.__rng = np.random.RandomState(settings.random_seed)

        if agent_q_vals is None:
            agent_q_vals = {}
        self.__agent_qs = agent_q_vals
        self.__freeze = freeze

    @property
    def frozen(self) -> bool:
        """
        Return whether the agent values are frozen (as opposed to being
        updated).
        """
        return self.__freeze

    @frozen.setter
    def frozen(self, value: bool):
        self.__freeze = bool(value)

    @property
    def rng(self) -> np.random.RandomState:
        """
        Agent's random number generator.
        """
        return self.__rng

    @property
    def epsilon(self) -> float:
        """
        Epsilon.
        """
        return self.__epsilon

    @property
    def epsilon_greedy(self) -> bool:
        """
        Whether to use epsilon greedy action selection or not. If false, use
        greedy action selection.
        """
        return self.__e_greedy

    @epsilon_greedy.setter
    def epsilon_greedy(self, value: bool):
        self.__e_greedy = bool(value)

    @property
    def agent_q_vals(self) -> Dict[str, StateActions]:
        """
        Agent's estimated q-values. Map translated state -> StateActions.
        """
        return self.__agent_qs

    @property
    def gamma(self) -> float:
        """
        Discount rate (gamma value).
        """
        return self.__gamma

    @property
    def alpha(self) -> float:
        """
        Step size / learning rate (alpha value).
        """
        return self.__step_size

    def get_egreedy_probs(self, qs: np.ndarray) -> np.ndarray:
        """
        Get the epsilon-greedy probability distribution for the given
        array of action values.
        :param qs:
        :return:
        """
        probs = (
            np.zeros_like(qs) + (self.epsilon / qs.shape[0])
        )
        probs[np.argmax(qs)] += (1.0 - self.epsilon)
        return probs

    def check_visited_state(
            self,
            state: str,
            available_actions: List[int]) -> bool:
        """
        If the state has been visited before, return True. If not, initialize
        the q-values for the state and return false.
        :param state:
        :param available_actions:
        :return:
        """
        if state in self.agent_q_vals:
            return True

        q_vals = np.array([self.__default_q] * len(available_actions), dtype=np.float32)
        self.agent_q_vals[state] = StateActions(
            actions=np.array(available_actions, dtype=np.int16),
            q_vals=q_vals
        )
        return False

    def dump_q_values(self) -> TabularPolicy:
        """
        Dump the learned q-values to a serialized policy.
        :return:
        """
        states = {}
        for state, qs in self.agent_q_vals.items():
            states[state] = {
                str(a): float(q)
                for a, q in zip(qs["actions"].tolist(), qs["q_vals"].tolist())
            }
        return TabularPolicy(states=states)

    def select_action(self, state: str) -> int:
        """
        Select the next action according to the saved q values.
        """
        state_qs = self.agent_q_vals[state]
        idx = np.argmax(state_qs["q_vals"])
        if not self.epsilon_greedy:
            return int(state_qs["actions"][idx])

        val = self.rng.rand()
        if val <= self.epsilon:
            # Choose random action
            return int(self.rng.choice(state_qs["actions"]))

        return int(state_qs["actions"][idx])


class BaseTDPlayer(BaseLearnedPlayer, metaclass=ABCMeta):
    """
    Base for players based on TD-Learning variants.
    """

    def __init__(
            self,
            mark: PLAYS,
            settings: TDSettings,
            agent_q_vals: Optional[Dict[str, StateActions]] = None,
            freeze: bool = False):
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
        self.__prev_action = None
        self.__prev_state = None

    @property
    def prev_action(self) -> Optional[int]:
        """
        Previous action taken by the agent.
        """
        return self.__prev_action

    @prev_action.setter
    def prev_action(self, action: Optional[int]) -> None:
        self.__prev_action = action

    @property
    def prev_state(self) -> Optional[str]:
        """
        Previous state the agent visited.
        """
        return self.__prev_state

    @prev_state.setter
    def prev_state(self, state: Optional[str]) -> None:
        self.__prev_state = state

    @abstractmethod
    def update(self, mapped_state: str, reward: float = 0.0):
        """
        Update q-val for previous state-action given the new state and reward.
        :param mapped_state: Translated state.
        :param reward:
        """
        pass

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
        if self.prev_state is not None:
            self.update(mapped_state, reward)

        next_action = self.select_action(mapped_state)
        self.prev_state = mapped_state
        self.prev_action = next_action
        return next_action

    def end_game(self, reward: float, state: str):
        """
        Perform final TD-learning update and set previous state and action to
        None for a future game.
        :param reward:
        :param state: Terminal board state.
        :return:
        """
        prev_qs = self.agent_q_vals[self.prev_state]
        prev_idx = np.argmax(prev_qs["actions"] == self.prev_action)
        prev_val = prev_qs["q_vals"][prev_idx]

        td_err = reward - prev_val
        prev_qs["q_vals"][prev_idx] = prev_val + self.alpha * td_err

        self.prev_action = None
        self.prev_state = None
