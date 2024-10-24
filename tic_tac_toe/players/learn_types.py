import os
import json
from pathlib import Path
from typing import Dict, Type, Tuple, Union, Optional

from .q_learn import QLearnPlayer
from .e_sarsa import ESarsaPlayer
from .learned_base import BaseLearnedPlayer
from .schemas import TDSettings, TabularPolicy

PLAYER_TYPES: Dict[str, Type[BaseLearnedPlayer]] = {
    "q_learn": QLearnPlayer,
    "expected_sarsa": ESarsaPlayer
}


def instantiate_agent(
        agent_type: str,
        td_settings_file: Union[str, Path],
        policy_file: Optional[Union[str, Path]] = None,
        ) -> Tuple[BaseLearnedPlayer, TDSettings]:
    """
    Instantiate an agent to train.
    :param agent_type:
    :param td_settings_file:
    :param policy_file: Starting policy to load from. If None, instantiate
        with empty policy.
    :return: player, settings
    """
    if not os.path.isfile(td_settings_file):
        raise FileNotFoundError("TD Settings file not found")

    with open(td_settings_file, "r") as f:
        td_settings = TDSettings(**json.load(f))

    policy = None
    if policy_file is not None:
        if not os.path.isfile(policy_file):
            raise FileNotFoundError("Policy file not found")

        with open(policy_file, "r") as f:
            policy = TabularPolicy(**json.load(f))

    if agent_type not in PLAYER_TYPES:
        raise KeyError("Unknown player type '%s'" % agent_type)

    _cls = PLAYER_TYPES[agent_type]
    agent = _cls.from_policy(mark="X", settings=td_settings, policy=policy)
    return agent, td_settings
