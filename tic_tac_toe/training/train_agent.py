import os
import json
import random
from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional
from datetime import datetime, timezone

from ..game import Game
from . import schemas as sch
from ..players import BasePlayer
from ..schemas import GameSettings
from ..players.random import RandomPlayer
from ..players.schemas import TDSettings, TabularPolicy
from ..players.learn_types import PLAYER_TYPES, BaseLearnedPlayer
from ..constants import OUTPUTS_DIR, CONFIGS_DIR, DEFAULT_GAME_CFG


def instantiate_agent(
        agent_type: str,
        td_settings: TDSettings,
        policy_file: Optional[Union[str, Path]] = None,
        ) -> BaseLearnedPlayer:
    """
    Instantiate an agent to train.
    :param agent_type:
    :param td_settings:
    :param policy_file: Starting policy to load from. If None, instantiate
        with empty policy.
    :return:
    """
    policy = None
    if policy_file is not None:
        if not os.path.isfile(policy_file):
            raise FileNotFoundError("Policy file not found")

        with open(policy_file, "r") as f:
            policy = TabularPolicy(**json.load(f))

    if agent_type not in PLAYER_TYPES:
        raise KeyError("Unknown player type '%s'" % agent_type)

    _cls = PLAYER_TYPES[agent_type]
    return _cls.from_policy(mark="X", settings=td_settings, policy=policy)


def run_game(
        game_settings: GameSettings,
        agent: BaseLearnedPlayer,
        rival: BasePlayer) -> sch.EpisodeSummary:
    """
    Run a game between the agent and the rival.
    :param game_settings:
    :param agent:
    :param rival:
    :return:
    """
    agent_mark = random.choice(("X", "O"))
    if agent_mark == "X":
        game = Game(
            game_settings,
            x_player=agent,
            o_player=rival,
        )
    else:
        game = Game(
            game_settings,
            o_player=agent,
            x_player=rival,
        )


    done = None
    while done is None:
        done = game.make_move()

    out = sch.EpisodeSummary(
        agent_mark=agent_mark,
        winner=done,
        end_board=game.state,
        x_player_type=type(game.players["X"]).__name__,
        o_player_type=type(game.players["O"]).__name__,
    )
    return out


def train_agent_random(
        run_name: str,
        agent_type: str = "q_learn",
        total_episodes: int = 5000,
        game_settings_file: Union[Path, str] = DEFAULT_GAME_CFG,
        td_settings_file: Union[Path, str] = CONFIGS_DIR / "q-learn-cfg.json",
        policy_file: Optional[Union[str, Path]] = None,) -> sch.TrainSummary:
    """
    Train an agent against a random opponent.
    :param run_name:
    :param agent_type:
    :param total_episodes:
    :param game_settings_file:
    :param td_settings_file:
    :param policy_file:
    :return:
    """
    if not (OUTPUTS_DIR / run_name).is_dir():
        os.makedirs(OUTPUTS_DIR / run_name)

    if not os.path.isfile(game_settings_file):
        raise FileNotFoundError("Game settings file not found")

    if not os.path.isfile(td_settings_file):
        raise FileNotFoundError("TD Settings file not found")

    with open(td_settings_file, "r") as f:
        td_settings = TDSettings(**json.load(f))

    with open(game_settings_file, "r") as f:
        game_settings = GameSettings(**json.load(f))

    agent = instantiate_agent(
        agent_type,
        policy_file=policy_file,
        td_settings=td_settings
    )
    rival = RandomPlayer(
        "O",
        random_seed=round(datetime.now(timezone.utc).timestamp())
    )

    episodes = []
    for _ in tqdm(range(total_episodes)):
        ep = run_game(
            game_settings=game_settings,
            agent=agent,
            rival=rival,
        )
        episodes.append(ep)

    out = sch.TrainSummary(
        total_episodes=total_episodes,
        episodes=episodes,
        game_settings=game_settings,
        td_settings=td_settings,
    )

    with open(OUTPUTS_DIR / run_name / "summary.json", "w") as f:
        json.dump(out.model_dump(), f)

    with open(OUTPUTS_DIR / run_name / "policy.json", "w") as f:
        json.dump(agent.dump_q_values().model_dump(), f)

    return out
