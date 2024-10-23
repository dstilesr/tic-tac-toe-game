import os
import json
import random
from pathlib import Path
from typing import Union

from .game import Game
from .schemas import GameSettings
from .players.random import RandomPlayer
from .players.console import ConsolePlayer
from .players.learn_types import PLAYER_TYPES
from .players.schemas import TDSettings, TabularPolicy
from .constants import DEFAULT_GAME_CFG, OUTPUTS_DIR, CONFIGS_DIR


__DEFAULT = OUTPUTS_DIR / "test-01"


def play_against_bot(
        game_cfg: Union[str, Path] = DEFAULT_GAME_CFG,
        opponent_type: str = "random",
        policy_file: Union[Path, str] = __DEFAULT / "policy.json",
        td_cfg_file: Union[Path, str] = CONFIGS_DIR / "q-learn-cfg.json"):
    """
    Play game against random player.
    :param game_cfg:
    :param opponent_type:
    :param policy_file:
    :param td_cfg_file:
    :return:
    """
    rand = random.randint(0, 1000)
    if not os.path.isfile(game_cfg):
        raise FileNotFoundError("Did not find game cfg file.")

    with open(game_cfg, "r") as f:
        settings = GameSettings(**json.load(f))

    if opponent_type == "random":
        opponent = RandomPlayer("O", rand)
    else:
        if not os.path.isfile(policy_file):
            raise FileNotFoundError("Did not find policy file.")
        if not os.path.isfile(td_cfg_file):
            raise FileNotFoundError("Did not find td cfg file.")

        with open(policy_file, "r") as f_p,\
             open(td_cfg_file, "r") as f_t:
            policy = TabularPolicy(**json.load(f_p))
            td_set = TDSettings(**json.load(f_t))

        _cls = PLAYER_TYPES[opponent_type]
        opponent = _cls.from_policy(policy=policy, settings=td_set, mark="O")

    which = random.choice(("X", "O"))
    if which == "X":
        x_play = ConsolePlayer("X")
        o_play = opponent
    else:
        opponent.mark = "X"
        o_play = ConsolePlayer("O")
        x_play = opponent

    print("Playing as '%s'" % which)
    game = Game(settings, x_play, o_play)
    done = None

    print("Begin!")
    while done is None:
        print("\x1b[2J")
        print("=" * 30)
        print("Board:")
        print(game.state_string())
        done = game.make_move()

    print("=" * 30)
    print("Game finished! Result:")
    print(game.state_string())
    if done == "-":
        print("DRAW!")
    else:
        print("'%s' WINS!" % done)
