import os
import json
import random
from pathlib import Path
from typing import Union

from .game import Game
from .schemas import GameSettings
from .constants import DEFAULT_GAME_CFG
from .players.random import RandomPlayer
from .players.console import ConsolePlayer


def play_against_random(game_cfg: Union[str, Path] = DEFAULT_GAME_CFG):
    """
    Play game against random player.
    :param game_cfg:
    :return:
    """
    if not os.path.isfile(game_cfg):
        raise FileNotFoundError("Did not find game cfg file.")

    with open(game_cfg, "r") as f:
        settings = GameSettings(**json.load(f))

    rand = random.randint(0, 1000)
    which = random.choice(("X", "O"))
    if which == "X":
        x_play = ConsolePlayer("X")
        o_play = RandomPlayer("O", rand)
    else:
        o_play = ConsolePlayer("O")
        x_play = RandomPlayer("X", rand)

    print("Playing as '%s'" % which)
    game = Game(settings, x_play, o_play)
    done = None

    print("Begin!")
    while done is None:
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
