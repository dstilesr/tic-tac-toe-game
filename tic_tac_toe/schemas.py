from typing import Literal
from pydantic import BaseModel


class GameSettings(BaseModel):
    """
    Settings for the game.
    """
    win_reward: float = 1.0  #: Reward for winning
    lose_reward: float = -1.0  #: Reward for losing
    draw_reward: float = 0.0  #: Reward for ending in draw
    step_reward: float = 0.0  #: Reward at each non-terminal step

PLAYS = Literal["X", "O"]
