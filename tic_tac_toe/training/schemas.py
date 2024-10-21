from pydantic import BaseModel
from typing import Literal, List

from ..players.schemas import TDSettings
from ..schemas import PLAYS, GameSettings


class EpisodeSummary(BaseModel):
    """
    Summary of an episode during training.
    """
    winner: Literal["X", "O", "-"]
    end_board: str
    x_player_type: str
    o_player_type: str

    agent_mark: PLAYS


class TrainSummary(BaseModel):
    """
    Summary of training run.
    """
    total_episodes: int
    game_settings: GameSettings
    td_settings: TDSettings
    episodes: List[EpisodeSummary]
