import json
import numpy as np
import streamlit as st
from typing import Tuple, TypedDict

from tic_tac_toe import constants as const
from tic_tac_toe.players.schemas import TabularPolicy
from tic_tac_toe.training.schemas import TrainSummary


class ParsedSummary(TypedDict):
    """
    Parsed training summary for plotting and display.
    """
    agent_wins: np.ndarray
    agent_losses: np.ndarray
    agent_draws: np.ndarray
    opponent_type: str
    agent_type: str
    total_episodes: int
    played_as_x: int


def parse_summary(run: TrainSummary) -> ParsedSummary:
    """
    Parse the summary data.
    :param run:
    :return:
    """
    wins = []
    losses = []
    draws = []
    as_x = 0

    for episode in run.episodes:
        agent_mark = episode.agent_mark
        if agent_mark == "X":
            as_x += 1

        if agent_mark == episode.winner:
            wins.append(1.0)
            losses.append(0.0)
            draws.append(0.0)
        elif episode.winner not in ("X", "O"):
            wins.append(0.0)
            losses.append(0.0)
            draws.append(1.0)
        else:
            wins.append(0.0)
            losses.append(1.0)
            draws.append(0.0)

    out = ParsedSummary(
        agent_wins=np.array(wins),
        agent_losses=np.array(losses),
        agent_draws=np.array(draws),
        opponent_type=run.agent_type,
        agent_type=run.rival_type,
        total_episodes=run.total_episodes,
        played_as_x=as_x,
    )
    return out


@st.cache_data
def load_summary_policy(run_name: str) -> Tuple[TabularPolicy, TrainSummary]:
    """
    Load the learned policy (Q-vals, really...) and the run summary for the
    given run.
    :param run_name:
    :return:
    """
    folder = const.OUTPUTS_DIR / run_name

    with (folder / "policy.json").open("r") as f:
        policy = TabularPolicy(**json.load(f))

    with (folder / "summary.json").open("r") as f:
        summary = TrainSummary(**json.load(f))

    return policy, summary


