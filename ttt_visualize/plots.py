import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, TypedDict, Tuple

from tic_tac_toe.schemas import PLAYS
from tic_tac_toe.training.schemas import TrainSummary
from tic_tac_toe.players.schemas import TabularPolicy


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


def plot_policy_state(
        state: str,
        agent_mark: PLAYS,
        qs: TabularPolicy) -> Optional[go.Figure]:
    """
    Plot the policy's values for the given state as a heatmap.
    :param state:
    :param agent_mark:
    :param qs:
    :return:
    """
    # Translate state
    mapping = {
        agent_mark: "1",
        ("O" if agent_mark == "X" else "X"): "2",
        "-": "0"
    }
    mapped_state = "".join(mapping[c] for c in state)

    if mapped_state not in qs.states:
        # State not visited
        return None

    values = []
    action_vals = qs.states[mapped_state]
    for i in range(9):
        key = str(i)
        if key in action_vals:
            values.append(action_vals[key])
        else:
            values.append(np.nan)

    values_arr = np.array(values).reshape((3, 3))
    fig = px.imshow(
        values_arr,
        zmin=-1.0,
        zmax=1.0,
        title="Action Values for State",
        text_auto=".4f"
    )
    return fig


def plot_summary(train_summary: TrainSummary) -> Tuple[go.Figure, ParsedSummary]:
    """
    Plot summary of the training run.
    :param train_summary:
    :return:
    """
    parsed = parse_summary(train_summary)
    df = pd.DataFrame({
        "episode": np.arange(parsed["total_episodes"]),
        "total_wins": np.cumsum(parsed["agent_wins"]),
        "total_losses": np.cumsum(parsed["agent_losses"]),
        "total_draws": np.cumsum(parsed["agent_draws"]),
    })
    group = parsed["total_episodes"] > 30000
    if group:
        df["episode"] = np.floor(df["episode"] / 1000)
        df = df.groupby("episode", as_index=False).agg("max")

    fig = px.line(
        df,
        x="episode",
        y="total_wins",
        title="Training Run"
    )
    fig.add_trace(
        go.Scatter(
            x=df["episode"],
            y=df["total_losses"],
            mode="lines",
            name="total_losses"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["episode"],
            y=df["total_draws"],
            mode="lines",
            name="total_draws"
        )
    )
    return fig, parsed
