import numpy as np
import plotly.express as px
from typing import Optional
import plotly.graph_objects as go

from tic_tac_toe.schemas import PLAYS
from tic_tac_toe.players.schemas import TabularPolicy


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
