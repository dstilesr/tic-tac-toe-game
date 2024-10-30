import json
from typing import Tuple

from tic_tac_toe import constants as const
from tic_tac_toe.players.schemas import TabularPolicy
from tic_tac_toe.training.schemas import TrainSummary


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
