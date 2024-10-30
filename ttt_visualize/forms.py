import os
import re
import streamlit as st
from typing import List
from pathlib import Path
from typing import Optional

from tic_tac_toe.constants import OUTPUTS_DIR


def __list_runs() -> List[str]:
    """
    List the output directories of training runs. Each one must contain
    a `policy.json` file and a `summary.json` file.
    :return: List of training run names.
    """
    if not os.path.isdir(OUTPUTS_DIR):
        return  []

    out = []
    for fp in OUTPUTS_DIR.iterdir():
        if (fp / "policy.json").is_file() and (fp / "summary.json").is_file():
            out.append(fp.parts[-1])

    return out


def __evaluate_state(state: str) -> Optional[str]:
    """
    Check that the input state is valid.
    :return:
    """
    state = re.sub(r"\s", "", state)
    if len(state) != 9:
        return None

    state = re.sub(r"[^XO]", "-", state.upper())
    return state


def choose_run_form():
    """
    Form to choose a run to visualize the policy and run summary.
    :return:
    """
    runs = __list_runs()
    with st.form("Choose Run"):
        default = None
        if st.session_state.get("run_name") is not None:
            default = runs.index(st.session_state.get("run_name"))

        run = st.selectbox(
            "Choose Run to Visualize:",
            options=runs,
            index=default
        )
        if st.form_submit_button("Visualize"):
            st.session_state["run_name"] = run
            st.rerun()


def evaluate_state_form():
    """
    Form to show action values in a given state.
    :return:
    """
    label = "Write the state to evaluate. Example:\nXOX\n---\nO--"
    default = st.session_state.get("state", "---\n---\n---")

    with st.form("Evaluate State"):
        agent_mark = st.selectbox("Agent Mark:", options=["X", "O"])
        state = st.text_area("State:", placeholder=label, value=default)

        clean = __evaluate_state(state)

        if st.form_submit_button("Select"):
            if clean is None:
                st.error("Unable to parse state!")
            else:
                st.session_state["agent_mark"] = agent_mark
                st.session_state["state"] = clean
