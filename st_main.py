import streamlit as st

from ttt_visualize import plots, load_data
from ttt_visualize.forms import choose_run_form, evaluate_state_form


st.set_page_config(
    page_title="Tic Tac Toe",
)

choose_run_form()

with st.expander("Visualize Policy"):
    if "run_name" in st.session_state:
        evaluate_state_form()
        policy, _ = load_data.load_summary_policy(st.session_state["run_name"])

        if "state" in st.session_state:
            fig = plots.plot_policy_state(
                st.session_state["state"],
                agent_mark=st.session_state["agent_mark"],
                qs=policy
            )
            if fig is None:
                st.error("State was not visited during run!")
            else:
                st.plotly_chart(fig)
