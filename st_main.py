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


with st.expander("Visualize Training Run"):
    if "run_name" in st.session_state:
        _, summary = load_data.load_summary_policy(st.session_state["run_name"])
        summary_fig, parsed = plots.plot_summary(summary)

        st.plotly_chart(summary_fig)
        st.markdown(
            (
                "Total Episodes: `%d`\n\n"
                "Played as 'X' `%d`\n\n"
                "Agent: `%s`\n\n"
                "Rival: `%s`\n\n"
                "Percentage Won: `%.3f%%`\n\n"
                "Percentage Lost: `%.3f%%`\n\n"
                "Percentage Drawn: `%.3f%%`\n\n"
            ) % (
                parsed["total_episodes"],
                parsed["played_as_x"],
                parsed["agent_type"],
                parsed["opponent_type"],
                100 * parsed["agent_wins"].sum() / parsed["total_episodes"],
                100 * parsed["agent_losses"].sum() / parsed["total_episodes"],
                100 * parsed["agent_draws"].sum() / parsed["total_episodes"],
            )
        )
