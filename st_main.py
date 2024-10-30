import streamlit as st

from ttt_visualize.forms import choose_run_form, evaluate_state_form


st.set_page_config(
    page_title="Tic Tac Toe",
)

choose_run_form()

with st.expander("Visualize Policy"):
    if "run_name" in st.session_state:
        evaluate_state_form()
