import streamlit as st
from Input import show_explore_page
from readme import show_readme

st.write ("### CSE 587 Project : Flight Delay Prediction")
readme_but = st.button(" Take me to ReadMe Page")
if readme_but:
    show_readme()

show_explore_page()