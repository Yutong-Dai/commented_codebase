import langchain_helper as lch
import streamlit as st

st.title("Animal Name Generator")
animal_type = st.sidebar.selectbox("What is your pet", ("dog", "cat", "cow", "horse"))
animal_color = st.text_area("What is the color of your {animal_type}", max_chars=20)