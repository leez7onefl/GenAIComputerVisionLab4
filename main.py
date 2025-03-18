import streamlit as st

ex1_page = st.Page("exercice1.py", title="Ex. 1", icon="")
noise_test_page = st.Page("test_noise.py", title="Noise function", icon="")
ex2_page = st.Page("exercice2bis.py", title="Ex. 2", icon="")
questions_page = st.Page("questions.py", title="Questions", icon="")
pg = st.navigation([ex1_page, noise_test_page, ex2_page, questions_page])
st.set_page_config(page_title="Lab4 GenAI Computer Vision", page_icon="👀")
pg.run()