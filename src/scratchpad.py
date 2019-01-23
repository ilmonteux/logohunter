"""
This scratchpad usess Streamlit which should be installed on your machine if
you follow the Insight Installation Instructions:

https://docs.google.com/presentation/d/1qo_MDz3iF0YRykuElF6I9WC4yAQIYzOA-GY16_NOuUM

Or by running:

pip install -r requirements.txt

from the top-level project folder.
"""

import streamlit as st
import numpy as np

st.write('This is a scratchpad for *Streamlit.* **Edit it and see what happens!**')

st.subheader('A Numpy Array')

st.write(np.random.randn(10, 10))

st.subheader('A Graph!')

st.line_chart(np.random.randn(200, 2))
