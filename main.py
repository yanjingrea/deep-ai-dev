from logzero import logger

import pandas as pd
import streamlit as st

import data_munging
from data_munging import *


padding = 0
st.set_page_config(
    page_title="New Launch Condo Demand Curve",
    layout="wide",
    page_icon="📉"
)

st.markdown(
    """
    <style>
    .small-font {
        font-size:12px;
        font-style: italic;
        color: #b1a7a6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TABLE_PAGE_LEN = 10

st.title("Demand Curve")

projects_choices = list(available_projects.index.get_level_values(0))

with st.sidebar.form(key="my_form"):
    selectbox_project = st.selectbox("Choose a project", projects_choices)
    selectbox_bed_num = st.number_input(
        """Choose num of bedrooms""",
        value=3,
        min_value=1,
        max_value=5,
        step=1,
        format="%i",
    )

    pressed = st.form_submit_button("Generate Demand Curve")

expander = st.sidebar.expander("What is this project?")
expander.write(
    """
    Objective: To model the **price** and **quantity** sold relationship for condominium new launches in Singapore
    Input data: transaction data, project feature data, geospatial data, market condition data, macro-economic factor data
    Output: model that takes in attributes of a building, and output the quantity at various prices
"""
)

network_place, _, descriptor = st.columns([6, 1, 3])
network_loc = network_place.empty()

# Create starting graph
# subhead
descriptor.subheader(
    data_munging.display_project(selectbox_project)
)
# graph
demand_curve_image = data_munging.plot_2d_demand_curve(selectbox_project, selectbox_bed_num)
network_loc.plotly_chart(demand_curve_image)
logger.info("Graph Created, doing app stuff")

if pressed:
    demand_curve_image = data_munging.plot_2d_demand_curve(
        selectbox_project, selectbox_bed_num
    )

    if demand_curve_image is None:
        st.markdown(
            f'Unable to load data of {selectbox_project} {selectbox_bed_num}-bedroom.'
        )

    network_loc.plotly_chart(demand_curve_image)
