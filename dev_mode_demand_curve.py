import streamlit as st
from demand_curve_sep.streamlit_app.func_core_function import *
from x_demand_curve_on_map.func_helper import *

st.set_page_config(
    page_title="New Launch Condo Demand Curve Lab",
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

st.title("Demand Curve")

projects_choices = list(
    available_projects.index.get_level_values(0).unique()
)

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

    pressed = st.form_submit_button("Locate Comparable Projects")

col, _ = st.columns([9, 1])
tab1, tab2 = col.tabs(['map', 'curve'])

if pressed:
    tab1.empty().plotly_chart(
        plot_project_locations(
            locate_comparable_projects(
                selectbox_project,
                selectbox_bed_num
            )
        )
    )

    tab2.empty().plotly_chart(
        plot_2d_demand_curve(
            selectbox_project,
            selectbox_bed_num,
            dev_mode=True
        )
    )
