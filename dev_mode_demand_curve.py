import streamlit as st
from demand_model.streamlit_web_app.core_function import *
from demand_model.scr_project_location import *

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

col1, col2 = st.columns([5, 5])

with col1:

    filtered_data = locate_comparable_projects(
        selectbox_project,
        selectbox_bed_num
    )

    filtered_projects = filtered_data.project_name.unique()

    multi_projects = st.multiselect(
        f"""select projects to display""",
        filtered_projects,
        default=filtered_projects
    )

    min_p = filtered_data.price.min()
    max_p = filtered_data.price.max() * 1.2

    selected_price_range = st.slider(
        f"select price range",
        min_p * 0.8,
        max_p * 1.2,
        (min_p * 0.9, max_p * 1.1)
    )

    pressed_col1 = st.button("Display")

    if pressed_col1:
        container = st.container()
        multi_filtered_projects = filtered_data[
            (filtered_data['project_name'].isin(multi_projects)) &
            (filtered_data['price'].between(*selected_price_range))
            ]

        container.empty().plotly_chart(
            plot_projects_location(multi_filtered_projects)
        )
