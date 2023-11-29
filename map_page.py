import streamlit as st
from neighborhood_clusters.helper_function import *

st.set_page_config(
    page_title="Neighborhood Clustering",
    layout="wide",
    page_icon="📍"
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

tab1, tab2 = st.tabs(['heatmap', 'clustering'])
with tab1:
    st.title("Singapore New Launch Condo Heatmap")
    tab1_col1, tab1_col2, _ = st.columns([3, 6, 1])
    tab1_network_col = tab1_col2.empty()

    feature_choices = list(data.columns.difference(['geometry', 'label']))
    with tab1_col1:
        selectbox_feature = st.selectbox("Choose a feature", feature_choices)
        selectbox_unit = st.selectbox("Choose a unit", ['density', 'absolute'])
        pressed_button1 = st.button("Visualize Feature Heatmap")

    heatmap = plot_feature_heatmap(selectbox_feature, selectbox_unit)
    with tab1_network_col:
        st.plotly_chart(heatmap)

    if pressed_button1:
        heatmap = plot_feature_heatmap(selectbox_feature, selectbox_unit)
        with tab1_network_col:
            tab1_network_col.plotly_chart(heatmap)

with tab2:
    st.title("Singapore Neighborhood Clustering")

    col1, col2, _ = st.columns([3, 6, 1])
    network_col = col2.empty()

    tab2_feature_choices = list(
        data.columns.difference(
            ['geometry', 'label', 'cluster']
        )
    )

    with col1:
        selectbox_feature = st.multiselect(
            "Choose features",
            feature_choices,
            default=feature_choices
        )
        selectbox_elbow = st.slider(
            'please decide the range of clusters num for elbow method',
            0, 30, (5, 15)
        )
        selectbox_clusters_num = st.number_input(
            """Choose num of clusters""",
            value=10,
            min_value=2,
            max_value=20,
            step=1,
            format="%i",
        )
        pressed = st.button("Visualize Clusters")

    if pressed:
        with network_col:
            with st.container():
                con_tab1, con_tab2 = st.tabs(['elbow', 'clusters'])

                cluster_map = plot_clusters(selectbox_feature, selectbox_clusters_num)

                con_tab1.plotly_chart(
                    plot_elbow(
                        selectbox_feature,
                        range_of_clusters_num=range(*selectbox_elbow)
                    )
                )

                con_tab2.plotly_chart(cluster_map)
