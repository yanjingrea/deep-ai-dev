from typing import Literal

import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from neighborhood_clusters.neighborhood_features import df as data
from sklearn.cluster import KMeans


def plot_feature_heatmap(
        feature,
        unit: Literal['density', 'absolute']
):
    geo_df = data.copy()

    if (unit == 'density') & ('num_of' in feature):
        geo_df[feature] = geo_df[feature] / geo_df['size_sqkm']

    # to create a color scale that can deal with na value
    na = -0.01 if 'percent' in feature else -1

    pct_of_na = geo_df[geo_df[feature].isna() == True][feature].shape[0] / len(geo_df)
    scale = px.colors.get_colorscale("Agsunset_r")

    scale[0] = [0.0, 'rgb(211,211,211)']
    for idx, i in enumerate(np.linspace(pct_of_na, 1, len(scale) - 1)):
        scale[idx + 1][0] = i

    geo_df[feature] = geo_df[feature].fillna(na)
    fig = px.choropleth_mapbox(
        geo_df,
        color=feature,
        geojson=geo_df.geometry,
        locations=geo_df.index,
        hover_name=geo_df.label,
        color_continuous_scale=scale,
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 1.290270, "lon": 103.851959},
        opacity=0.5
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=False,
        width=1200,
        height=600
    )

    return fig


def plot_elbow(
        cols,
        range_of_clusters_num
):
    X = data[cols].fillna(-1)

    cs = []
    for i in range_of_clusters_num:
        kmeans = KMeans(
            n_clusters=i,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=42
        )
        kmeans.fit(X)
        cs.append(kmeans.inertia_)

    fig = make_subplots()

    fig.add_trace(
        go.Scatter(
            x=np.asarray(range_of_clusters_num),
            y=cs,
            mode="lines+markers",
            marker=dict(
                opacity=0.5,
                size=12
            ),
            line=dict(
                width=4
            ),
            opacity=0.6
        )
    )

    return fig


def plot_clusters(
        cols,
        num_of_clusters
):
    X = data[cols].fillna(-1)
    kmeans = KMeans(
        n_clusters=num_of_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    kmeans.fit(X)

    data['cluster'] = KMeans(
        10, random_state=42, n_init=10
    ).fit_predict(X).astype(str)

    color_discrete_map = {
        str(clu): col
        for clu, col in zip(
            np.arange(num_of_clusters),
            px.colors.qualitative.Light24
        )
    }

    hover_data = {
        i: ':.2f' for i in cols
    }
    hover_data['cluster'] = True
    hover_data['label'] = True

    fig = px.choropleth_mapbox(
        data.fillna(-1),
        color='cluster',
        geojson=data.geometry,
        locations=data.index,
        hover_name=data.label,
        hover_data=hover_data,
        color_discrete_map=color_discrete_map,
        category_orders={'cluster': list(color_discrete_map.keys())},
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 1.290270, "lon": 103.851959},
        opacity=0.5
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=False,
        width=1200,
        height=600
    )

    return fig


if False:
    plot_clusters(
        [
            'num_of_completed_units_condo',
            'average_psf_condo',
            'num_of_completed_units_hdb',
            'average_psf_hdb',
            'num_of_completed_units_landed',
            'average_psf_landed',
            'buyer_profile_company_percent',
            'buyer_profile_foreigner_percent',
            'buyer_profile_pr_percent',
            'buyer_profile_sg_percent'
        ], 10
    )
