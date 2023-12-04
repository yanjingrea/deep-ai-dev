import pandas as pd
from shapely import wkt

from constants.redshift import query_data
from demand_model.scr_common_training import *
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

from neighborhood_clusters.func_helper_function import plot_feature_heatmap
from demand_model.scr_neighborhood_clusters import *

projects_location = query_data(
    f"""
    select
        b.project_dwid as dw_project_id,
        geom
    from ui_app.land_parcel_summary_dev_sg a
    join ui_app.project_summary_prod_sg b
    on a.land_parcel_dwid = b.land_parcel_dwid
    where b.project_dwid in (
        {','.join(i.__repr__() for i in comparable_demand_model.data.dw_project_id.unique())}
    )
    """
)

projects_location['geom'] = projects_location['geom'].apply(wkt.loads)
projects_location = gpd.GeoDataFrame(
    projects_location,
    geometry='geom'
)


def locate_comparable_projects(
        project_name,
        num_of_bedroom,
        price_range=None
):
    project_data = get_rebased_project_data(
        project_name,
        num_of_bedroom
    ).copy().reset_index()

    if project_data.empty:
        return None

    if price_range is None:
        price_range = (
            project_data[price].min() * 0.9,
            project_data[price].max() * 1.1
        )

    nearby_projects = comparable_demand_model.query_clusters_projects(
        project_data.neighborhood.iloc[0]
    )

    training_data = comparable_demand_model.filter_comparable_projects(
        project_data.dw_project_id.iloc[0],
        num_of_bedroom,
        price_range=price_range,
        nearby_projects=nearby_projects,
        project_data=project_data
    ).copy()

    training_data = pd.merge(
        pd.concat([training_data, project_data], ignore_index=True),
        projects_location,
        on='dw_project_id'
    )[['project_name', 'geom'] + comparable_demand_model.features].drop_duplicates()

    return training_data


def plot_projects_location(
        training_data
):
    if False:

        fig = px.choropleth_mapbox(
            training_data,
            color='price',
            geojson=projects_location.geom,
            locations=training_data.index,
            hover_name="project_name",
            hover_data={
                i: ':.2f' for i in comparable_demand_model.features
            },
            color_continuous_scale=px.colors.cyclical.IceFire,
            mapbox_style="carto-positron",
            zoom=10,
            center={"lat": 1.290270, "lon": 103.851959},
            opacity=0.5
        )

    else:
        fig = px.scatter_mapbox(
            training_data,
            lat=training_data.geom.apply(lambda p: p.centroid.y),
            lon=training_data.geom.apply(lambda p: p.centroid.x),
            center={"lat": 1.290270, "lon": 103.851959},
            color="price",
            size="proj_num_of_units",
            mapbox_style="carto-positron",
            hover_name="project_name",
            hover_data={
                i: ':.2f' for i in comparable_demand_model.features
            },
            color_continuous_scale=px.colors.cyclical.IceFire,
            zoom=10,
            template="plotly_white"
        )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        # autosize=False,
        # width=1200,
        # height=600
    )

    return fig


if True:
    training_data = comparable_demand_model.data[comparable_demand_model.data['launching_period'] == 3]

    training_data = pd.merge(
        training_data,
        projects_location,
        on='dw_project_id'
    )[['project_name', 'geom'] + comparable_demand_model.features].drop_duplicates()

    project_location = cluster_model.plot_clusters()

    project_location.add_scattermapbox(
        lat=training_data.geom.apply(lambda p: p.centroid.y),
        lon=training_data.geom.apply(lambda p: p.centroid.x),
        mode='markers',
        marker=dict(
            size=training_data['proj_num_of_units'] / 50,
            color=training_data['price'],
            colorscale=px.colors.cyclical.IceFire,
            showscale=True,
            colorbar={'orientation': "h"},
            opacity=0.6
        ),
        text=training_data['project_name'],
        name='projects'
    )
    project_location.show()

if False:
    t = locate_comparable_projects(
        'Sanctuary @ Newton',
        3
    )

    fig = plot_projects_location(t)
    fig.show()
