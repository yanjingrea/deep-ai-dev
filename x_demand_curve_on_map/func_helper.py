from dataclasses import dataclass

import numpy as np
import geopandas as gpd
from shapely import wkt
import plotly.express as px
import plotly.graph_objects as go

from constants.redshift import query_data
from demand_curve_main.scr_neighborhood_clusters import *
from demand_curve_sep.scr_common_training import *

base_data = query_data(
    f"""
    with base_index as (
        select
            transaction_month_index,
            hi_avg_improved as rebase_index,
            (
                    select hi_avg_improved
                    from data_science.sg_condo_resale_index_sale
                    order by transaction_month_index desc limit 1
            ) as current_index,
            1 / rebase_index * current_index as time_adjust_coef
        from data_science.ui_master_daily_sg_index_sale umdsis
        where property_group = 'condo'
            and index_area_type = 'country'
    ), base_floor_coef as (
        select
            address_floor_num,
            (
                select coefficient
                from data_science_test.partial_coef_address_floor_num_sg_country
                where coef_change = 0
            ) as base_coef,
            1 / (1 + coefficient - base_coef) as floor_adjust_coef
        from data_science_test.partial_coef_address_floor_num_sg_country
    ), base_area_coef as (
        select
            floor_area_sqft as area_lower_bound,
            lag(floor_area_sqft, 1) over (order by floor_area_sqft desc) as next_area,
            case when next_area is null then floor_area_sqft * 1000 else next_area end as area_upper_bound,
            (
                select coefficient
                from data_science_test.partial_coef_floor_area_sqft_sg_country
                where coef_change = 0
            ) as base_coef,
            1 / (1 + coefficient - base_coef) as area_adjust_coef
    from data_science_test.partial_coef_floor_area_sqft_sg_country
    ), base_property_price as (
        select
            *,
            row_number() over (partition by dw_property_id order by transaction_date desc) as seq
        from data_science.ui_master_sg_transactions_view_filled_features_condo a
        join data_science.ui_master_sg_project_geo_view_filled_features_condo b
            using (dw_project_id)
        where a.property_type_group = 'Condo'
            and transaction_sub_type = 'new sale'
            and transaction_date < current_date
    ), base_property_panel as (
        select
            b.dw_project_id,
            to_date(transaction_month_index, 'YYYYMM') as transaction_month,
            (
                select hi_avg_improved
                from data_science.sg_condo_resale_index_sale
                order by transaction_month_index desc limit 1
            ) as current_index,
            avg(
                unit_price_psf
                    * floor_adjust_coef
                    * area_adjust_coef
                    --* zone_adjust_coef
                    * time_adjust_coef
            ) as price,
            avg(floor_area_sqm) as floor_area_sqm,
            count(*) as sales
        from base_property_price a
        left outer join data_science.ui_master_sg_project_geo_view_filled_features_condo b
            using (dw_project_id)
        left outer join  base_index c
            using (transaction_month_index)
        left outer join  base_floor_coef f
            using (address_floor_num)
        left outer join base_area_coef g
           on a.floor_area_sqft >= g.area_lower_bound and a.floor_area_sqft < g.area_upper_bound
        left outer join (
            select
                2023 as transaction_year,
                zone,
                1 / hi_coef as zone_adjust_coef
            from data_science.sg_panel_zone_year_index
            where transaction_year = 2022
            union
            select
                transaction_year,
                zone,
                1 / hi_coef as zone_adjust_coef
            from data_science.sg_panel_zone_year_index
            ) d
            on b.zone = d.zone and left(a.transaction_month_index, 4)::int = d.transaction_year
        group by 1, 2
    )
    select distinct
        dw_project_id,
        c.project_display_name as project_name,
        c.geom,
        to_date(project_launch_month, 'YYYYMM') as launch_year_month,
        avg(price) over (partition by dw_project_id)as price,
        sum(sales) over (partition by dw_project_id)as sales,
        proj_num_of_units,
        avg(floor_area_sqm) over (partition by dw_project_id) as floor_area_sqm,
        tenure,
        proj_max_floor,
        zone,
        neighborhood
    from base_property_panel a
    join data_science.ui_master_sg_project_geo_view_filled_features_condo b
        using(dw_project_id)
    join (
        select
        project_dwid as dw_project_id,
        project_display_name,
        location_marker as geom
        from ui_app.project_summary_prod_sg
     ) c
        using(dw_project_id)
    where (
        to_date(project_launch_month, 'YYYYMM') > dateadd(year, -9, current_date)
    )
    and
    (
        datediff(
            month, 
            to_date(project_launch_month, 'YYYYMM'), 
            transaction_month
        ) + 1 <= 12
    )
    order by 1, 2, 3
    """
)

base_data['geom'] = base_data['geom'].apply(wkt.loads)
base_data = gpd.GeoDataFrame(base_data, geometry='geom')
base_data.set_index('dw_project_id', inplace=True)


def plot_base_map():
    base_map = cluster_model.plot_clusters()
    base_map.update_traces(
        marker_line_color='white',
        marker_line_width=0.5,
    )

    return base_map


def locate_comparable_projects(
        project_name,
        num_of_bedrooms,
        price_range=None
):
    project_data = get_rebased_project_data(
        project_name,
        num_of_bedrooms
    ).copy().reset_index(drop=True)

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
        num_of_bedrooms,
        price_range=price_range,
        nearby_projects=nearby_projects,
        project_data=project_data
    ).copy()

    display_project_data = base_data.loc[[project_data.dw_project_id.iloc[0]]]
    display_training_data = base_data.loc[
        base_data.index.isin(training_data.dw_project_id.unique())
    ]
    display_rest_data = base_data[
        ~base_data.index.isin(training_data.dw_project_id.unique())
    ]
    display_rest_data = display_rest_data[
        display_rest_data['proj_num_of_units'] > 100
        ].copy()

    return LocationParams(
        project_data=display_project_data,
        training_data=display_training_data,
        rest_data=display_rest_data
    )


@dataclass
class LocationParams:
    project_data: pd.DataFrame
    training_data: pd.DataFrame
    rest_data: pd.DataFrame


def plot_project_locations(
        params: LocationParams
):
    base_map = plot_base_map()

    target_data = pd.concat(
        [
            params.training_data,
            params.project_data
        ],
        ignore_index=True
    )

    dty = target_data.dtypes

    def format_hover_data(i):
        if dty[i] == np.dtype('float64'):
            return ':.2f'
        elif dty[i] == np.dtype('int64'):
            return ':.0f'
        else:
            return True

    hover_data = {
        i: format_hover_data(i)
        for i in dty[dty != 'geometry'].index
    }

    def plot_scatter_on_map(
            scatter_data,
            colorscale,
            name
    ):

        hovertemplate = """
            <b>%{hovertext}</b>
                <br>
                    <br>size=%{marker.size}
                    <br>project_name=%{customdata[0]}
                    <br>launch_year_month=%{customdata[1]}
                    <br>price=%{customdata[2]:.2f}
                    <br>sales=%{customdata[3]:.0f}
                    <br>proj_num_of_units=%{customdata[4]:.0f}
                    <br>floor_area_sqm=%{customdata[5]:.2f}
                    <br>tenure=%{customdata[6]}
                    <br>proj_max_floor=%{customdata[7]:.0f}
                    <br>zone=%{customdata[8]}
                    <br>neighborhood=%{customdata[9]}
                    <br>color=%{marker.color}
            <extra></extra>
        """

        base_map.add_trace(
            go.Scattermapbox(
                lat=scatter_data.geom.apply(lambda p: p.centroid.y).values,
                lon=scatter_data.geom.apply(lambda p: p.centroid.x).values,
                marker=go.scattermapbox.Marker(
                    size=scatter_data['proj_num_of_units'].values / 100,
                    color=scatter_data['price'].values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar={'orientation': "h"},
                    opacity=0.7,
                    sizeref=0.027,
                    sizemode='area'
                ),
                customdata=scatter_data[hover_data.keys()].fillna(-1).values,
                hovertext=scatter_data['project_name'].apply(lambda p: p + f' {name}').values,
                hovertemplate=hovertemplate,
                name=name
            )
        )

        return base_map

    for d, c, n in zip(
            [target_data, params.rest_data],
            [px.colors.sequential.Purp, px.colors.cyclical.IceFire],
            ['comparable_data', 'nearby_data']
    ):
        base_map = plot_scatter_on_map(scatter_data=d, colorscale=c, name=n)

    params.project_data.geometry.crs = {'init': 'epsg:4326'}
    center = params.project_data.geom.iloc[0].centroid
    buffer = params.project_data.geometry.to_crs({'init': 'epsg:3857'}).buffer(3000).to_crs(epsg=4326).iloc[0]

    x, y = buffer.exterior.coords.xy

    base_map.add_trace(
        go.Scattermapbox(
            fill='toself',
            lat=y.tolist(),
            lon=x.tolist(),
            mode='lines',
            marker={'size': 10, 'color': "orange", 'opacity': 0.1},
            name='buffer',
            opacity=0
        )
    )

    base_map.update_layout(
        mapbox=dict(
            center={
                "lat": center.y,
                "lon": center.x
            }
        )
    )

    return base_map


a = plot_project_locations(
    locate_comparable_projects(
        'Lentor Modern',
        num_of_bedrooms=3
    )
)
