import numpy as np

from constants.redshift import query_data
import plotly.graph_objects as go
import plotly.express as px

from constants.utils import COLOR_SCALE, get_output_dir

output_dir = get_output_dir(__file__)


data = query_data(
    f"""
    with
        base_launch_data as (
                                select
                                    dw_project_id,
                                    min(transaction_date) as launch_date
                                from (
                                         select
                                             *,
                                                     row_number()
                                                     over (partition by dw_property_id order by transaction_date desc) as seq
                                         from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                              join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                   using (dw_project_id)
                                         where a.property_type_group = 'Condo'
                                           and transaction_sub_type = 'new sale'
                                           and property_type != 'ec'
                                     ) as "*2"
                                where seq = 1
                                group by 1
                            )
            ,
        base_sales as (
                          select
                              a.dw_project_id,
                              min(to_date(transaction_month_index, 'YYYYMM')) as transaction_month,
                              launch_date,
                              (
                                  select
                                      non_landed_index
                                  from developer_tool.sg_gov_residential_index
                                  order by quarter_index desc
                                  limit 1
                              ) as current_index,
                              count(dw_property_id) as sales,
                              avg(unit_price_psf / c.non_landed_index * current_index) as average_launch_psf,
                              avg(transaction_amount / c.non_landed_index * current_index) as average_launch_price
                          from base_launch_data a
                               left join data_science.ui_master_sg_transactions_view_filled_features_condo b
                                         on a.dw_project_id = b.dw_project_id
                                             and b.transaction_date::date <= dateadd(day, 7, launch_date::date)
                               join developer_tool.sg_gov_residential_index c
                                    on b.transaction_quarter_index = c.quarter_index
                          group by 1, launch_date
                      )
            ,
        base as (
                    select
                        dw_project_id,
                        project_display_name,
                        case when project_display_name in ('Hillhaven', 'The Arcady At Boon Keng') then '2024-01-01' else launch_date end as launch_date,
                        transaction_month,
                        case when sales is null then 0 else sales end as sales,
                        proj_num_of_units,
                        average_launch_psf,
                        latitude,
                        longitude,
                        region_group
                    from data_science.ui_master_sg_project_geo_view_filled_features_condo b
                         left join base_sales c
                                   using (dw_project_id)
                         left join (
                                       select
                                           *
                                       from (
                                                select
                                                    project_dwid as dw_project_id,
                                                    project_display_name,
                                                    latitude,
                                                    longitude
                                                from ui_app.project_summary_prod_sg
                                                where property_group = 'condo'
                                                  and property_type != 'ec'
                                                  and launch_date is not null
                                            ) a
                                   ) as project_static
                                   using (dw_project_id)
                    where left(project_launch_month, 4)::int >= 2019 and proj_num_of_units >= 75
                ),
        base_com as (
                        select
                            p_base.project_display_name,
                            p_base.launch_date,
                            p_base.sales::float / p_base.proj_num_of_units::float as sales_rate,
                            p_base.longitude,
                            p_base.latitude,
                            p_base.region_group,
                            c_base.project_display_name as ref_projects,
                            c_base.launch_date as ref_launch_date,
                            c_base.sales::float / c_base.proj_num_of_units::float as ref_sales_rate,
                            c_base.longitude as ref_longitude,
                            c_base.latitude as ref_latitude,
                            
                            ST_DistanceSphere(
                                    st_point(p_base.longitude, p_base.latitude),
                                    st_point(c_base.longitude, c_base.latitude)
                            ) as distance,
                                    dense_rank()
                                    over (partition by p_base.project_display_name order by datediff(day, c_base.launch_date::date, p_base.launch_date::date)) as rank
                        from base as p_base
                             left join base as c_base
                                  on ST_DistanceSphere(
                                             st_point(p_base.longitude, p_base.latitude),
                                             st_point(c_base.longitude, c_base.latitude)
                                     ) <= 3000
                                      and p_base.launch_date > c_base.launch_date
                        order by p_base.launch_date, rank, distance
                    )
    select
        *
    from base_com
    where rank = 1 and project_display_name is not null
    """
)

lon = 'longitude'
lat = 'latitude'
ref_lon = 'ref_' + lon
ref_lat = 'ref_' + lat

data['launch_year'] = data['launch_date'].apply(lambda a: int(a[:4]))
dty = data.dtypes


def format_hover_data(i):
    if dty[i] == np.dtype('float64'):
        return ':.2f'
    elif dty[i] == np.dtype('int64'):
        return ':.0f'
    else:
        return True


hover_data = {
    i: format_hover_data(i)
    for i in dty.index
}

lst = "\n".join(
    [
        f"<br>{n}=%" +
        "{" + f"customdata[{idx}]" +
        f":{hover_data[n]}" + "}"
        for idx, n in enumerate(hover_data.keys())
    ]
)

hovertemplate = """
            <b>%{hovertext}</b>
                <br>
                    lst
            <extra></extra>
        """.replace("lst", lst)

colors = {
    y: c
    for y, c in zip(
        np.sort(data['launch_year'].unique()),
        COLOR_SCALE
    )
}


def plot_sequence_map(dataset):

    fig = px.choropleth_mapbox(
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 1.290270, "lon": 103.851959},
        opacity=0.5
    )

    fig.add_trace(
        go.Scattermapbox(
            lon=dataset[lon],
            lat=dataset[lat],
            marker=go.scattermapbox.Marker(
                size=dataset['sales_rate'] * 80,
                color=dataset['launch_year'].apply(lambda y: colors[y]),
                # color=dataset['launch_year'],
                opacity=0.3,
                sizemode='area',
                sizeref=0.027,
            ),
            customdata=dataset,
            hovertext=dataset['project_display_name'],
            hovertemplate=hovertemplate,
            name='project'
        )
    )

    for idx, row in dataset.iterrows():

        if row['ref_projects'] is None:
            continue

        final_color = colors[row['launch_year']]

        fig.add_trace(
            go.Scattermapbox(
                lon=[row[lon], row[ref_lon]],
                lat=[row[lat], row[ref_lat]],
                mode='lines',
                line=dict(
                    width=2,
                    color=final_color
                ),
                name=f"{row['ref_projects']} -> {row['project_display_name']}"
            )
        )

        adjust_coef = 10 ** 3

        l = 1.1  # the arrow length
        widh = 0.35  # 2*widh is the width of the arrow base as triangle

        A = np.array([row[ref_lon], row[ref_lat]])
        B = np.array([row[lon], row[lat]])
        v = B - A
        w = v / np.linalg.norm(v)
        u = np.array([-w[1], w[0]])  # u orthogonal on w

        P = B - l * w / adjust_coef
        S = P - widh * u / adjust_coef
        T = P + widh * u / adjust_coef

        final_lon = [a[0] for a in [S, T, B, S]]
        final_lat = [a[1] for a in [S, T, B, S]]

        fig.add_trace(
            go.Scattermapbox(
                lon=final_lon,
                lat=final_lat,
                mode='lines',
                fill='toself',
                fillcolor=final_color,
                line_color=final_color,
                showlegend=False
            )
        )

        print()

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=False,
        width=1200,
        height=600
    )

    fig.write_html(
        output_dir + 'sequence_map_' + dataset['region_group'].iloc[0] +'.html'
    )


data.groupby('region_group').apply(plot_sequence_map)


