import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import pyplot as plt

from constants.redshift import query_data
from constants.utils import COLOR_SCALE, get_output_dir

output_dir = get_output_dir(__file__)

target_projects = [
    'Lentoria',
    'The Hill @ One North'
]


lon = 'longitude'
lat = 'latitude'
ref_lon = 'ref_' + lon
ref_lat = 'ref_' + lat

sql_query = ','.join(i.__repr__() for i in target_projects)



data = query_data(
    f"""
    with
    base_index as (
                      select
                          index_date,
                          quarter_index
                      from ui_app.house_index_summary_prod_sg
                      where location_level = 'country'
                      and property_type_group = 'private-stack'
                      and unit_mix = 'all'
                      and activity_type = 'sale'
                      order by index_date desc
                  ),
    base_trans as (
                      select
                          neighborhood_id,
                          project_dwid,
                          property_dwid,
                          activity_psf/quarter_index * (select quarter_index from base_index limit 1) as activity_psf,
                          activity_date_index as activity_date,
                          days_on_market
                      from (
                               select
                                   neighborhood_id,
                                   project_dwid,
                                   property_dwid,
                                   activity_psf,
                                   to_date(left(activity_date, 7), 'YYYY-MM') as activity_date_index,
                                           min(activity_date)
                                           over (partition by project_dwid) as actual_launch_date,
                                   datediff(days, actual_launch_date, activity_date) as days_on_market,
                                           row_number()
                                           over (partition by project_dwid, property_dwid order by activity_date) as seq
                               from ui_app.transacted_summary_prod_sg t
                               where t.property_group = 'condo'
                                 and t.activity_type = 'new-sale'
                                 and t.property_type != 'executive condominium'
                                 and unit_mix in (0, 1, 2, 3, 4, 5, 6)
                           ) as a
                      join base_index b
                        on a.activity_date_index = b.index_date
                      where seq = 1
                  ),
    base_launch_sales as (
                             select
                                 project_dwid,
                                 count(*) as sales,
                                 avg(activity_psf) as price_psf
                             from base_trans
                             where days_on_market <= 7
                             group by 1
                         )
    , base as (
                  select
                      project_dwid,
                      project_display_name,
                      case
                          when project_display_name in ({sql_query}) then '2024-02-01'
                          else launch_date end as launch_date,
                      latitude,
                      longitude,
                      price_psf,
                      sales,
                      residential_unit_count as num_of_units,
                      sales::float/num_of_units::float as sales_rate
                  from ui_app.project_summary_prod_sg
                  left join base_launch_sales
                    using (project_dwid)
                  where (
                              property_group = 'condo'
                          and property_type != 'ec'
                          and residential_unit_count != 0
                          and (
                                      (launch_date >= '2019-01-01') or
                                      (project_display_name in ({sql_query}))
                                  )
                      )
              )

    select
    p_base.project_display_name,
    p_base.{lon},
    p_base.{lat},
    p_base.launch_date,
    p_base.price_psf,
    p_base.sales,
    p_base.num_of_units,
    p_base.sales_rate,
    c_base.{lon} as {ref_lon},
    c_base.{lat} as {ref_lat},
    c_base.project_display_name as ref_project,
    c_base.launch_date as ref_launch_date,
    c_base.price_psf as ref_price_psf,
    c_base.sales as ref_sales,
    c_base.num_of_units as ref_num_of_units,
    c_base.sales_rate as ref_sales_rate,
    ST_DistanceSphere(
            st_point(p_base.longitude, p_base.latitude),
            st_point(c_base.longitude, c_base.latitude)
    ) as distance,
    datediff(day, c_base.launch_date::date, p_base.launch_date::date) as days_gap,
            dense_rank()
            over (partition by p_base.project_display_name order by days_gap) as rank,
    count(*) over (partition by p_base.project_display_name) as num_of_comparables
from base as p_base
     left join base as c_base
               on ST_DistanceSphere(
                          st_point(p_base.longitude, p_base.latitude),
                          st_point(c_base.longitude, c_base.latitude)
                  ) <= 3000
                   and datediff(day, c_base.launch_date::date, p_base.launch_date::date) between 0 and 365 * 3
                    --and abs(c_base.price_psf / p_base.price_psf - 1) <= 0.5
                   and abs(c_base.num_of_units / p_base.num_of_units - 1) <= 0.5
                   and p_base.project_dwid != c_base.project_dwid
order by p_base.launch_date desc
    """
)


data = data.sort_values('launch_date')
comparable_projects = data['project_display_name'].unique()

initial_projects = []
clusters_dict = {}
sources = []

for idx, row in data.iterrows():

    project = row['project_display_name']
    ref_project = row['ref_project']

    if (ref_project not in comparable_projects) or (ref_project is None):
        initial_projects += [project]
        clusters_dict[project] = [project]
        sources += [project]

    else:

        sourced = False

        for k, v in clusters_dict.items():

            if ref_project in v:
                clusters_dict[k] += [project]
                sources += [k]

                sourced = True

                break

        if not sourced:
            clusters_dict[project] = [project]
            sources += [project]




data['launch_year'] = data['launch_date'].apply(lambda a: int(a[:4]))
data['initial_projects'] = sources
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
                size=dataset['sales_rate'].fillna(0.0001) * 50,
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

        if row['ref_project'] is None:
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
                name=f"{row['ref_project']} -> {row['project_display_name']}"
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
        output_dir + 'sequence_map_' + dataset['initial_projects'].iloc[0] + '.html'
    )


target_projects_data = data[data['project_display_name'].isin(target_projects)]
plot_data = data[data['initial_projects'].isin(target_projects_data['initial_projects'])]
plot_data.groupby('initial_projects').apply(plot_sequence_map)


def plot_clusters(dataset):

    x = dataset['price_psf']
    y = dataset['sales_rate']

    if len(x) == 1:
        return None

    local_color = dataset['launch_year'].apply(lambda a: colors[a])

    fig, ax = plt.subplots(figsize=(12, 8))

    import seaborn as sns
    sns.scatterplot(
        data=dataset,
        x=x,
        y=y,
        hue='launch_year',
        palette=colors,
        # size=80,
        alpha=0.5,
        zorder=10,
        edgecolors=COLOR_SCALE[-1],
    )

    # ax.scatter(
    #     x,
    #     y,
    #     edgecolors=COLOR_SCALE[-1],
    #     s=80,
    #     color=local_color,
    #     alpha=0.5,
    #     zorder=10
    # )

    # ax.legend()

    for local_idx, local_row in dataset.iterrows():

        ref_x = local_row['ref_price_psf']
        ref_y = local_row['ref_sales_rate']

        proj_x = local_row['price_psf']
        proj_y = local_row['sales_rate']

        ax.text(
            proj_x * 1.001,
            proj_y * 1.001,
            local_row['project_display_name']  # + f'\nprice: {proj_x: .0f}' + f'\nsales rate: {proj_y * 100: .1f}%'
        )

        if local_row['ref_project'] is None:
            continue

        adjust_coef = 0.9

        dx = (proj_x - ref_x) * adjust_coef
        dy = (proj_y - ref_y) * adjust_coef

        ax.arrow(
            x=ref_x,
            y=ref_y,
            dx=dx,
            dy=dy,
            color=colors[local_row['launch_year']],
            alpha=0.8,
            head_width=0.02,  # arrow head width
            head_length=0.5
            # arrow head length
        )

    fig.savefig(
        output_dir + 'sequence_scatter_' + dataset['initial_projects'].iloc[0] + '.png', dpi=300
    )

plot_data.groupby('initial_projects').apply(plot_clusters)