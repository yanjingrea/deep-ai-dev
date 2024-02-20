import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from constants.redshift import query_data
from constants.utils import COLOR_SCALE, get_output_dir

output_dir = get_output_dir(__file__)

with open(
        'evolution_data.sql',
        'r'
) as sql_file:
    sql_script = sql_file.read()

data = query_data(sql_script)

data = data.sort_values('launch_date')
comparable_projects = data['ref_projects'].unique()

initial_projects = []
clusters_dict = {}
sources = []

for idx, row in data.iterrows():

    project = row['project_display_name']
    ref_project = row['ref_projects']

    if (ref_project not in comparable_projects) or (ref_project is None):
        initial_projects += [project]
        clusters_dict[project] = [project]
        sources += [project]

    else:

        for k, v in clusters_dict.items():

            if ref_project in v:
                clusters_dict[k] += [project]
                sources += [k]

lon = 'longitude'
lat = 'latitude'
ref_lon = 'ref_' + lon
ref_lat = 'ref_' + lat

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
                size=dataset['sales_rate'] * 50,
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
        output_dir + 'sequence_map_' + dataset['initial_projects'].iloc[0] + '.html'
    )


data.groupby('initial_projects').apply(plot_sequence_map)
