import folium
from folium.plugins import PolyLineOffset
import math

import numpy as np

from constants.utils import COLOR_SCALE
from launch_weekend.cls_sales_rate_regressor import LaunchSalesModel

main_model = LaunchSalesModel(
    min_stock=75,
    min_year=2019,
    y_col='sales_rate'
)

data = main_model.query_raw_data()

data['sales'] = data['sales_rate'] * data['num_of_units']

project_level_data = data.groupby(main_model.project_key).agg(
    {
        'sales': "sum",
        'num_of_units': 'sum',
        'latitude': "mean",
        'longitude': "mean",
        'launch_year': "mean",
        'launch_date': np.unique
    }
)
project_level_data['sales_rate'] = project_level_data['sales'] / project_level_data['num_of_units']

m = folium.Map(
    location=[1.290270, 103.851959],
    tiles="OpenStreetMap",
    zoom_start=10
)

colors = {
    y: c
    for y, c in zip(
        np.sort(project_level_data['launch_year'].astype(int).unique()),
        COLOR_SCALE
    )
}

for idx, proj in project_level_data.iterrows():

    local_deformation = math.cos(proj.latitude * math.pi / 180)
    folium.Circle(
        location=[proj.latitude, proj.longitude],
        popup='%s (%.1f) (%s)' % (proj.name, proj.sales_rate, proj.launch_date[0]),
        radius=proj.sales_rate * 8 * local_deformation,
        color=colors[int(proj.launch_year)],
        fill=True,
        fill_color=colors[int(proj.launch_year)]
    ).add_to(m)



print()