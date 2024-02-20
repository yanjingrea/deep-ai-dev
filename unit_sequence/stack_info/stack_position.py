import pandas as pd
import geopandas as gpd
from shapely import wkt

from constants.redshift import query_data
from constants.utils import get_output_dir

output_dir = get_output_dir(__file__)


project = 'symphony suites'
# project = 'bellewaters'
target = 'median_days_on_market'

trans_datas = query_data(
    f"""
    with
        base as (
                    select
                        *,
                        min(transaction_date::date) over (partition by dw_project_id) as launch_date,
                        datediff(day, launch_date, transaction_date::date) as days_on_market
                    from data_science.ui_master_sg_transactions_view_filled_features_condo
                         join data_science.sg_new_project_geo_view_filled_features_condo
                              using (dw_project_id)
                    where project_name = '{project.replace(" ", "-")}'
                      and property_type_group = 'Condo'
                      and transaction_sub_type = 'new sale'
    
                )
    select
        address_stack::int as address_stack,
        num_of_bedrooms,
        median(days_on_market) as median_days_on_market,
        avg(days_on_market) as mean_days_on_market
    from base
    group by 1, 2
    order by 1, 2
    """
)

position_data = pd.read_csv(
    "/unit_sequence/stack_info"
    f"/orientation_result/{project}_orientation.csv",
    index_col=0
)

geos = ['Polygon', 'Centroid']

position_data['Polygon'] = position_data['Polygon'].apply(wkt.loads)

buildings_gdf = gpd.GeoDataFrame(
    data=position_data,
    geometry='Polygon'
)

buildings_gdf = buildings_gdf.merge(
    trans_datas,
    left_on='Stack',
    right_on='address_stack'
)

ax = buildings_gdf.plot(
    column=target,
    cmap='Reds',
    alpha=.3,
    figsize=(8, 8)
)
buildings_gdf.apply(
    lambda x: ax.annotate(
        text=f"stk {x['address_stack']} \n{x['num_of_bedrooms']}b",
        xy=x['Polygon'].centroid.coords[0],
        ha='center',
        fontsize=8
    ),
    axis=1
)


title = f"{project} stack days on market"
ax.set_title(title)
ax.figure.savefig(
    output_dir + f'{title}.png', bbox_inches='tight', dpi=300
)