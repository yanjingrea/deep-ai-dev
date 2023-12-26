"""
This file serves as an example demonstrating how to utilize the functions for
obtaining the best selling path and optimizing land use configurations.
"""
import numpy as np

from demand_curve_live.scr_get_paths import figure_dir, table_dir
from optimization.cls_revenue_optimization import BestLandModel, BestPathsModels
from optimization.project_config import ProjectConfig
from src.main.building_config import TBuildingConfig
from src.main.land_config import LandConstraints, TLandLot
from demand_curve_live.scr_get_model import linear_models as models

# land_constraints = LandConstraints.from_location(TLandLot('MK10-01653L'))
land_constraints = LandConstraints(
    max_height=5,
    max_dwelling_units=113,
    gross_floor_area=9687.0  # calculate from the project's current config
)

initial_config = ProjectConfig(
    project_name='The Arden',
    launching_year=2023,
    launching_month=8,
    tenure='leasehold',

    total_unit_count=(0, 45, 30, 30, 0),
    avg_unit_size=(0, 64.8, 101.0, 124.7),
    max_floor=5,
    num_of_stacks=3,

    completion_year=2025,
    is_top10_developer=1
)

# refer to 'the-myst', 'midwood', 'dairy-farm-residences', 'the-botany-at-dairy-farm'
unit_size_ranges = {
    2: (60, 75),
    3: (80, 100),
    4: (115, 150)
}

unit_stock_ranges = {
    2: (45, 60),
    3: (20, 40),
    4: (20, 40)
}

config_params = LandConstraints.TConfigParams(
    n_buildings=3,
    unit_size_ranges=unit_size_ranges,
    unit_stock_ranges=unit_stock_ranges,
    max_stacks=18
)

current_price = {
    2: 1760,
    3: 1744,
    4: 1744
}

price_ranges = {
    2: (1750, 1930),
    3: (1680, 1850),
    4: (1710, 1900)
}

max_launching_period = 12
index_to_multiply = 1.0729082749614085
max_growth_psf = 60

paths_model = BestPathsModels(
    demand_models=models,
    initial_config=initial_config
)

# best selling path
if False:

    suggested_paths = paths_model.get_best_selling_paths(
        price_ranges,
        path_lengths={
            2: 6,
            3: 12,
            4: 12
        },
        time_index_to_multiply=index_to_multiply,
        max_growth_psf=max_growth_psf
    )
    fig, ax = suggested_paths.plot()

    project_tr = suggested_paths.revenue
    project_tr_mill = project_tr / 10 ** 6

    fig_name = f"Best Selling Path {initial_config.project_name}\n" \
               f"Total revenue {project_tr_mill: g} millions"

    fig.suptitle(fig_name)

    file_name = f"best selling path {initial_config.project_name} {max_launching_period} periods {project_tr_mill:.2f}m"
    fig.savefig(figure_dir + file_name.replace(' ', '_') + '.png', format='png')

    res = suggested_paths.detailed_dataframe()
    res.to_csv(table_dir + file_name.replace(' ', '_') + '.csv', index=False)

    res2 = suggested_paths.summarized_dataframe()
    res2.to_csv(table_dir + "sum_" + file_name.replace(' ', '_') + '.csv', index=False)

if False:
    current_paths = paths_model.get_current_selling_paths(
        current_price,
        time_index_to_multiply=index_to_multiply,
    )

    project_tr = current_paths.revenue
    project_tr_mill = project_tr / 10 ** 6

    file_name = (
        f"current selling path "
        f"{initial_config.project_name} "
        f"{current_paths.max_length} periods "
        f"{project_tr_mill:.2f}m"
    )
    res = current_paths.summarized_dataframe()
    res.to_csv(table_dir + file_name.replace(' ', '_') + '.csv', index=False)

# best land use
if True:

    land_model = BestLandModel(
        land_constraints=land_constraints,
        demand_models=models,
        initial_config=initial_config
    )

    res = land_model.get_best_land_use(
        config_params,
        price_ranges,
        max_periods={
            2: 6,
            3: 12,
            4: 12
        },
        time_index_to_multiply=index_to_multiply
    )
