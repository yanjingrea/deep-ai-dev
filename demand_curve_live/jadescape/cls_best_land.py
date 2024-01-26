import pandas as pd

from optimization.cls_base_simulation_results import ConfigRevenue, PathGeneratorParams
from optimization.cls_revenue_optimization import BestLandModel
from optimization.project_config import ProjectConfig
from optimization.src.main.land_config import LandConstraints

from demand_curve_live.jadescape.func_helper import models, LocalBestPath, LocalBestPaths
from demand_curve_live.jadescape.scr_get_paths import table_dir
from optimization.src.utils.cm_timer import cm_timer

land_constraints = LandConstraints(
    max_height=44,
    max_dwelling_units=1251,
    gross_floor_area=106285.4,  # calculate from the project's current config (9687)
)

initial_config = ProjectConfig(
    project_name='Jadescape',
    launching_year=2018,
    launching_month=9,
    tenure='leasehold',

    total_unit_count=(236, 403, 265, 261, 39),
    avg_unit_size=(49, 68.65, 94.0, 131.89, 195.0),
    max_floor=23,
    num_of_stacks=55,

    completion_year=2023,
    is_top10_developer=1
)


unit_size_ranges = {
    1: (49, 50),
    2: (60, 70),
    3: (80, 95),
    4: (115, 135),
    5: (150, 195)
}

unit_stock_ranges = {
    1: (200, 300),
    2: (300, 350),
    3: (265, 320),
    4: (200, 300),
    5: (35, 70)
}

config_params = LandConstraints.TConfigParams(
    n_buildings=7,
    unit_size_ranges=unit_size_ranges,
    unit_stock_ranges=unit_stock_ranges,
    max_stacks=80,
)

price_ranges = {
    1: (1625, 1850),
    2: (1625, 1800),
    3: (1665, 1750),
    4: (1640, 1750),
    5: (1550, 1730)
}

path_params = {
    1: {
        1: PathGeneratorParams(path_length=9, price_range=(1625, 1800)),
        2: PathGeneratorParams(path_length=8, price_range=(1800, 1880))
    },
    2: {
        1: PathGeneratorParams(path_length=9, price_range=(1625, 1800)),
        2: PathGeneratorParams(path_length=8, price_range=(1880, 1880))
    },
    3: {
        1: PathGeneratorParams(path_length=9, price_range=(1665, 1750)),
        2: PathGeneratorParams(path_length=8, price_range=(1750, 1850))
    },
    4: {
        1: PathGeneratorParams(path_length=9, price_range=(1640, 1750)),
        2: PathGeneratorParams(path_length=8, price_range=(1750, 1865))
    },
    5: {
        1: PathGeneratorParams(path_length=9, price_range=(1550, 1625)),
        2: PathGeneratorParams(path_length=8, price_range=(1625, 1725))
    }
}

path_length = 17

new_transformed_models = {
    i: LocalBestPath(
        num_of_bedrooms=i,
        demand_model=models[i],
        initial_config=initial_config
    )
    for i in models.keys()
}

land_model = BestLandModel(
    land_constraints=land_constraints,
    demand_models=models,
    initial_config=initial_config,
    transformed_models=new_transformed_models
)

with cm_timer(fmt=f'getting the best selling paths ' + '{:g} s'):
    promising_configs = land_model.get_promising_configs(
        config_params,
        price_ranges,
        max_periods=path_length,
        output_num=5
    )

    output_df = pd.DataFrame()
    for c in promising_configs:

        if c == initial_config:
            continue

        paths_model = LocalBestPaths(
            demand_models=models,
            initial_config=initial_config,
            new_config=c.cfg
        )

        suggestion_paths = paths_model.get_best_selling_paths(
            path_params,
            discount_rate=0.0125
        )

        paths_record = suggestion_paths.detailed_dataframe()
        paths_record['num_of_units'] = paths_record['bed_num'].apply(
            lambda b: c.cfg.get_units_count(bed=b)
        )
        paths_record['area_sqm'] = paths_record['bed_num'].apply(
            lambda b: c.cfg.avg_unit_size_per_bed(bed=b)
        )

        project_tr_mill = suggestion_paths.revenue / 10 ** 6
        file_name = f"best selling path of best config {c.cfg.project_name} {project_tr_mill: .2f}m.csv"
        paths_record.to_csv(table_dir + file_name.replace(' ', '_'))

        res = ConfigRevenue(cfg=c.cfg, paths=suggestion_paths)
        res.summary()
        temp = res.to_dataframe()
        temp['Total Revenue'] = res.revenue

        output_df = pd.concat([output_df, temp], ignore_index=False)

    from demand_curve_live.the_arden.scr_get_paths import table_dir

    tr_mill = output_df['Total Revenue'].max() / 10 ** 6
    output_df.to_csv(
        table_dir +
        f'best_land_use_{initial_config.project_name.replace(" ", "_")}_{tr_mill: .2f}m.csv'
    )

actual_revenue = 1948056998.38 / 10 ** 6
gain = tr_mill / actual_revenue - 1
print(f'total gain is {gain * 100: .2f}%')
