import pandas as pd

from optimization.cls_base_simulation_results import ConfigRevenue
from optimization.cls_revenue_optimization import BestLandModel
from optimization.project_config import ProjectConfig
from optimization.src.main.land_config import LandConstraints

from demand_curve_live.le_quest.func_helper import models, LocalBestPath, LocalBestPaths
from demand_curve_live.le_quest.scr_get_paths import table_dir
from optimization.src.utils.cm_timer import cm_timer

land_constraints = LandConstraints(
    max_height=16,
    max_dwelling_units=519,
    gross_floor_area=37561,
)

initial_config = ProjectConfig(
    project_name='Le Quest',
    launching_year=2017,
    launching_month=8,
    tenure='leasehold',

    total_unit_count=(132, 144, 192, 48, 0),
    avg_unit_size=(45.02, 63.20, 88.52, 115.04, 0),
    max_floor=16,
    num_of_stacks=43,

    completion_year=2020,
    is_top10_developer=1
)

unit_size_ranges = {
    1: (40, 50),
    2: (60, 70),
    3: (80, 95),
    4: (95, 135)
}

unit_stock_ranges = {
    1: (100, 160),
    2: (160, 200),
    3: (150, 192),
    4: (45, 50)
}

config_params = LandConstraints.TConfigParams(
    n_buildings=5,
    unit_size_ranges=unit_size_ranges,
    unit_stock_ranges=unit_stock_ranges,
    max_stacks=80,
)

price_ranges = {
    1: (1325, 1500),
    2: (1350, 1525),
    3: (1275, 1450),
    4: (1250, 1450),
}

if False:
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

path_length = 13

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
            price_ranges=price_ranges,
            path_lengths=path_length,
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

        res2 = suggestion_paths.summarized_dataframe()
        res2.to_csv(table_dir + "sum_" + file_name.replace(' ', '_') + '.csv', index=False)

        res = ConfigRevenue(cfg=c.cfg, paths=suggestion_paths)
        res.summary()
        temp = res.to_dataframe()
        temp['Total Revenue'] = res.revenue

        output_df = pd.concat([output_df, temp], ignore_index=False)

    from demand_curve_live.le_quest.scr_get_paths import table_dir

    tr_mill = output_df['Total Revenue'].max() / 10 ** 6
    output_df.to_csv(
        table_dir +
        f'best_land_use_{initial_config.project_name.replace(" ", "_")}_{tr_mill: .2f}m.csv'
    )

actual_revenue = 547270035.8828 / 10 ** 6
gain = tr_mill / actual_revenue - 1
print(f'total gain is {gain * 100: .2f}%')
