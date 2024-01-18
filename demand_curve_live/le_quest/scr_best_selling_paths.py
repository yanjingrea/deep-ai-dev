import numpy as np
import pandas as pd

from demand_curve_live.jadescape.func_helper import (
    calculate_actual_revenue,
    calculate_path_revenues,
)
from optimization.cls_base_simulation_results import ProjectSalesPaths
from optimization.project_config import ProjectConfig

# back test
# ------------------------------------------------------------------------
if False:
    true_paths = {}
    pred_paths = {}

    for bed_num in np.arange(1, 6):

        true_bed_revenue = calculate_actual_revenue(bed_num)
        pred_bed_revenue = calculate_path_revenues(bed_num)

        n_trans = true_bed_revenue.quantity_path.sum()
        actual_stock = {
            1: 236,
            2: 403,
            3: 265,
            4: 261,
            5: 39
        }.get(bed_num, lambda a: a)
        actual_revenue = true_bed_revenue.total_revenue / n_trans * actual_stock
        true_bed_revenue.__setattr__('total_revenue', actual_revenue)

        quantity_diff = pred_bed_revenue.quantity_path.sum() - actual_stock
        error = pred_bed_revenue.total_revenue / actual_revenue - 1

        print(
            f"""
            {bed_num}-bedroom
            actual price path:          {true_bed_revenue.psf_path.astype(int)}
            actual quantity path:       {true_bed_revenue.quantity_path}
            predictive quantity path:   {pred_bed_revenue.quantity_path}
            cumulative quantity error:  {quantity_diff}

            actual total revenue:           {actual_revenue / 10 ** 6: .2f} million dollars
            predictive total revenue:       {pred_bed_revenue.total_revenue / 10 ** 6: .2f} million dollars
            relative error:   {error * 100: .2f}%
            """
        )

        true_paths[bed_num] = true_bed_revenue
        pred_paths[bed_num] = pred_bed_revenue

    actual_paths = ProjectSalesPaths(true_paths)
    predictive_paths = ProjectSalesPaths(pred_paths)

    total_relative_error = predictive_paths.revenue / actual_paths.revenue - 1

    print(
        f"""
        the whole project
        actual total revenue:               {actual_paths.revenue / 10 ** 6: .2f} million dollars
        predictive total revenue:           {predictive_paths.revenue / 10 ** 6: .2f} million dollars
        relative error of total revenue:    {total_relative_error * 100: .2f}%
        """
    )

# best selling path
# ------------------------------------------------------------------------
if True:
    initial_config = ProjectConfig(
        project_name='Jadescape',
        launching_year=2018,
        launching_month=9,
        tenure='leasehold',

        total_unit_count=(236, 403, 265, 261, 39),
        avg_unit_size=(49.4, 68.7, 94.0, 131.9, 195.0),
        max_floor=23,
        num_of_stacks=55,

        completion_year=2023,
        is_top10_developer=1
    )

    paths_model = LocalBestPath(
        demand_models=models,
        initial_config=initial_config
    )

    print()