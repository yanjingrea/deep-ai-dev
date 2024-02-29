"""
# Overview:
# This script is part of a suite of tools for ranking and evaluating property unit sequences.
It leverages the CatBoostAVMRanker for training on historical data and evaluates performance using U-curves and Tower View visualizations.
# It's designed to assess and visualize the expected selling sequence of units within new property projects.
"""


import pickle

import pandas as pd
from matplotlib import pyplot as plt

from DeepAI_weekly_report.test.cls_paths_collections import PathsCollections
from unit_sequence.CatBoostAVMRanker import CatBoostAVMRanker
from unit_sequence.RankerEvaluator import plot_random_u_curve, RankerEvaluator
from DeepAI_weekly_report.scr_get_paths import (
    dev_data_dir, td, dev_figure_dir, report_dir, dev_res_dir
)

# test params
# -------------------------
min_year = 2020,
min_stock = 150,
n_groups = 5
highlight_projects = [
    'Hillhaven',
    'The Arcady At Boon Keng',
    'Lumina Grand'
]

# training
# -------------------------
common_params = dict(
    target='ranking', metrics='relative', min_year=2018
)
ranker_model = CatBoostAVMRanker(**common_params).fit()
latest_projects = ranker_model.raw_data[
    (ranker_model.raw_data['project_launch_date'].dt.year >= min_year) &
    (ranker_model.raw_data['proj_num_of_units'] >= min_stock)
    ]['project_name'].sort_values().unique()

n_latest_projects = len(latest_projects)
n_project_per_group = n_latest_projects // n_groups

tested_data = pd.DataFrame()
image_paths = []

for i in range(0, n_latest_projects, n_project_per_group):

    temp_test_projects = latest_projects[i:i + n_project_per_group]

    temp_train_data = ranker_model.raw_data[~ranker_model.raw_data['project_name'].isin(temp_test_projects)]
    temp_test_data = ranker_model.raw_data[ranker_model.raw_data['project_name'].isin(temp_test_projects)]

    ranker_model._fit_ranker(temp_train_data)

    tested_data = pd.concat(
        [
            tested_data,
            ranker_model._test_ranker(temp_test_data)
        ], ignore_index=True
    )

tested_data.to_csv(dev_data_dir + 'property_selling_sequence.csv', index=False)

# U curve
# ---------------------------------------------------------
fig, ax = plot_random_u_curve()
evaluator = RankerEvaluator(
    raw_data=tested_data,
    actual_label=ranker_model.actual_label,
    pred_label=ranker_model.pred_label,
    groupby_keys=ranker_model.groupby_keys,
    price_difference_col=ranker_model.price_difference_col
)

fig, ax = evaluator.plot_scatter_above_u_curve(
    fig=fig,
    ax=ax,
    control_stock=min_stock,
    highlight_projects=highlight_projects
)

fig, ax = evaluator.plot_project_level_fitted_curve(
    fig=fig,
    ax=ax,
    control_stock=min_stock
)

title = f'project level u curve'
ax.set_title(title)

report_path = title.replace('-', '_').replace(' ', '_')
plt.savefig(dev_figure_dir + f'{report_path}.png', dpi=300)
plt.savefig(report_dir + f'{report_path}.png', dpi=300)

projects_score_table = evaluator.projects_score_table
mid_big_projects = projects_score_table[projects_score_table['stock'] > min_stock]

print(projects_score_table)
print(f"overall gain {projects_score_table['gain'].mean() * 100 :g}%")
print(f"mid-big projects gain {mid_big_projects['gain'].mean() * 100 :g}%")

projects_score_table.to_csv(
    dev_data_dir + f'u_curve_projects_score_table.csv', index=False
)

if False:
    # Tower View
    for proj in highlight_projects:

        project_data = tested_data[tested_data['project_name'] == proj].copy()
        available_beds = project_data['num_of_bedrooms'].unique()

        for bed in available_beds:
            bed_data = project_data[project_data['num_of_bedrooms'] == bed]

            fig, ax = ranker_model.plot_tower_view(
                project_data=bed_data,
                values=ranker_model.pred_ranking
            )

            title = f'Selling Sequence {proj} {bed}-bedroom'
            ax.set_title(title)

            report_path = title.replace('-', '_').replace(' ', '_')
            plt.savefig(dev_figure_dir + report_path + '.png', dpi=300)
            plt.savefig(report_dir + report_path + '.png', dpi=300)

            image_paths += [
                PathsCollections(
                    project_name=proj,
                    num_of_bedrooms=bed,
                    paths=report_path
                )
            ]

paths_df = pd.DataFrame(image_paths)
image_paths_des = dev_res_dir + 'u_curve_paths_df.plk'
pickle.dump(paths_df, open(image_paths_des, 'wb'))
