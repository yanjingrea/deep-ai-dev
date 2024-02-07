import pandas as pd
from matplotlib import pyplot as plt

from DeepAI_weekly_report.scr_get_paths import dev_data_dir, dev_figure_dir, report_dir
from unit_sequence.RankerEvaluator import plot_random_u_curve, RankerEvaluator

tested_data = pd.read_csv(dev_data_dir + 'property_selling_sequence.csv')

min_year = 2020,
min_stock = 150,
n_groups = 5
highlight_projects = [
    'Hillhaven',
    'The Arcady At Boon Keng',
    'Lumina Grand'
]

# U curve
# ---------------------------------------------------------
fig, ax = plot_random_u_curve()
evaluator = RankerEvaluator(
    raw_data=tested_data,
    actual_label='actual_label',
    pred_label='actual_ranking',
    groupby_keys=['project_name', 'num_of_bedrooms'],
    price_difference_col='price_relative_difference'
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
    control_stock=min_stock,
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
