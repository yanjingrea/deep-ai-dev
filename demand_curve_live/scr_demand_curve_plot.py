import os
import pickle
from os.path import dirname, realpath
import seaborn as sns

from demand_curve_sep.scr_common_training import (
    price,
    quantity,
    get_adjusted_project_data,
    comparable_demand_model, get_rebased_project_data
)
from demand_curve_sep.cls_plt_demand_curve import *
from demand_curve_live.scr_get_paths import *

project_name = 'The Arden'
launching_period = 1
bedrooms_list = [2, 3, 4]

models_path = model_dir + f'{project_name}'.replace(' ', '_')

linear_models = {}
for num_of_bedroom in bedrooms_list:

    rebased_project_data = get_rebased_project_data(
        project_name,
        num_of_bedroom
    )

    adjusted_project_data = get_adjusted_project_data(
        project_name,
        num_of_bedroom
    )

    project_id = adjusted_project_data.dw_project_id.iloc[0]

    price_range = (
        adjusted_project_data[price].min() * 0.8,
        adjusted_project_data[price].max() * 1.2
    )

    manual_min = rebased_project_data.price.min()
    manual_max = rebased_project_data.price.max()

    if num_of_bedroom == 2:

        manual_price_range = (
            manual_min,
            manual_max
        )

    else:
        manual_price_range = (
            manual_min / 0.9 * 0.85,
            manual_max / 1.1 * 1.2
        )

    linear_model, training_data = comparable_demand_model.fit_project_room_demand_model(
        project_id,
        num_of_bedroom,
        price_range=manual_price_range,
        exclude_ids=[project_id],
        # include_ids=['7a0eaa8196d9676a189aa4a7fbabc7e5']
    )

    linear_models[num_of_bedroom] = linear_model


    def normalize_date(date):
        return str(date.year) + ' ' + date.month_name()[:3]


    adjusted_project_data['date_to_display'] = adjusted_project_data.apply(
        lambda row: f"from {normalize_date(row['transaction_month'])} "
                    f"to {normalize_date(row['transaction_month_end'])}",
        axis=1
    )

    row = adjusted_project_data.iloc[[0]].copy()

    display_date = row['date_to_display'].iloc[0]

    curve = linear_model.extract_2d_demand_curve(
        row,
        launching_period=launching_period,
        price_range=price_range,
        fig_format='plt'
    )

    # notes: since we are extracting the first month's demand curve
    # we need to rebase the price back to launch time
    index_to_multiply = comparable_demand_model.query_time_rebase_index(adjusted_project_data)

    rebase_curve = PltDemandCurve(
        P=curve.P * index_to_multiply,
        Q=curve.Q
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = rebase_curve.plot(
        fig=fig,
        ax=ax,
        color=NatureD['blue'],
        label=f'Demand Curve \n{display_date}'
    )

    common_params = dict(
        xmin=price_range[0] * index_to_multiply,
        xmax=price_range[1] * index_to_multiply,
        alpha=0.7,
        linestyles='--',
    )
    ax.hlines(
        y=adjusted_project_data['num_of_units'].iloc[-1],
        **common_params,
        colors='grey',
        label='num_of_units'.replace('_', ' ')
    )

    scatter_params = dict(
        s=80,
        alpha=0.5,
        ax=ax
    )

    scatterplot_data = adjusted_project_data
    scatter_params['hue'] = scatterplot_data['date_to_display']

    if not scatterplot_data.empty:
        sns.scatterplot(
            x=scatterplot_data[price] * index_to_multiply,
            y=scatterplot_data[quantity],
            **scatter_params
        )

    plt.legend()

    title = f'{project_name} {int(num_of_bedroom)}-bedroom'
    ax.set_title(f'{title}')
    report_path = title.replace('-', '_').replace(' ', '_')
    plt.savefig(figure_dir + f"{report_path}.png", dpi=300)
    plt.close()

    training_data.to_csv(model_dir + f"training_{report_path}.csv", index=False)

pickle.dump(
    linear_models, open(models_path, 'wb')
)
