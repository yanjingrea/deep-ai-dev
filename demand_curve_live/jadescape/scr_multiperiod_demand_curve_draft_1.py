from matplotlib import pyplot as plt
import seaborn as sns

from constants.utils import NatureD, NatureL
from demand_model_utils.cls_plt_demand_curve import PltDemandCurve
from demand_curve_live.jadescape.cls_data import BaseCMData
from demand_curve_live.jadescape.cls_model import *
from demand_curve_live.jadescape.scr_get_paths import model_dir, figure_dir

training_data_class = BaseCMData()
training_data = training_data_class.data.sort_values(
    by=['dw_project_id', 'num_of_bedrooms', 'transaction_month']
)
training_data['is_first_period'] = training_data['launching_period'].apply(lambda a: 1 if a <= 1 else 0)

comparable_demand_model = ComparableDemandModel(
    data=training_data,
    features=np.array(
        [
            'price',
            'is_first_period',
            'launching_period',
            'num_of_remaining_units',
            'proj_num_of_units'
        ]
    )
)

price = comparable_demand_model.price
quantity = comparable_demand_model.quantity

project_name = 'Jadescape'

for num_of_bedroom in np.arange(1, 6):

    if num_of_bedroom != 5:
        min_quantity = 1
    else:
        min_quantity = 5

    bed_training_data = training_data[training_data['sales'] >= min_quantity].copy()

    comparable_demand_model.__setattr__('data', bed_training_data)

    rebased_projects_data = training_data[
        (training_data['project_name'] == project_name) &
        (training_data['num_of_bedrooms'] == num_of_bedroom)
        ].reset_index()

    coef_to_multiply = query_adjust_coef(rebased_projects_data)
    rebased_projects_data[price] = rebased_projects_data[price] * coef_to_multiply

    linear_model, adjusted_training_data = comparable_demand_model.fit_project_room_demand_model(
        project_id=rebased_projects_data.dw_project_id.iloc[0],
        num_of_bedroom=num_of_bedroom,
        exclude_ids=[rebased_projects_data.dw_project_id.iloc[0]] if num_of_bedroom == 5 else None,
        include_ids=[
            'fdfdbf4f5dfc8b1e55008dd25d349183',
            'b1d98ea4fd98d6699388a3d0cce389f2'
        ] if num_of_bedroom == 5 else None,
        threshold=-3 if num_of_bedroom >= 4 else -4,
        max_launching_period={
            1: None,
            2: None,
            3: 12,
            4: 12,
            5: 24
        }.get(num_of_bedroom, lambda a: a)
    )

    adjusted_training_data.to_csv(
        model_dir + f'{project_name} {int(num_of_bedroom)}-bedroom.csv'
    )

    last_date = rebased_projects_data['transaction_month_end'].iloc[-1]


    def normalize_date(date):
        return str(date.year) + ' ' + date.month_name()[:3]


    rebased_projects_data['date_to_display'] = rebased_projects_data.apply(
        lambda row: f"from {normalize_date(row['transaction_month'])} "
                    f"to {normalize_date(row['transaction_month_end'])}",
        axis=1
    )

    price_range = (
        rebased_projects_data[price].min() * 0.8,
        rebased_projects_data[price].max() * 1.2
    )

    Q = rebased_projects_data[quantity]
    pred_Q = linear_model.predict(rebased_projects_data)

    from sklearn.metrics import mean_absolute_percentage_error

    score = mean_absolute_percentage_error(Q, pred_Q)

    print(f'mean absolute percentage error: {score * 100:.2f}%')

    plot_periods = np.arange(1, 51, 3)

    for idx, row in rebased_projects_data.iterrows():

        temp_period = row['launching_period']
        temp_display_date = row['date_to_display']
        temp_adj_coef = 1 / row['time_adjust_coef']

        if temp_period not in plot_periods:
            continue

        temp_curve = linear_model.extract_2d_demand_curve(
            rebased_projects_data.iloc[[idx]],
            launching_period=temp_period,
            price_range=price_range,
            fig_format='plt'
        )

        adjusted_curve = PltDemandCurve(
            P=temp_curve.P * temp_adj_coef,
            Q=temp_curve.Q
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        fig, ax = adjusted_curve.plot(
            fig=fig,
            ax=ax,
            color=NatureD['blue'] if idx == 0 else NatureL['blue'],
            label=f'Demand Curve \n{temp_display_date}'
        )

        common_params = dict(
            xmin=price_range[0] * temp_adj_coef,
            xmax=price_range[1] * temp_adj_coef,
            alpha=0.7,
            linestyles='--',
        )

        if idx == 0:

            ax.hlines(
                y=rebased_projects_data['num_of_units'].iloc[-1],
                **common_params,
                colors='grey',
                label='num_of_units'.replace('_', ' ')
            )
        else:
            ax.hlines(
                y=rebased_projects_data['num_of_remaining_units'].iloc[idx],
                **common_params,
                colors='lightgrey',
                label='num_of_remaining_units'.replace('_', ' ')
            )

        scatter_params = dict(
            s=80,
            alpha=0.5,
            ax=ax
        )

        scatterplot_data = rebased_projects_data.iloc[[idx]]
        # scatter_params['hue'] = scatterplot_data['date_to_display']

        if not scatterplot_data.empty:
            sns.scatterplot(
                x=scatterplot_data[price] * temp_adj_coef,
                y=scatterplot_data[quantity],
                **scatter_params
            )

        title = f'{project_name} {num_of_bedroom}-bedroom period {temp_period}'
        ax.set_title(f'{title}')
        report_path = title.replace('-', '_').replace(' ', '_')
        plt.savefig(figure_dir + f"{report_path}.png", dpi=300)
        plt.close()


print()
