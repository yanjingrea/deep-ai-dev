import pickle
from datetime import datetime

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
training_data['is_first_period'] = training_data['launching_period'].apply(lambda a: 1 if a <= 3 else 0)
training_data['is_minor_first_period'] = training_data['minor_launching_period'].apply(lambda a: 1 if a <= 1 else 0)
training_data['good_market'] = training_data['transaction_month'].apply(
    lambda a: 1 if a in pd.date_range(start='2020-07-01', end='2020-12-01', freq='MS') else 0
)

# training_data['sales_lag1'] = training_data.groupby(['dw_project_id', 'num_of_bedrooms'])['sales'].shift(1)
# training_data = training_data.dropna()

comparable_demand_model = ComparableDemandModel(
    data=training_data,
    features=np.array(
        [
            'price',
            'is_first_period',
            'is_minor_first_period',
            'launching_period',
            'num_of_remaining_units',
            'proj_num_of_units',
            'num_of_units_launched',
            'minor_launching_period',
            'good_market',
            'transaction_month_idx'
            # 'time_adjust_coef'
            # 'sales_lag1'
        ]
    )
)

price = comparable_demand_model.price
quantity = comparable_demand_model.quantity

project_name = 'Jadescape'
today = datetime.today().date()
models_path = model_dir + f'{project_name} {today}'.replace(' ', '_')

linear_models = {}
for num_of_bedroom in np.arange(1, 6):

    if num_of_bedroom in [1]:
        min_quantity = 1
    elif num_of_bedroom in [2, 3, 4]:
        min_quantity = 2
    else:
        min_quantity = 1

    bed_training_data = training_data[training_data['sales'] >= min_quantity].copy()

    # bed_training_data = bed_training_data[
    #     bed_training_data['transaction_month'] >= pd.to_datetime('2019-11-01')
    #     ].copy()

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
        include_ids=[
            'fdfdbf4f5dfc8b1e55008dd25d349183',
            'b1d98ea4fd98d6699388a3d0cce389f2'
        ],
        threshold=-3
    )

    adjusted_training_data.to_csv(
        model_dir + f'{project_name} {int(num_of_bedroom)}-bedroom.csv'
    )

    linear_models[num_of_bedroom] = linear_model


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
    good_markets = pd.date_range(start='2020-04-01', end='2020-12-01', freq='MS')

    for idx, row in rebased_projects_data.iterrows():

        temp_period = row['launching_period']
        temp_display_date = row['date_to_display']
        temp_adj_coef = 1 / row['time_adjust_coef']

        if num_of_bedroom == 1:

            if temp_period == 1:
                adj = 1.06

            elif row['transaction_month'] in good_markets:

                if temp_period == 22:
                    adj = 0.9
                elif temp_period == 25:
                    adj = 0.83
                else:
                    adj = 0.8

            else:
                adj = 1

        if num_of_bedroom == 4:

            if temp_period == 1:
                adj = 0.93

            elif row['transaction_month'] in good_markets:

                if temp_period == 22:
                    adj = 0.9
                elif temp_period == 25:
                    adj = 0.9
                else:
                    adj = 0.85
            elif temp_period == 31:
                adj = 0.85

            else:
                adj = 1

        if num_of_bedroom == 5 and temp_period == 22:
            adj = 0.8

        if temp_period not in plot_periods:
            continue

        temp_curve = linear_model.extract_2d_demand_curve(
            rebased_projects_data.iloc[[idx]],
            launching_period=temp_period,
            price_range=price_range,
            fig_format='plt'
        )

        adjusted_curve = PltDemandCurve(
            P=temp_curve.P * temp_adj_coef / adj,
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

pickle.dump(linear_models, open(models_path, 'wb'))

data_class_path = model_dir + f'dataclass {project_name} {today}'.replace(' ', '_')
pickle.dump(training_data_class, open(data_class_path, 'wb'))
