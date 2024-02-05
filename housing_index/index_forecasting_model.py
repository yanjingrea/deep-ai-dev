from os.path import dirname, realpath

import numpy as np
import pandas as pd
import shap
import sklearn.linear_model
import statsmodels.api

from statsmodels.api import add_constant, RLM, OLS, WLS

from matplotlib import pyplot as plt
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from constants.utils import COLOR_SCALE, set_plot_format
from housing_index.features import data

figure_dir = dirname(realpath(__file__)) + f'/output/'

set_plot_format(plt)

time_index = 'quarter_index'
nth_step = 1

start = 201000
end = 202303

index_cols = [
    'residential_properties_index',
    'landed_index',
    'non_landed_index',
    'hdb_resale_index'
]

X_cols = [
    # 'year',
    # 'quarter_int',
    # 'gdp',
    # 'housing_gdp',
    'cpi',
    # 'unemployment_rate',
    # 'monthly_household_income',
    # 'gini_coefficient',
    'population',
    # 'nature_increase_rate',
    # 'population_sc',
    'available_condo',
    'vacant_condo',
    'developing_condo',
    'constructed_hdb',
    'sold_hdb',
    # 'covid',
    # 'crisis',
    # 'sc_percentage',
    'avg_price_condo_new_sale',
    # 'avg_price_condo_new_sale_gr',
    'trans_num_condo_new_sale',
    'avg_price_condo_resale',
    # 'avg_price_condo_resale_gr',
    'trans_num_condo_resale',
    'avg_price_hdb_resale',
    # 'avg_price_hdb_resale_gr',
    'trans_num_hdb_resale',
    'ar.L1'
]

data.to_csv(figure_dir + f'training_data.csv', index=False)


def quarter_index_to_pd_quarter(col):
    def func(cell):
        str_cell = str(cell)

        year = int(str_cell[:4])
        quarter = int(str_cell[-1])

        return pd.Period(year=year, quarter=quarter, freq='Q')

    new_col = col.apply(lambda a: func(a)).dt.to_timestamp()

    return new_col


def split_data(start_index, end_index):

    train = data[(data[time_index] >= start_index) & (data[time_index] < end_index)]
    test = data[data[time_index] >= end_index]

    return train, test


def fit_auto_arima(endog, exog):

    import pmdarima as pm

    # auto = pm.auto_arima(
    #     endog, exog,
    #     start_p=1, start_q=1,
    #     test='adf',
    #     max_p=3, max_q=3, m=1,
    #     start_P=0, seasonal=False,
    #     d=None, D=1, trace=True,
    #     error_action='ignore',
    #     suppress_warnings=True,
    #     stepwise=True
    # )

    # model = ARIMA(
    #     endog, exog,
    #     order=auto.order, seasonal_order=auto.seasonal_order
    # ).fit()

    # model = ARIMA(
    #     endog, exog,
    #     order=(2, 0, 0),
    #     seasonal_order=(0, 0, 0, 0),
    #     freq='QS-OCT',
    #     # trend='t'
    # ).fit()

    model = OLS(endog, exog).fit()

    # start_params = pd.Series(
    #     model.start_params,
    #     index=['const'] + X_cols + ['ar.L1', 'ma.L1', 'sigma2']
    # )
    # start_params.loc['avg_price_condo_new_sale'] = 1
    # start_params.loc['avg_price_condo_resale'] = 1
    # start_params.loc['avg_price_hdb_resale'] = 1
    # start_params.loc['developing_condo'] = -10

    # model = model.fit(start_params=start_params)
    # .fit_constrained({'ar.L1': 0.001})

    return model


data = data.set_index(quarter_index_to_pd_quarter(data[time_index]))
data = data.fillna(method='ffill')

test_index = 202001

params_results = pd.DataFrame()

for target_var in index_cols:

    y_col = target_var + f"_lead{nth_step}"
    data[y_col] = data[target_var].shift(-nth_step)
    data['ar.L1'] = data[y_col].shift(1)
    data['quarter_int'] = data['quarter'].str[-2].astype(int)
    data['avg_price_condo_new_sale_gr'] = data['avg_price_condo_new_sale'] / data['avg_price_condo_new_sale'].shift(
        1
    ) - 1
    data['avg_price_condo_resale_gr'] = data['avg_price_condo_resale'] / data['avg_price_condo_new_sale'].shift(1) - 1
    data['avg_price_hdb_resale_gr'] = data['avg_price_hdb_resale'] / data['avg_price_condo_new_sale'].shift(1) - 1

    results = data[data[time_index] >= test_index][['quarter_date', time_index, y_col]].copy()

    y_results = []

    combined_waterfall = go.Figure()
    traces = []

    for i, q in enumerate(results[time_index]):

        print(f'forecast on {q}... \n')

        train, test = split_data(start, q)

        y = train[y_col]
        X = train[X_cols]

        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X_cols, index=train.index)
        X = statsmodels.api.add_constant(X, has_constant='add')

        # model = fit_auto_arima(y, X)
        model = OLS(y, X).fit()
        # model = WLS(y, X, weights=np.log(np.arange(2, len(y)+2))).fit()


        row = test[X_cols].iloc[[0]]

        X_test = pd.DataFrame(scaler.transform(row), columns=X_cols, index=row.index)
        X_test = statsmodels.api.add_constant(X_test, has_constant='add')

        # y_pred = model.forecast(
        #     step=nth_step,
        #     exog=pred_X
        # ).values[0]

        y_pred = model.predict(X_test).values[0]

        y_results += [y_pred]

        if i == len(results) - 1:

            results['pred_index'] = y_results

        if i >= len(results) - 12:

            if str(q)[-1] != '1':
                continue

            i = 0

            params = pd.DataFrame(model.params.rename('scaled_coefficients'))
            params.insert(0, 'target', target_var)

            params['scaler_scale'] = np.append(1, scaler.scale_)
            params['scaler_min'] = np.append(0, scaler.min_)

            params['unscaled_coefficients'] = params['scaled_coefficients'] * params['scaler_scale']
            params['p_values'] = model.pvalues

            params['feature_value'] = np.append(1, row.iloc[0].astype(int))

            force = (
                            params['feature_value'] * params['unscaled_coefficients'] +
                            params['scaler_min'] * params['scaled_coefficients']
            )

            def display(a):

                if a < 10**3:
                    return str(int(a))
                elif a < 10**6:
                    return str(round(a / 10**3, 1)) + 'k'
                elif a < 10**9:
                    return str(round(a / 10**6, 1)) + 'm'
                elif a < 10**12:
                    return str(round(a / 10**9, 1)) + 'b'
                else:
                    return str(a)

            trace = go.Waterfall(
                name=q,
                orientation="v",
                measure=["relative"] * len(params) + ["total"],
                x=[i.replace('_', ' ') for i in params.index] + ["pred_index"],
                textposition="outside",
                text=[display(i) for i in np.append(params['feature_value'], y_pred)],
                y=np.append(force, y_pred),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )

            combined_waterfall.add_trace(trace=trace)

            if False:

                fig = go.Figure(trace)

                fig.update_layout(
                    title=f"{target_var.replace('_', ' ')} forecast {q}",
                    showlegend=True
                )

                fig.show()

            if False:

                from sklearn.linear_model import LinearRegression

                regressor = LinearRegression()
                regressor.fit(X, y)
                explainer = shap.LinearExplainer(regressor, X)

                shap_values = explainer(X_test)

                force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values.values,
                    params['feature_value'],
                    feature_names=[i.replace('_', ' ') for i in X_test.columns]
                )

                force_plot.matplotlib(
                    figsize=(12, 12),
                    show=True,
                    text_rotation='vertical'
                )

                shap.plots.force(
                    explainer.expected_value,
                    shap_values.values[0],
                    params['feature_value'],
                    feature_names=[i.replace('_', ' ').capitalize() for i in X_test.columns],
                    matplotlib=True,
                    show=False,
                    text_rotation='vertical'
                )
                plt.title('SHAP Force Plot', fontsize=16)
                plt.show()

    combined_waterfall.update_layout(
        title=f"{target_var.replace('_', ' ')} forecast",
        showlegend=True,
        waterfallgroupgap=0.2
    )

    combined_waterfall.show()

    fig, ax = plt.subplots(figsize=(12, 4))

    plot_start_index = 202001
    plot_data = data[data[time_index] >= plot_start_index]


    def plot_scatter(
        x,
        y,
        ax,
        color,
        label,
        annotate: bool
    ):
        ax.plot(x, y, color=color, lw=2.4, zorder=10, label=label)
        ax.scatter(x, y, fc="w", ec=color, s=60, lw=2.4, zorder=12)

        if annotate:

            for idx, value in enumerate(y):
                ax.annotate(
                    value,
                    xy=(x[idx], value + 2),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    zorder=12
                )

        return ax


    ax = plot_scatter(
        x=plot_data['quarter_date'],
        y=plot_data[target_var],
        ax=ax,
        label='actual_gov_index',
        color=COLOR_SCALE[0],
        annotate=True
    )

    current_index = '2024-01-01'
    ax = plot_scatter(
        x=results['quarter_date'].shift(-1).fillna(current_index).values,
        y=results['pred_index'],
        ax=ax,
        label='pred_index',
        color=COLOR_SCALE[1],
        annotate=False
    )

    final_prediction = results['pred_index'][-1]

    ax.annotate(
        f"forecast: {final_prediction:.2f}",
        xy=(current_index, final_prediction - 10),
        ha="left",
        va="bottom",
        fontsize=10,
        zorder=12
    )

    error = results['pred_index'] / results[y_col] - 1

    title = f"{target_var.replace('_', ' ')} forecast"
    ax.set_title(title + f'\nback test error: {error.abs().mean() * 100 :.2f}%')

    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(figure_dir + title.replace(' ', '_') + '.png', dpi=300)

    combined_waterfall.write_html(
        figure_dir + 'waterfall_' + title.replace(' ', '_')+'.html'
    )

# params_results.to_csv(figure_dir + 'coefficients.csv', index=True)
print()