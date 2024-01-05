from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.regression.linear_model

from demand_curve_sep.cls_plotly_demand_curve import PlotlyDemandCurve
from demand_curve_sep.cls_plt_demand_curve import PltDemandCurve


@dataclass
class BaseLinearDemandModel:
    quantity: str
    price: str
    features: list
    core_model: statsmodels.regression.linear_model = None
    params = None

    def process_data(self, data):

        data = data.copy()

        data[self.price] = np.log(data[self.price])
        data[self.quantity] = np.log(data[self.quantity])

        y = data[self.quantity]
        X = data[self.features]
        X = sm.add_constant(X, has_constant='add')

        return y, X

    def fit(self, training_subset):

        y, X = self.process_data(training_subset)

        Q = np.exp(y)
        w = Q**(Q/X['proj_num_of_units'])
        # w = Q/training_subset['proj_num_of_units']
        # w = (Q/X['proj_num_of_units'])
        # w = np.log(Q**2/training_subset['num_of_units'])

        try:
            self.core_model = sm.WLS(y, X, weights=w).fit()
        except np.linalg.LinAlgError:
            self.core_model = sm.GLS(y, X).fit()

        self.params = self.core_model.params

        return self

    def predict(self, data, last_period_discount=1):
        _, X = self.process_data(data)
        temp_y_hat = np.exp(self.core_model.predict(X)) * last_period_discount
        pred_Q = np.clip(
            temp_y_hat,
            0,
            X['num_of_remaining_units'].values
        )

        return pred_Q

    def extract_2d_demand_curve(
            self,
            project_data,
            launching_period,
            price_range: tuple = None,
            fig_format: Literal['plotly', 'plt'] = 'plotly'
    ):

        n_points = 50

        if price_range:
            P = np.linspace(*price_range)
        else:
            P = np.linspace(project_data[self.price].min() * 0.8, project_data[self.price].max() * 1.2)

        sample_data = pd.concat([project_data.iloc[[0]]] * n_points, ignore_index=True)
        sample_data[self.price] = P
        sample_data['launching_period'] = launching_period

        pred_Q = self.predict(sample_data).values

        if fig_format == 'plotly':
            return PlotlyDemandCurve(
                P=P,
                Q=pred_Q
            )
        else:
            return PltDemandCurve(
                P=P,
                Q=pred_Q
            )

    def extract_3d_demand_curve(
            self,
            adj_project_data
    ):

        demand_curves = []
        launching_periods = adj_project_data['launching_period']

        for idx, t in enumerate(launching_periods):
            demand_curves += [
                self.extract_2d_demand_curve(
                    project_data=adj_project_data.iloc[[idx]],
                    launching_period=t,
                    price_range=(
                        adj_project_data[self.price].min() * 0.8,
                        adj_project_data[self.price].max() * 1.2
                    )
                )
            ]

        fig, ax = DemandCurve3D(
            demand_curves=demand_curves,
            launching_periods=launching_periods
        ).plot()

        ax.set_yticks(
            np.arange(1, len(launching_periods) + 1),
            adj_project_data['transaction_month'].dt.date
        )

        return fig, ax
