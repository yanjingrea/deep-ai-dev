from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal

import pandas as pd


@dataclass
class BaseCMData:
    aggregate_level: Literal['project', 'bedrooms']
    max_year_gap: Optional[int] = 3
    rolling_windows: Optional[int] = 3
    min_stock: Optional[int] = 75

    quantity: Optional[str] = 'sales'
    price: Optional[str] = 'price'
    project_key: Optional[str] = 'dw_project_id'

    features = [
        'price',
        'launching_period',
        'num_of_remaining_units',
        'proj_num_of_units',
    ]

    def __post_init__(self):
        self.data = self.preprocess_base_training_data()
        self.forecasting_data = self.preprocess_forecasting_data()

    def preprocess_base_training_data(self):
        ...

    def preprocess_forecasting_data(self):
        ...

    def calculate_launching_period(self, project_data):
        project_data['launch_year_month'] = pd.to_datetime(project_data['launch_year_month'], dayfirst=False)
        project_data['transaction_month_end'] = (
            project_data['transaction_month'].apply(
                lambda d:
                pd.period_range(
                    d,
                    periods=self.rolling_windows,
                    freq='M'
                ).to_timestamp()[-1]
            )
        )

        project_data['launching_period'] = project_data.apply(
            lambda row: len(
                pd.period_range(
                    start=row['launch_year_month'],
                    end=row['transaction_month'],
                    freq='M'
                )
            ), axis=1
        )

        return project_data

    def prepare_forecast_demand_curve_data(
        self,
        project_data
    ):
        period_data = project_data.iloc[[-1]].copy()

        remaining = project_data['num_of_remaining_units'].iloc[-1] - project_data[self.quantity].iloc[-1]

        if remaining <= 0:
            return pd.DataFrame()

        period_data['num_of_remaining_units'] = remaining
        period_data['transaction_month'] = pd.to_datetime(datetime.today().replace(day=1).date())
        period_data = self.calculate_launching_period(period_data)

        return period_data
