import warnings

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from demand_model_utils.scr_coef import query_adjust_coef
from demand_model_utils.cls_linear_demand_model import BaseLinearDemandModel

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ComparableDemandModel:

    data: pd.DataFrame

    features: Optional[np.ndarray] = np.array([
        'price',
        'launching_period',
        'num_of_remaining_units',
        'proj_num_of_units'
    ])
    max_year_gap: Optional[int] = 3
    rolling_windows: Optional[int] = 3

    quantity: Optional[str] = 'sales'
    price: Optional[str] = 'price'
    project_key: Optional[str] = 'dw_project_id'

    # todo: tackle magic number
    def filter_comparable_projects(
        self,
        project_id,
        num_of_bedrooms,
        price_range: tuple,
        project_data=None,
        max_launching_period=None,
        nearby_projects: pd.DataFrame = None,
        max_distance: int = None

    ):

        training_data = self.data

        if project_data is None:
            project_data = training_data[
                (training_data['dw_project_id'] == project_id) &
                (training_data['num_of_bedrooms'] == num_of_bedrooms)
                ]

        project_size = project_data.proj_num_of_units.iloc[0]

        if not price_range:
            price_desc = project_data['price'].describe()
            min_price = price_desc['min']
            max_price = price_desc['max']

        else:
            min_price = price_range[0]
            max_price = price_range[1]

        comp_data = training_data[
            (training_data.dw_project_id.isin(nearby_projects.nearby_project_id)) &
            (training_data.num_of_bedrooms == num_of_bedrooms) &
            (training_data.price.between(min_price * 0.9, max_price * 1.1)) &
            (training_data.proj_num_of_units.between(project_size * 0.25, project_size * 1.75))
            ]

        if max_launching_period:
            comp_data = comp_data[comp_data['launching_period'] <= max_launching_period]

        if True:

            sales_percent = comp_data[self.quantity] / comp_data['num_of_units']

            outliers_idx_1 = comp_data[
                (comp_data[self.price] < comp_data[self.price].quantile(0.25)) &
                (sales_percent < sales_percent.quantile(0.25))
                ].index

            outliers_idx_2 = comp_data[
                (comp_data[self.price] > comp_data[self.price].quantile(0.75)) &
                (sales_percent > sales_percent.quantile(0.75))
                ].index

            filtered_comp_data = comp_data[
                ~comp_data.index.isin(np.append(outliers_idx_1, outliers_idx_2))
            ]

        else:
            filtered_comp_data = comp_data.copy()

        return filtered_comp_data

    def fit_project_room_demand_model(
        self,
        project_id,
        num_of_bedroom,
        price_range=None,
        exclude_ids=None,
        include_ids: list = None,
        project_data=None,
        max_launching_period=None,
        threshold=-3
    ):
        # threshold = -3
        distance_gap = 3000
        n_attempt = 10

        def fit_local_linear_model(data):

            local_model = BaseLinearDemandModel(
                quantity=self.quantity,
                price=self.price,
                features=self.features
            ).fit(data)

            return local_model

        if project_data is None:
            project_data = self.data[
                (self.data[self.project_key] == project_id) &
                (self.data['num_of_bedrooms'] == num_of_bedroom)
                ].copy()

        coef_to_multiply = query_adjust_coef(project_data)

        max_radius_projects = self.data.rename(columns={'dw_project_id': 'nearby_project_id'})

        print()
        print(project_data['project_name'].iloc[0], f'{num_of_bedroom}-bedroom')

        min_model = None
        min_training_data = None

        for max_distance in np.arange(1, n_attempt + 1) * distance_gap:

            nearby_projects = max_radius_projects[max_radius_projects.distance <= max_distance]

            if include_ids is not None:
                include_projects = max_radius_projects[max_radius_projects.nearby_project_id.isin(include_ids)]
                nearby_projects = pd.concat([nearby_projects, include_projects]).drop_duplicates()

            print(f'finding comparable projects within {max_distance / 1000 :.0f}km...')

            training_data = self.filter_comparable_projects(
                project_id,
                num_of_bedroom,
                price_range=price_range,
                nearby_projects=nearby_projects,
                project_data=project_data,
                max_launching_period=max_launching_period
            ).copy()

            training_data = training_data[training_data['num_of_remaining_units'] > 0].copy()

            if exclude_ids is not None:
                training_data = training_data[~training_data[self.project_key].isin(exclude_ids)]

            if (len(training_data) < 10) or (training_data.nunique()['dw_project_id'] < 3):
                if max_distance != n_attempt * distance_gap:
                    continue
                elif project_data.price.mean() > self.data.price.quantile(0.75):
                    nearby_projects = pd.DataFrame(
                        {'nearby_project_id': self.data[self.project_key].unique()}
                    )

                    proj_avg_price = project_data.price.mean()

                    training_data = self.filter_comparable_projects(
                        project_id,
                        num_of_bedroom,
                        price_range=(proj_avg_price / 0.9 * 0.8, proj_avg_price / 1.1 * 1.2),
                        nearby_projects=nearby_projects,
                        project_data=project_data
                    ).copy()

            adj_training_data = training_data.copy()
            adj_training_data[self.price] = training_data[self.price] * coef_to_multiply

            if exclude_ids:
                adj_training_data = adj_training_data[~adj_training_data[self.project_key].isin(exclude_ids)]

            linear_model = fit_local_linear_model(adj_training_data)

            if min_model is None:
                min_model = linear_model
                min_training_data = adj_training_data
            elif linear_model.params[self.price] < min_model.params[self.price]:
                min_model = linear_model
                min_training_data = adj_training_data

            if linear_model.params[self.price] > threshold:
                if max_distance < n_attempt * distance_gap:
                    continue

                elif len(adj_training_data) > 10:

                    print('finding comparable project randomly...')

                    for i in np.arange(1, (n_attempt + 1) * 2):
                        random_sample = adj_training_data.sample(min(n_attempt, len(adj_training_data) - 1))
                        random_model = fit_local_linear_model(random_sample)
                        if random_model.params[self.price] > -threshold:
                            continue
                        else:

                            print('fail to get qualified curve by randomly filtering')

                            linear_model = min_model
                            adj_training_data = min_training_data
                            break

            return linear_model, adj_training_data
