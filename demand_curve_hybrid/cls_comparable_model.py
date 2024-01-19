import warnings

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from constants.redshift import query_data
from demand_curve_hybrid.scr_coef import query_adjust_coef

from demand_model_utils.cls_linear_demand_model import BaseLinearDemandModel
from demand_curve_sep.scr_neighborhood_clusters import clustering_res

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ComparableDemandModel:

    data: pd.DataFrame
    forecasting_data: pd.DataFrame

    features = [
        'price',
        'launching_period',
        'num_of_remaining_units',
        'proj_num_of_units',
    ]
    max_year_gap: Optional[int] = 3
    rolling_windows: Optional[int] = 3

    quantity: Optional[str] = 'sales'
    price: Optional[str] = 'price'
    project_key: Optional[str] = 'dw_project_id'

    @property
    def available_projects(self):

        training_data = self.data.copy()
        forecasting_data = self.forecasting_data.copy()

        training_data['with_trans'] = True
        forecasting_data['with_trans'] = False

        retain_cols = ['project_name', 'num_of_bedrooms', 'with_trans']

        return pd.concat(
            [
                training_data[retain_cols].drop_duplicates(),
                forecasting_data[retain_cols].drop_duplicates()
            ],
            ignore_index=True
        )

    def query_nearby_projects(
        self,
        test_projects: Union[str, np.ndarray, list],
        max_distance
    ):

        if isinstance(test_projects, str):
            test_projects = [test_projects]

        if test_projects == ['b00873b9b7adb4c799c5cb7ff5ac4150']:

            base_new_launch_projects = f"""
            select
                'b00873b9b7adb4c799c5cb7ff5ac4150'::varchar as dw_project_id,
                1.3855::float as avg_lat,
                103.8336::float as avg_long
            """

        elif test_projects == ['86bcc94255d04aaab84c1a5a11f3f5b0']:

            base_new_launch_projects = f"""
            select
                '86bcc94255d04aaab84c1a5a11f3f5b0'::varchar as dw_project_id,
                1.3622899330455676::float as avg_lat,
                103.76381592854237::float as avg_long
            """
        else:
            base_new_launch_projects = f"""
            select distinct
                    dw_project_id,
                    avg(latitude) as avg_lat,
                    avg(longitude) as avg_long
                from data_science.ui_master_sg_project_geo_view_filled_features_condo a
                join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                    using (dw_project_id)
                where dw_project_id in ({','.join([i.__repr__() for i in test_projects])})
                group by 1
            """

        nearby_projects = query_data(
            f"""
            with base_new_launch_projects as (
                {base_new_launch_projects}
            ),
            base_historical_project as (
                select distinct
                    dw_project_id,
                    avg(latitude) as avg_lat,
                    avg(longitude) as avg_long
                from data_science.ui_master_sg_project_geo_view_filled_features_condo a
                join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                    using (dw_project_id)
                where dateadd(year, -{self.max_year_gap}, current_date) <= to_date(project_launch_month, 'YYYYMM')
                group by 1
            )
            select distinct
                new_p.dw_project_id as base_project_id,
                his_p.dw_project_id as nearby_project_id,
                ST_DistanceSphere(
                    st_point(his_p.avg_long, his_p.avg_lat),
                    st_point(new_p.avg_long, new_p.avg_lat)
                ) as distance
            from base_new_launch_projects new_p
            join base_historical_project his_p
                on ST_DistanceSphere(
                    st_point(his_p.avg_long, his_p.avg_lat),
                    st_point(new_p.avg_long, new_p.avg_lat)
                ) <= {max_distance}
            """
        )

        return nearby_projects

    def query_clusters_projects(
        self,
        neighborhood: str
    ):
        cluster = clustering_res[clustering_res['neighborhood'] == neighborhood]['cluster'].iloc[0]

        cluster_nei = clustering_res[clustering_res['cluster'] == cluster]['neighborhood']
        cluster_projects = self.data[self.data['neighborhood'].isin(cluster_nei)][['dw_project_id']].copy().rename(
            columns={'dw_project_id': 'nearby_project_id'}
        )

        return cluster_projects.drop_duplicates()

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

        project_size = training_data.proj_num_of_units.iloc[0]

        if not price_range:
            price_desc = project_data['price'].describe()
            min_price = price_desc['min']
            max_price = price_desc['max']

        else:
            min_price = price_range[0]
            max_price = price_range[1]

        if nearby_projects is None:
            nearby_projects = self.query_nearby_projects(
                project_id, max_distance=max_distance
            )

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
        include_ids=None,
        project_data=None,
        max_launching_period=None
    ):
        threshold = -3
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

            if project_data.empty:
                project_data = self.forecasting_data[
                    (self.forecasting_data[self.project_key] == project_id) &
                    (self.forecasting_data['num_of_bedrooms'] == num_of_bedroom)
                    ].copy()

        coef_to_multiply = query_adjust_coef(project_data)

        max_radius_projects = self.query_nearby_projects(
            project_id,
            max_distance=n_attempt * distance_gap
        )

        project_neighborhood = project_data.neighborhood.iloc[0]
        nearby_projects = self.query_clusters_projects(project_neighborhood)

        print()
        print(project_data['project_name'].iloc[0], f'{num_of_bedroom}-bedroom')

        min_model = None
        min_training_data = None

        for max_distance in np.arange(1, n_attempt + 1) * distance_gap:

            if max_distance != distance_gap:
                nearby_projects = max_radius_projects[max_radius_projects.distance <= max_distance]

                nearby_projects = pd.concat(
                    [
                        nearby_projects,
                        max_radius_projects[max_radius_projects.distance <= max_distance][['nearby_project_id']]
                    ],
                    ignore_index=True
                ).drop_duplicates()

                print(f'finding comparable projects within {max_distance / 1000 :.0f}km...')

            else:
                print(f'finding comparable projects in the same clusters...')

            if include_ids is not None:
                nearby_projects = pd.concat(
                    [
                        nearby_projects,
                        pd.DataFrame({'nearby_project_id': include_ids})
                    ],
                    ignore_index=True
                )

            training_data = self.filter_comparable_projects(
                project_id,
                num_of_bedroom,
                price_range=price_range,
                nearby_projects=nearby_projects,
                project_data=project_data,
                max_launching_period=max_launching_period
            ).copy()

            if exclude_ids is not None:
                training_data = training_data[~training_data[self.project_key].isin(exclude_ids)]
            if include_ids is not None:

                to_add_ids = [i for i in include_ids if i not in training_data.dw_project_id]

                include_data = self.data[
                    (self.data[self.project_key].isin(to_add_ids)) &
                    (self.data['num_of_bedrooms'] == num_of_bedroom)
                    ]
                training_data = pd.concat([training_data, include_data], ignore_index=True)

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
