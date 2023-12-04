import pickle
import warnings

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd

from demand_model.cls_linear_demand_model import RoomTypeDemandModel
from demand_model.cls_ds_partial_coef import FloorCoef, AreaCoef, TimeIndex, ZoneCoef
from constants.redshift import query_data
from constants.utils import OUTPUT_DIR
from demand_model.scr_neighborhood_clusters import clustering_res

warnings.filterwarnings("ignore", category=RuntimeWarning)

ALL_BEDS = ['one', 'two', 'three', 'four', 'five']
bed_nums = np.arange(1, 6)
floor_coef = FloorCoef()
area_coef = AreaCoef()
time_index = TimeIndex()
zone_coef = ZoneCoef()


@dataclass
class ProjectDemandModel:
    features = [
        'price',
        'launching_period',
        'num_of_remaining_units',
        'proj_num_of_units',
    ]
    max_year_gap = 3
    rolling_windows = 3
    quantity: str = 'sales'
    price: str = 'price'
    project_key: str = 'dw_project_id'

    model = []
    image_paths = []
    image_3d_paths = []

    @property
    def transactions_query(self):
        return f"""
            base_property_price as (
            select
                *,
                row_number() over (partition by dw_property_id order by transaction_date desc) as seq
            from data_science.ui_master_sg_transactions_view_filled_features_condo a
            join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                using (dw_project_id)
            where a.property_type_group = 'Condo'
                and transaction_sub_type = 'new sale'
                and transaction_date < '{datetime.today().replace(day=1).date()}'
        )
        """

    @property
    def developer_pricing_query(self):

        return f"""
            base_developer_price as (
                select
                    a.dw_project_id,
                    a.project_launch_month,
                    b.property_dwid as dw_property_id,
                    avg(b.developer_price/ b.area_sqft) as developer_psf
                from data_science.ui_master_sg_project_geo_view_filled_features_condo a
                join raw_reference.sg_new_launch_developer_price b
                    on a.dw_project_id = b.project_dwid
                where project_launch_month >= dateadd(months, -3, current_date)
                group by 1, 2, 3, project_launch_month
                order by project_launch_month desc
            ),
            base_property_price as (
                select
                    c.dw_project_id,
                    a.project_launch_month::varchar as transaction_month_index,
                    num_of_bedrooms,
                    floor_area_sqft,
                    floor_area_sqm,
                    address_floor_num,
                    developer_psf as unit_price_psf
                from base_developer_price a
                join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                    using (dw_property_id)
            )
        """

    def query_raw_data(self, mode: Literal['training', 'forecasting']):

        price_query = {
            'training': self.transactions_query,
            'forecasting': self.developer_pricing_query
        }.get(mode, lambda a: a)

        launch_date_filter = {
            'training': f"""
                where (
                    to_date(project_launch_month, 'YYYYMM') between
                    dateadd(year, -{self.max_year_gap}, current_date)
                    and
                    dateadd(month, -{self.rolling_windows}, current_date)
                )
                """,
            'forecasting': f"""
                where dateadd(month, -{self.max_year_gap}, current_date) <= to_date(project_launch_month, 'YYYYMM')
            """
        }.get(mode, lambda a: a)

        raw_data_path = f'{OUTPUT_DIR}{mode}_data.plk'
        if False:
            data = pickle.load(open(raw_data_path, 'rb'))
        else:
            data = query_data(
                f"""
                with base_index as ({time_index.query_scripts}),
                base_floor_coef as ({floor_coef.query_scripts}),
                base_area_coef as ({area_coef.query_scripts}), 
                {price_query},
                base_property_panel as (
                    select
                        b.dw_project_id,
                        to_date(transaction_month_index, 'YYYYMM') as transaction_month,
                        (
                            select hi_avg_improved
                            from data_science.sg_condo_resale_index_sale
                            order by transaction_month_index desc limit 1
                        ) as current_index,
                        avg(
                            unit_price_psf
                                * floor_adjust_coef
                                * area_adjust_coef
                                * zone_adjust_coef
                                * time_adjust_coef
                        ) as price,
                        avg(floor_area_sqm) as floor_area_sqm,
                        {0 if mode == 'forecasting' else 'count(*)'} as sales
                    from base_property_price a
                    left outer join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                        using (dw_project_id)
                    left outer join  base_index c
                        using (transaction_month_index)
                    left outer join  base_floor_coef f
                        using (address_floor_num)
                    left outer join base_area_coef g
                       on a.floor_area_sqft >= g.area_lower_bound and a.floor_area_sqft < g.area_upper_bound
                    left outer join (
                        {zone_coef.query_scripts}
                    ) d
                        on b.zone = d.zone and left(a.transaction_month_index, 4)::int = d.transaction_year
                    group by 1, 2
                )
                select
                    dw_project_id,
                    c.project_display_name as project_name,
                    to_date(project_launch_month, 'YYYYMM') as launch_year_month,
                    transaction_month,
                    datediff(
                        month,
                        launch_year_month,
                        transaction_month
                    ) + 1 as launching_period,
                    price,
                    sales,
                    case
                        when lag(sales, 1) over (
                            partition by dw_project_id
                            order by transaction_month
                            ) is null then proj_num_of_units
                        else proj_num_of_units - sum(sales) over (
                            partition by dw_project_id
                            order by transaction_month
                            rows between unbounded preceding and 1 preceding
                            )
                        end as num_of_remaining_units,
                    proj_num_of_units,
                    floor_area_sqm,
                    tenure,
                    proj_max_floor,
                    zone,
                    neighborhood
                from base_property_panel a
                join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                    using(dw_project_id)
                join (
                    select 
                    project_dwid as dw_project_id,
                    project_display_name 
                    from ui_app.project_summary_prod_sg
                 ) c
                    using(dw_project_id)
                {launch_date_filter}
                order by 1, 2, 3
                """
            )
            pickle.dump(
                data, open(raw_data_path, 'wb')
            )

        # categorical data processing
        if 'tenure' in self.features:
            data['tenure'] = data['tenure'].apply(lambda a: 1 if a == 'freehold' else 0)
        data['transaction_month'] = pd.to_datetime(data['transaction_month'])
        data = data[~data[self.price].isna()].copy()
        data.set_index(self.project_key, drop=False, inplace=True)

        return data

    def calculate_launching_period(self, project_data):
        project_data['launch_year_month'] = pd.to_datetime(project_data['launch_year_month'])
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
                    end=row['transaction_month_end'],
                    freq='M'
                )
            ), axis=1
        )

        return project_data

    def preprocess_base_training_data(self):

        data = self.query_raw_data(mode='training')

        def rolling_process(time_series_data):

            period_start_min = time_series_data['transaction_month'].min()
            period_start_max = time_series_data['transaction_month'].max()

            to_pd_timeseries = lambda s: s.to_timestamp().to_series().reset_index(drop=True)

            T_start = to_pd_timeseries(
                pd.period_range(
                    period_start_min, period_start_max, freq='M'
                ).rename('transaction_month')
            )

            if len(T_start) == 1:
                return time_series_data.reset_index(drop=True)
            else:
                wins = max(2, min(self.rolling_windows, len(T_start)))
                rolling_params = dict(window=wins, min_periods=wins)

            expended_data = pd.merge(time_series_data, T_start, how='right')

            foreward_cumsum = lambda s: s.fillna(0).rolling(**rolling_params).sum().shift(-(wins - 1))

            Q = foreward_cumsum(expended_data[self.quantity])
            PQ = foreward_cumsum(expended_data[self.quantity] * expended_data[self.price])

            P = PQ / Q

            expended_data[self.quantity] = Q
            expended_data[self.price] = P

            final_data = expended_data[expended_data[self.quantity] != 0] \
                .dropna(subset=self.quantity) \
                .fillna(method='bfill') \
                .fillna(method='ffill')
            final_data['num_of_remaining_units'] = final_data['proj_num_of_units'] - expended_data[self.quantity].shift(
                1).cumsum().fillna(0)

            return final_data.reset_index(drop=True)

        rolling_data = pd.DataFrame()
        for project in data.dw_project_id.unique():

            temp = data.loc[[project]].copy()

            if temp.empty:
                continue

            processed_temp = rolling_process(temp)

            rolling_data = pd.concat([rolling_data, processed_temp], axis='rows', ignore_index=True)

        rolling_data = self.calculate_launching_period(rolling_data)

        return rolling_data.set_index('dw_project_id', drop=False)

    def preprocess_forcasting_data(self, min_proj_size=50):

        data = self.query_raw_data(mode='forecasting')
        data = data[~data.dw_project_id.isin(self.data.dw_project_id)]
        data = data[data['proj_num_of_units'] > min_proj_size]
        data = self.calculate_launching_period(data)

        return data

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

    def get_rebased_project_data(self, dw_project_id):

        try:
            trans_status = self.available_projects.loc[dw_project_id]['with_trans']
        except KeyError:
            return pd.DataFrame()

        if trans_status:
            data_source = self.data
        else:
            data_source = self.forecasting_data

        project_data = data_source.loc[[dw_project_id]]

        return project_data

    def get_adjusted_project_data(self, dw_project_id):

        rebased_project_data = self.get_rebased_project_data(dw_project_id)

        if rebased_project_data.empty:
            return rebased_project_data

        coef_to_multiply = self.query_adjust_coef(rebased_project_data)

        adjusted_project_data = rebased_project_data.copy()
        adjusted_project_data[self.price] = adjusted_project_data[self.price] * coef_to_multiply

    def __post_init__(self):
        self.data = self.preprocess_base_training_data()
        self.forecasting_data = self.preprocess_forcasting_data()

    @property
    def available_projects(self):

        training_data = self.data.copy()
        forecasting_data = self.forecasting_data.copy()

        training_data['with_trans'] = True
        forecasting_data['with_trans'] = False

        retain_cols = ['dw_project_id', 'project_name', 'with_trans']

        ap = pd.concat(
            [
                training_data[retain_cols].drop_duplicates(),
                forecasting_data[retain_cols].drop_duplicates()
            ],
            ignore_index=True
        ).set_index('dw_project_id', drop=False)

        return ap

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
            price_range: tuple,
            project_data=None,
            max_launching_period=None,
            nearby_projects: pd.DataFrame = None,
            max_distance: int = None

    ):

        training_data = self.data

        if project_data is None:
            project_data = self.get_rebased_project_data(project_id)

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
            (training_data.price.between(min_price * 0.9, max_price * 1.1)) &
            (training_data.proj_num_of_units.between(project_size * 0.25, project_size * 1.75))
            ]

        if max_launching_period:
            comp_data = comp_data[comp_data['launching_period'] <= max_launching_period]

        outliers_idx_1 = comp_data[
            (comp_data[self.price] < comp_data[self.price].quantile(0.25)) &
            (comp_data[self.quantity] < comp_data[self.quantity].quantile(0.25))
            ].index

        outliers_idx_2 = comp_data[
            (comp_data[self.price] > comp_data[self.price].quantile(0.75)) &
            (comp_data[self.quantity] > comp_data[self.quantity].quantile(0.75))
            ].index

        filtered_comp_data = comp_data[
            ~comp_data.index.isin(np.append(outliers_idx_1, outliers_idx_2))
        ]

        return filtered_comp_data

    @staticmethod
    def query_adjust_coef(project_data):

        local_area_coef = area_coef.get_coef(project_data.floor_area_sqm.iloc[0] * 10.76)
        local_floor_coef = floor_coef.get_coef(project_data.proj_max_floor.iloc[0] // 2)
        local_zone_coef = zone_coef.get_coef(project_data.transaction_month.iloc[0].year, project_data.iloc[0].zone)
        coef_to_multiply = 1 / local_area_coef / local_floor_coef / local_zone_coef

        return coef_to_multiply

    def fit_project_room_demand_model(
            self,
            project_id,
            price_range=None,
            exclude_ids=None,
            project_data=None
    ):
        threshold = -3
        distance_gap = 3000
        n_attempt = 10

        def fit_local_linear_model(data):

            local_model = RoomTypeDemandModel(
                quantity=self.quantity,
                price=self.price,
                features=self.features
            ).fit(data)

            return local_model

        if project_data is None:
            project_data = self.get_rebased_project_data(project_id)

        coef_to_multiply = self.query_adjust_coef(project_data)

        max_radius_projects = self.query_nearby_projects(
            project_id,
            max_distance=n_attempt * distance_gap
        )

        project_neighborhood = project_data.neighborhood.iloc[0]
        nearby_projects = self.query_clusters_projects(project_neighborhood)

        for max_distance in np.arange(1, n_attempt + 1) * distance_gap:

            if max_distance != distance_gap:
                nearby_projects = max_radius_projects[max_radius_projects.distance <= max_distance]

            training_data = self.filter_comparable_projects(
                project_id,
                price_range=price_range,
                nearby_projects=nearby_projects,
                project_data=project_data
            ).copy()

            if exclude_ids:
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
                        price_range=(proj_avg_price / 0.9 * 0.8, proj_avg_price / 1.1 * 1.1),
                        nearby_projects=nearby_projects,
                        project_data=project_data
                    ).copy()

            adj_training_data = training_data.copy()
            adj_training_data[self.price] = training_data[self.price] * coef_to_multiply

            if exclude_ids:
                adj_training_data = adj_training_data[~adj_training_data[self.project_key].isin(exclude_ids)]

            linear_model = fit_local_linear_model(adj_training_data)

            if linear_model.params[self.price] > threshold:
                if max_distance < n_attempt * distance_gap:
                    continue

                elif len(adj_training_data) > 10:
                    for i in np.arange(1, n_attempt + 1):
                        random_sample = adj_training_data.sample(min(n_attempt, len(adj_training_data) - 1))
                        random_model = fit_local_linear_model(random_sample)
                        if random_model.params[self.price] > -threshold:
                            continue
                        else:
                            linear_model = random_model
                            adj_training_data = random_sample
                            break

            return linear_model, adj_training_data
