import pickle

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd

from constants.redshift import query_data
from constants.utils import OUTPUT_DIR
from demand_curve_hybrid.scr_coef import *

ALL_BEDS = ['one', 'two', 'three', 'four', 'five']
bed_nums = np.arange(1, 6)


@dataclass
class BaseCMData:

    aggregate_level: Literal['project', 'bedrooms']

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

    def __post_init__(self):
        self.data = self.preprocess_base_training_data()
        self.forecasting_data = self.preprocess_forecasting_data()

    @property
    def transactions_query(self):
        return f"""
                base_property_price as (
                    select * from (
                        select
                            *,
                            row_number() over (partition by dw_property_id order by transaction_date desc) as seq
                        from data_science.ui_master_sg_transactions_view_filled_features_condo a
                        join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                            using (dw_project_id)
                        where a.property_type_group = 'Condo'
                            and transaction_sub_type = 'new sale'
                            and transaction_date < '{datetime.today().replace(day=1).date()}'
                    ) where seq = 1
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
                        c.dw_building_id,
                        a.project_launch_month::varchar as transaction_month_index,
                        num_of_bedrooms,
                        floor_area_sqft,
                        floor_area_sqft * 10.76 as floor_area_sqm,
                        address_floor_num,
                        developer_psf as unit_price_psf
                    from base_developer_price a
                    join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                        using (dw_property_id)
                )
            """

    def query_raw_data(
        self,
        mode: Literal['training', 'forecasting']
    ):

        price_query = {
            'training': self.transactions_query,
            'forecasting': self.developer_pricing_query
        }.get(mode, lambda a: a)

        launch_date_filter = {
            'training': f"""
                    where to_date(project_launch_month, 'YYYYMM') between
                    dateadd(year, -{self.max_year_gap}, current_date)
                    and
                    dateadd(month, -{self.rolling_windows}, current_date)
                    """,
            'forecasting': f"""
                    where dateadd(year, -{self.max_year_gap}, current_date) <= to_date(project_launch_month, 'YYYYMM')
                    """
        }.get(mode, lambda a: a)

        num_of_units_query = {
            'bedrooms': """
                                case
                                when num_of_bedrooms = 0 then project_units_zero_rm
                                when num_of_bedrooms = 1 then project_units_one_rm
                                when num_of_bedrooms = 2 then project_units_two_rm
                                when num_of_bedrooms = 3 then project_units_three_rm
                                when num_of_bedrooms = 4 then project_units_four_rm
                                when num_of_bedrooms = 5 then project_units_five_rm
                            end
                            """,
            'project': 'proj_num_of_units'
        }.get(self.aggregate_level, lambda a: a)

        floor_area_sqm_query = {
            'bedrooms': """
                        case
                            when num_of_bedrooms = 0 then project_avg_size_of_zero_rm
                            when num_of_bedrooms = 1 then project_avg_size_of_one_rm
                            when num_of_bedrooms = 2 then project_avg_size_of_two_rm
                            when num_of_bedrooms = 3 then project_avg_size_of_three_rm
                            when num_of_bedrooms = 4 then project_avg_size_of_four_rm
                            when num_of_bedrooms = 5 then project_avg_size_of_five_rm
                        end
                        """,
            'project': 'floor_area_sqm'
        }.get(self.aggregate_level, lambda a: a)

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
                            {"a.num_of_bedrooms" if self.aggregate_level == 'bedrooms' else '--'},
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
                                    --* zone_adjust_coef
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
                                        on a.floor_area_sqft >= g.area_lower_bound and
                                           a.floor_area_sqft < g.area_upper_bound
                        left outer join (
                            {zone_coef.query_scripts}
                        ) d
                            on b.zone = d.zone and left(a.transaction_month_index, 4)::int = d.transaction_year
                        group by 1, 2 {",3" if self.aggregate_level == 'bedrooms' else '--'}
                    )
                    select
                        dw_project_id,
                        c.project_display_name as project_name,
                        {"num_of_bedrooms" if self.aggregate_level == 'bedrooms' else '-1'} as num_of_bedrooms,
                        to_date(project_launch_month, 'YYYYMM') as launch_year_month,
                        transaction_month,
                        datediff(
                            month,
                            launch_year_month,
                            transaction_month
                        ) + 1 as launching_period,
                        price,
                        {num_of_units_query} as num_of_units,
                        sales,
                        case
                            when lag(sales, 1) over (
                                    partition by dw_project_id, num_of_bedrooms
                                    order by  transaction_month
                                ) is null then num_of_units
                            else num_of_units - sum(sales) over (
                                partition by dw_project_id, num_of_bedrooms
                                order by transaction_month
                                rows between unbounded preceding and 1 preceding
                            )
                        end as num_of_remaining_units,
                        proj_num_of_units,
                        tenure,
                        {floor_area_sqm_query} as floor_area_sqm,
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
                    order by 1, 2, 3, 4
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
                    end=row['transaction_month'],
                    freq='M'
                )
            ), axis=1
        )

        return project_data

    def preprocess_base_training_data(self):

        data = self.query_raw_data(mode='training')

        def rolling_process(time_series_data):

            to_pd_timeseries = lambda s: s.to_timestamp().to_series().reset_index(drop=True)

            period_start_min = time_series_data['transaction_month'].min()
            period_start_max = time_series_data['transaction_month'].max()

            t_start = to_pd_timeseries(
                pd.period_range(
                    period_start_min, period_start_max, freq='M'
                ).rename('transaction_month')
            )

            if len(t_start) == 1:
                return time_series_data.reset_index(drop=True)
            else:
                wins = max(2, min(self.rolling_windows, len(t_start)))
                rolling_params = dict(window=wins, min_periods=wins)

            expended_data = pd.merge(time_series_data, t_start, how='right')

            foreward_cumsum = lambda s: s.fillna(0).rolling(**rolling_params).sum()

            filled_monthly_quantity = expended_data[self.quantity].fillna(0)

            Q = foreward_cumsum(filled_monthly_quantity)
            PQ = foreward_cumsum(filled_monthly_quantity * expended_data[self.price].fillna(0))
            P = PQ / Q

            cumsum_quantity = expended_data[self.quantity].fillna(0).cumsum()
            S = time_series_data['num_of_units'].iloc[0] - cumsum_quantity.shift(1).fillna(0)

            re_expended_data = expended_data.copy()

            re_expended_data[self.quantity] = Q
            re_expended_data[self.price] = P
            re_expended_data['num_of_remaining_units'] = S.shift(wins - 1)
            re_expended_data['transaction_month'] = t_start.shift(wins - 1).dropna()

            final_data = re_expended_data.iloc[wins - 1:].dropna(subset=self.quantity)
            final_data = final_data[final_data[self.quantity] != 0].fillna(method='bfill').fillna(method='ffill')

            if np.any(final_data['num_of_remaining_units'] < 0):
                raise Exception(f'please check the method of rolling method')

            return final_data.reset_index(drop=True)

        rolling_data = pd.DataFrame()
        for project in data.dw_project_id.unique():

            iter_list = {
                'project': [-1],
                'bedrooms': bed_nums
            }.get(self.aggregate_level, lambda a: a)

            for bed in iter_list:
                temp = data[
                    (data.dw_project_id == project) &
                    (data.num_of_bedrooms == bed)
                    ].copy()

                if temp.empty:
                    continue

                processed_temp = rolling_process(temp)

                rolling_data = pd.concat([rolling_data, processed_temp], axis='rows', ignore_index=True)

        rolling_data = self.calculate_launching_period(rolling_data)

        return rolling_data

    def preprocess_forecasting_data(self, min_proj_size=50):

        data = self.query_raw_data(mode='forecasting')

        local_keys = {
            'project': ['dw_project_id'],
            'bedrooms': ['dw_project_id', 'num_of_bedrooms']
        }.get(self.aggregate_level, lambda a: a)

        data = data.merge(self.data[local_keys], how='left', on=local_keys, indicator=True)
        data = data[
            (data['_merge'] == 'left_only') &
            (data['proj_num_of_units'] > min_proj_size)
            ]

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


@dataclass
class CMData:
    aggregate_level: Literal['project', 'bedrooms']
    mode: Literal['training', 'forecasting']
    data: pd.DataFrame
