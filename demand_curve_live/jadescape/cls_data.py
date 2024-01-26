import pickle

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from demand_model_utils.scr_coef import query_adjust_coef
from demand_curve_live.jadescape.scr_get_paths import model_dir

ALL_BEDS = ['one', 'two', 'three', 'four', 'five']
bed_nums = np.arange(1, 6)


@dataclass
class BaseCMData:

    max_year_gap: Optional[int] = 3
    rolling_windows: Optional[int] = 3

    quantity: Optional[str] = 'sales'
    price: Optional[str] = 'price'
    project_key: Optional[str] = 'dw_project_id'

    def __post_init__(self):
        self.raw_data = self.query_raw_data()
        self.data = self.preprocess_base_training_data()

        self.month_index = self.rank_transaction_month()

    @property
    def index_table_query(self) -> str:
        return """
        select
            transaction_month_index,
            hi_avg_improved as rebase_index,
            (
              select
                  hi_avg_improved
              from data_science.sg_condo_resale_index_sale
              order by transaction_month_index desc
              limit 1
            ) as current_index,
            1 / rebase_index as time_adjust_coef
        from data_science.ui_master_daily_sg_index_sale umdsis
        where property_group = 'condo'
        and index_area_type = 'country'
        """

    def query_raw_data(self):

        raw_data_path = f'{model_dir}training_data.plk'
        if True:
            data = pickle.load(open(raw_data_path, 'rb'))
        else:
            from constants.redshift import query_data

            data = query_data(
                f"""
                with
                    base_new_launch_projects as (
                                                    select distinct
                                                        dw_project_id,
                                                        avg(latitude) as avg_lat,
                                                        avg(longitude) as avg_long
                                                    from data_science.ui_master_sg_project_geo_view_filled_features_condo a
                                                         join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                                                              using (dw_project_id)
                                                    where dw_project_id in ('ff32b92427bcd1254c42add72705d821')
                                                    group by 1
                                                ),
                    base_historical_project as (
                                                   select distinct
                                                       dw_project_id,
                                                       project_name,
                                                       avg(latitude) as avg_lat,
                                                       avg(longitude) as avg_long
                                                   from data_science.ui_master_sg_project_geo_view_filled_features_condo a
                                                        join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                                                             using (dw_project_id)
                                                   where left(project_launch_month, 4)::int >= 2015
                                                   group by 1, 2
                                               ),
                    base_nearby_projects as (
                                                select distinct
                                                    new_p.dw_project_id as base_project_id,
                                                    his_p.dw_project_id as nearby_project_id,
                                                    project_name,
                                                    ST_DistanceSphere(
                                                            st_point(his_p.avg_long, his_p.avg_lat),
                                                            st_point(new_p.avg_long, new_p.avg_lat)
                                                    ) as distance
                                                from base_new_launch_projects new_p
                                                     join base_historical_project his_p
                                                          on ST_DistanceSphere(
                                                                     st_point(his_p.avg_long, his_p.avg_lat),
                                                                     st_point(new_p.avg_long, new_p.avg_lat)
                                                             ) <= 30000
                                                order by distance
                                            ),
                    base_index as (
                                      {self.index_table_query}
                                  ),
                    base_floor_coef as (
                                           select
                                               address_floor_num,
                                               (
                                                   select
                                                       coefficient
                                                   from data_science_test.partial_coef_address_floor_num_sg_country
                                                   where coef_change = 0
                                               ) as base_coef,
                                               1 / (1 + coefficient - base_coef) as floor_adjust_coef
                                           from data_science_test.partial_coef_address_floor_num_sg_country
                                       ),
                    base_area_coef as (
                                          select
                                              floor_area_sqft as area_lower_bound,
                                              lag(floor_area_sqft, 1) over (order by floor_area_sqft desc) as next_area,
                                              case when next_area is null then floor_area_sqft * 1000 else next_area end as area_upper_bound,
                                              (
                                                  select
                                                      coefficient
                                                  from data_science_test.partial_coef_floor_area_sqft_sg_country
                                                  where coef_change = 0
                                              ) as base_coef,
                                              1 / (1 + coefficient - base_coef) as area_adjust_coef
                                          from data_science_test.partial_coef_floor_area_sqft_sg_country
                                      ),
                
                    base_property_price as (
                                               select
                                                   *
                                               from (
                                                        select
                                                            *,
                                                                    row_number()
                                                                    over (partition by dw_property_id order by transaction_date desc) as seq
                                                        from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                                             join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                  using (dw_project_id)
                                                        where a.property_type_group = 'Condo'
                                                          and transaction_sub_type = 'new sale'
                                                    ) as "*2"
                                               where seq = 1
                                           )
                        ,
                    base_property_panel as (
                                               select
                                                   b.dw_project_id,
                                                   num_of_bedrooms,
                                                   to_date(transaction_month_index, 'YYYYMM') as transaction_month,
                                                   time_adjust_coef,
                                                   avg(
                                                               unit_price_psf
                                                               * floor_adjust_coef
                                                               * area_adjust_coef
                                                               * time_adjust_coef
                                                   ) as price,
                                                   avg(floor_area_sqm) as floor_area_sqm,
                                                   count(*) as sales
                                               from base_property_price a
                                                    left outer join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                    using (dw_project_id)
                                                    left outer join base_index c
                                                                    using (transaction_month_index)
                                                    left outer join base_floor_coef f
                                                                    using (address_floor_num)
                                                    left outer join base_area_coef g
                                                                    on a.floor_area_sqft >= g.area_lower_bound and
                                                                       a.floor_area_sqft < g.area_upper_bound
                                               group by 1, 2, 3, 4
                
                                           ),
                    base_ura_launch_info as (
                                                select
                                                    b.project_dwid as dw_project_id,
                                                    to_date(concat(reference_year, reference_month), 'YYYYMM') as transaction_month,
                                                    a.units_launched_in_the_month as num_of_units_launched
                                                from raw_reference.sg_ura_developer_sale a
                                                     join ui_app.project_summary_prod_sg b
                                                          on lower(a.project_name) = lower(b.project_display_name)
                                                ORDER BY b.project_display_name, reference_year desc, reference_month desc
                                            )
                select
                    dw_project_id,
                    c.project_display_name as project_name,
                    num_of_bedrooms,
                    to_date(project_launch_month, 'YYYYMM') as launch_year_month,
                    transaction_month,
                    datediff(month, launch_year_month, transaction_month) + 1 as launching_period,
                    price,
                    case
                        when num_of_bedrooms = 1 and c.project_display_name = 'Jadescape' then project_units_one_rm - 2
                        when num_of_bedrooms = 1 or num_of_bedrooms = 0 then project_units_zero_rm + project_units_one_rm
                        when num_of_bedrooms = 2 then project_units_two_rm
                        when num_of_bedrooms = 3 then project_units_three_rm
                        when num_of_bedrooms = 4 then project_units_four_rm
                        when num_of_bedrooms = 5 then project_units_five_rm
                        end
                        as num_of_units,
                    sales,
                    case
                        when lag(sales, 1) over (
                            partition by dw_project_id, num_of_bedrooms
                            order by transaction_month
                            ) is null then num_of_units
                        else num_of_units - sum(sales) over (
                            partition by dw_project_id, num_of_bedrooms
                            order by transaction_month
                            rows between unbounded preceding and 1 preceding
                            )
                        end as num_of_remaining_units,
                    proj_num_of_units,
                    tenure,
                    case
                        when num_of_bedrooms = 1 or num_of_bedrooms = 0 then (
                                                                                         project_avg_size_of_zero_rm *
                                                                                         project_units_zero_rm +
                                                                                         project_avg_size_of_one_rm *
                                                                                         project_units_one_rm
                                                                                 ) / num_of_units
                        when num_of_bedrooms = 2 then project_avg_size_of_two_rm
                        when num_of_bedrooms = 3 then project_avg_size_of_three_rm
                        when num_of_bedrooms = 4 then project_avg_size_of_four_rm
                        when num_of_bedrooms = 5 then project_avg_size_of_five_rm
                        end
                        as floor_area_sqm,
                    proj_max_floor,
                    time_adjust_coef,
                    zone,
                    neighborhood,
                    distance,
                    case when 
                        (num_of_units_launched is null or num_of_units_launched=0) 
                        and launching_period = 1 then proj_num_of_units
                        when num_of_units_launched is null and launching_period != 1 then 0
                    else num_of_units_launched end as num_of_units_launched
                from base_property_panel a
                     join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                          using (dw_project_id)
                     join (
                              select
                                  project_dwid as dw_project_id,
                                  project_display_name
                              from ui_app.project_summary_prod_sg
                          ) c
                          using (dw_project_id)
                     join base_nearby_projects d
                          on a.dw_project_id = d.nearby_project_id
                     left join base_ura_launch_info
                          using (dw_project_id, transaction_month)
                order by 1, 2, 3, 4
            """
            )
            pickle.dump(data, open(raw_data_path, 'wb'))

        # categorical data processing
        if 'tenure' in data.columns:
            data['tenure'] = data['tenure'].apply(lambda a: 1 if a == 'freehold' else 0)
        data['transaction_month'] = pd.to_datetime(data['transaction_month'])
        # data['is_first_period'] = data['launching_period'].apply(lambda a: 1 if a == 1 else 0)
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

        project_data['minor_launch_year_month'] = project_data.apply(
            lambda row: row['transaction_month']
            if row['num_of_units_launched'] != 0 or row['launching_period'] == 1
            else np.nan,
            axis=1
        )

        project_data['minor_launch_year_month'] = project_data['minor_launch_year_month'].fillna(method='ffill')
        project_data['minor_launching_period'] = project_data.apply(
            lambda row: len(
                pd.period_range(
                    start=row['minor_launch_year_month'],
                    end=row['transaction_month'],
                    freq='M'
                )
            ), axis=1
        )

        return project_data

    def preprocess_base_training_data(self):

        data = self.raw_data

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
            re_expended_data['num_of_units_launched'] = foreward_cumsum(re_expended_data['num_of_units_launched'])

            final_data = re_expended_data.iloc[wins - 1:].dropna(subset=self.quantity)
            final_data = final_data[final_data[self.quantity] != 0].fillna(method='bfill').fillna(method='ffill')

            if np.any(final_data['num_of_remaining_units'] < 0):

                print(
                    f'{time_series_data["project_name"].iloc[0]} - '
                    f'{time_series_data["num_of_bedrooms"].iloc[0]}: '
                    f'please check the method of rolling method'
                )

                # raise Exception(f'please check the method of rolling method')

            return final_data.reset_index(drop=True)

        rolling_data = pd.DataFrame()
        for project in data.dw_project_id.unique():

            for bed in bed_nums:
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

    def rank_transaction_month(self):

        month_range = pd.date_range(
            start=self.data['transaction_month'].min(),
            end=max(self.data['transaction_month'].max(), self.raw_data['transaction_month'].max()),
            freq='MS',
            inclusive='both'
        )

        month_rank = np.arange(1, len(month_range) + 1)

        # self.__setattr__(
        #     'index_order',
        #     pd.Series(
        #         data=month_rank,
        #         index=month_range
        #     )
        # )

        index_order = pd.Series(
            data=month_rank,
            index=month_range
        )

        for dt in [self.data, self.raw_data]:
            dt['transaction_month_idx'] = dt['transaction_month'].apply(lambda tm: index_order.loc[tm])

        return index_order

    def get_rebased_project_data(
        self,
        project_name,
        *,
        mode: Literal['raw', 'rolling'],
        num_of_bedrooms=None
    ):

        dataset = {
            'raw': self.raw_data,
            'rolling': self.data
        }.get(mode, lambda a: a)

        proj_data = dataset[dataset['project_name'] == project_name].copy()

        if num_of_bedrooms:
            proj_data = proj_data[proj_data['num_of_bedrooms'] == num_of_bedrooms]

        return self.calculate_launching_period(proj_data)

    def get_nominal_project_data(
        self,
        project_name,
        *,
        mode: Literal['raw', 'rolling'],
        num_of_bedrooms=None
    ):

        rebased_project_data = self.get_rebased_project_data(
            project_name,
            mode=mode,
            num_of_bedrooms=num_of_bedrooms
        )
        coef_to_multiply = query_adjust_coef(rebased_project_data)
        p = rebased_project_data[self.price] * coef_to_multiply / rebased_project_data['time_adjust_coef']

        rebased_project_data[self.price] = p

        return rebased_project_data
