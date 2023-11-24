from dataclasses import dataclass
import pandas as pd
from constants.redshift import query_data


@dataclass
class PartialCoef:

    def query_scripts(self) -> str:
        ...

    def query_coef_table(self) -> pd.DataFrame:
        ...

    def __post_init__(self):
        self.reference_table = self.query_coef_table()

    def get_coef(self, **kwargs):
        ...


@dataclass
class AreaCoef(PartialCoef):

    @property
    def query_scripts(self):
        scripts = """
        select
                floor_area_sqft as area_lower_bound,
                lag(floor_area_sqft, 1) over (order by floor_area_sqft desc) as next_area,
                case when next_area is null then floor_area_sqft * 1000 else next_area end as area_upper_bound,
                (
                    select coefficient
                    from data_science_test.partial_coef_floor_area_sqft_sg_country
                    where coef_change = 0
                ) as base_coef,
                1 / (1 + coefficient - base_coef) as area_adjust_coef
        from data_science_test.partial_coef_floor_area_sqft_sg_country
        """
        return scripts

    def query_coef_table(self):
        df = query_data(self.query_scripts)

        return df[['area_lower_bound', 'area_upper_bound', 'area_adjust_coef']]

    def __post_init__(self):
        super().__post_init__()

    def get_coef(self, floor_area_sqft):

        if floor_area_sqft < self.reference_table['area_lower_bound'].min():
            return self.reference_table['area_adjust_coef'].min()
        elif floor_area_sqft > self.reference_table['area_upper_bound'].max():
            return self.reference_table['area_adjust_coef'].max()

        try:
            return self.reference_table[
                (self.reference_table['area_lower_bound'] <= floor_area_sqft) &
                (self.reference_table['area_upper_bound'] > floor_area_sqft)
                ]['area_adjust_coef'].iloc[0]
        except IndexError:
            return None

    def get_segment_coef(self, segment_range: tuple):
        try:
            a = self.reference_table[
                (self.reference_table['area_upper_bound'] >= segment_range[0]) &
                (self.reference_table['area_lower_bound'] <= segment_range[1])
                ]

            return a.rename(
                columns={
                    # 'area_lower_bound': 'floor_area_sqft',
                    'area_adjust_coef': 'coef'
                }
            )

        except IndexError:
            return None


@dataclass
class FloorCoef(PartialCoef):
    @property
    def query_scripts(self):
        scripts = """
            select
                address_floor_num,
                (
                    select coefficient
                    from data_science_test.partial_coef_address_floor_num_sg_country
                    where coef_change = 0
                ) as base_coef,
                1 / (1 + coefficient - base_coef) as floor_adjust_coef
            from data_science_test.partial_coef_address_floor_num_sg_country
            """
        return scripts

    def query_coef_table(self):
        df = query_data(self.query_scripts)
        return df

    def __post_init__(self):
        super().__post_init__()

    def get_coef(self, address_floor_num):

        try:
            return self.reference_table[
                self.reference_table['address_floor_num'] == address_floor_num
                ]['floor_adjust_coef'].iloc[0]
        except IndexError:
            return None

    def get_segment_coef(self, segment_range):
        try:
            a = self.reference_table[
                (self.reference_table['address_floor_num'] >= segment_range[0]) &
                (self.reference_table['address_floor_num'] <= segment_range[1])
                ]

            return a.rename(
                columns={
                    'floor_adjust_coef': 'coef'
                }
            )

        except IndexError:
            return None


@dataclass
class TimeIndex(PartialCoef):

    @property
    def query_scripts(self) -> str:
        return """
        select 
            transaction_month_index, 
            hi_avg_improved as rebase_index,
            (
                    select hi_avg_improved
                    from data_science.sg_condo_resale_index_sale
                    order by transaction_month_index desc limit 1
            ) as current_index,
            1 / rebase_index * current_index as time_adjust_coef 
        from data_science.ui_master_daily_sg_index_sale umdsis
        where property_group = 'condo'
            and index_area_type = 'country'
        """

    def query_coef_table(self):
        df = query_data(self.query_scripts)
        return df

    def __post_init__(self):
        super().__post_init__()

    def get_coef(self, transaction_month_index):

        try:
            return self.reference_table[
                self.reference_table['transaction_month_index'] == transaction_month_index
                ]['time_adjust_coef'].iloc[0]
        except IndexError:
            return None

    def get_segment_coef(self, segment_range):
        try:

            self.reference_table['transaction_month_index'] = pd.to_datetime(
                self.reference_table['transaction_month_index'], format='%Y%m', errors='coerce'
            )

            a = self.reference_table[
                (self.reference_table['transaction_month_index'] >= segment_range[0]) &
                (self.reference_table['transaction_month_index'] <= segment_range[1])
                ].copy()

            return a.rename(
                columns={
                    'rebase_index': 'coef'
                }
            )

        except IndexError:
            return None


class ZoneCoef(PartialCoef):
    @property
    def query_scripts(self):
        scripts = """
            select
                2023 as transaction_year,
                zone,
                1 / hi_coef as zone_adjust_coef
            from data_science.sg_panel_zone_year_index
            where transaction_year = 2022
            union
            select
                transaction_year,
                zone,
                1 / hi_coef as zone_adjust_coef
            from data_science.sg_panel_zone_year_index
            """
        return scripts

    def query_coef_table(self):
        df = query_data(self.query_scripts)
        return df

    def __post_init__(self):
        super().__post_init__()

    def get_coef(self, transaction_year, zone):

        try:
            return self.reference_table[
                (self.reference_table['transaction_year'] == transaction_year) &
                (self.reference_table['zone'] == zone)
            ]['zone_adjust_coef'].iloc[0]
        except IndexError:
            return None


transaction_table = ...
property_table = ...
project_table = ...
