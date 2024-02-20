from datetime import datetime
import pickle
import geopandas as gpd
from shapely import wkt
from constants.utils import OUTPUT_DIR


current_month = datetime.today().replace(day=1).date()

raw_data_path = f'{OUTPUT_DIR}neighborhood_features.plk'

if True:
    from constants.redshift import query_data

    data = query_data(
        f"""
        with base_new_transactions_condo as (
            select
                b.neighborhood,
                avg(unit_price_psf) as avg_psf_new_sale_condo
            from data_science.ui_master_sg_transactions_view_filled_features_condo a
            join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                using (dw_project_id)
            where  transaction_sub_type = 'new sale'
                and transaction_date > dateadd(year, -1, '{current_month}')
            group by 1
        ), base_resale_transactions_condo as (
            select
                b.neighborhood,
                avg(unit_price_psf) as avg_psf_resale_condo
            from data_science.ui_master_sg_transactions_view_filled_features_condo a
            join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                using (dw_project_id)
            where transaction_sub_type = 'resale'
                and transaction_date > dateadd(year, -1, '{current_month}')
            group by 1
        ), base_resale_transactions_hdb as (
            select
                b.neighborhood,
                avg(unit_price_psf) as avg_psf_resale_hdb
            from data_science.ui_master_sg_transactions_view_filled_features_hdb a
            join data_science.ui_master_sg_building_view_filled_features_hdb b
                using (dw_building_id)
            where transaction_sub_type = 'resale'
                and transaction_date > dateadd(year, -1, '{current_month}')
            group by 1
        ), base_new_transactions_landed as (
            select
                b.neighborhood,
                avg(unit_price_psf) as avg_psf_new_sale_landed
            from data_science.ui_master_sg_transactions_view_filled_features_landed a
            join data_science.ui_master_sg_project_geo_view_filled_features_landed b
                using (dw_project_id)
            where transaction_sub_type = 'new sale'
                and transaction_date > dateadd(year, -1, '{current_month}')
            group by 1
        ), base_price as (
            select
                neighborhood,
                avg_psf_new_sale_condo,
                avg_psf_resale_condo,
                avg_psf_resale_hdb,
                avg_psf_new_sale_landed
            from base_new_transactions_condo
            full outer join base_resale_transactions_condo
                using (neighborhood)
            full outer join base_resale_transactions_hdb
                using (neighborhood)
            full outer join base_new_transactions_landed
                using (neighborhood)
            where neighborhood is not null
            order by 1
        ), base_hdb_count as (
            select
                b.area_id,
                max(b.count_value) as num_of_completed_units_hdb
            from ui_app.area_summary_prod_sg a
            full outer join ui_app.area_count_summary_prod_sg b
                on a.id = b.area_id
            where location_level = 'neighborhood'
                and activity_type = 'sale'
                and count_type = 'num-of-completed-units'
                and unit_mix = 'all'
                and index_date > dateadd(year, -1, '{current_month}')
                and property_type_group = 'public-stack'
                and area_id is not null
            group by 1
        ), base_condo_count as (
            select
                b.area_id,
                max(b.count_value) as num_of_completed_units_condo
            from ui_app.area_summary_prod_sg a
            full outer join ui_app.area_count_summary_prod_sg b
                on a.id = b.area_id
            where location_level = 'neighborhood'
                and activity_type = 'sale'
                and count_type = 'num-of-completed-units'
                and unit_mix = 'all'
                and index_date > dateadd(year, -1, '{current_month}')
                and property_type_group = 'private-stack'
                and area_id is not null
            group by 1
        ), base_landed_count as (
            select
                b.area_id,
                max(b.count_value) as num_of_completed_units_landed
            from ui_app.area_summary_prod_sg a
            full outer join ui_app.area_count_summary_prod_sg b
                on a.id = b.area_id
            where location_level = 'neighborhood'
                and activity_type = 'sale'
                and count_type = 'num-of-completed-units'
                and unit_mix = 'all'
                and index_date > dateadd(year, -1, '{current_month}')
                and property_type_group = 'private-land'
                and area_id is not null
            group by 1
        ), base_buyer_profile as (
                select
                neighborhood_id as area_id,
                avg(buyer_profile_company_percent) as buyer_profile_company_percent,
                avg(buyer_profile_foreigner_percent) as buyer_profile_foreigner_percent,
                avg(buyer_profile_pr_percent) as buyer_profile_pr_percent,
                avg(buyer_profile_sg_percent) as buyer_profile_sg_percent
            from ui_app.project_summary_prod_sg
            where property_group = 'condo'
                and launch_date > '2008-01-01'
                and (
                        buyer_profile_company_percent != 0 or
                        buyer_profile_foreigner_percent != 0 or
                        buyer_profile_pr_percent != 0 or
                        buyer_profile_sg_percent != 0
                )
                and neighborhood_id is not null
            group by 1
        ), base_age_size_completed_condo as (
            select
                neighborhood_id as area_id,
                sum(project_age * unit_count) / sum (unit_count) as avg_age_completed_condo,
                avg(unit_count) as avg_size_completed_condo
            from ui_app.project_summary_prod_sg a
            where project_status = 'completed'
                and property_type_group = 'private-stack'
                and unit_count != 0
            group by neighborhood_id
        ), base_age_size_incomplete_condo as (
                select
                neighborhood_id as area_id,
                sum(project_age * unit_count) / sum (unit_count) as avg_age_incomplete_condo,
                avg(unit_count) as avg_size_incomplete_condo
            from ui_app.project_summary_prod_sg a
            where project_age <= 0
                and property_type_group = 'private-stack'
                and unit_count != 0
            group by neighborhood_id, project_status
        ), base_age_group as (
            with base as (
                select
                    *,
                    case when count_sub_type in ('0-14', '15-24') then '0-24'
                        when count_sub_type in ('25-34', '35-54') then '25-54'
                        when count_sub_type in ('55-64', '65+') then '55+'
                    end as general_age_group
                from ui_app.area_demographic_summary_prod_sg
                where count_type = 'age-group'
            ), base_general_count as (
                select
                area_id,
                index_date,
                general_age_group,
                sum(count_value) as num_of_residents
            from base
            group by 1, 2, 3
            )
            select
                area_id,
                num_of_residents_0_24,
                num_of_residents_25_54,
                num_of_residents_55_up
            from (
                select area_id, index_date, num_of_residents as num_of_residents_0_24
                from base_general_count where general_age_group = '0-24'
            ) a
            full outer join (
                select area_id, index_date, num_of_residents as num_of_residents_25_54
                from base_general_count where general_age_group = '25-54'
            ) b
            using (area_id, index_date)
            full outer join (
                select area_id, index_date, num_of_residents as num_of_residents_55_up
                from base_general_count where general_age_group = '55+'
            ) c
            using (area_id, index_date)
            where index_date = (
                select max(index_date) from ui_app.area_demographic_summary_prod_sg
                where count_type = 'age-group'
                )
            order by 1, 2
        )
        select
            a.label,
            a.geom as geometry,
            b.num_of_completed_units_condo,
            c.num_of_completed_units_hdb,
            e.num_of_completed_units_landed,
            d.buyer_profile_company_percent,
            d.buyer_profile_foreigner_percent,
            d.buyer_profile_pr_percent,
            d.buyer_profile_sg_percent,
            avg_psf_new_sale_condo,
            avg_psf_resale_condo,
            avg_psf_resale_hdb,
            avg_psf_new_sale_landed,
            avg_age_completed_condo,
            avg_size_completed_condo,
            avg_age_incomplete_condo,
            avg_size_incomplete_condo,
            num_of_residents_0_24,
            num_of_residents_25_54,
            num_of_residents_55_up
        from ui_app.area_summary_prod_sg a
        full outer join base_condo_count b
            on a.id = b.area_id
        full outer join base_hdb_count c
            using (area_id)
        full outer join base_buyer_profile d
            using (area_id)
        full outer join base_landed_count e
            using (area_id)
        full outer join base_price f
            on lower(a.label) = f.neighborhood
        full outer join base_age_size_completed_condo g
            using (area_id)
        full outer join base_age_size_incomplete_condo h
            using (area_id)
        full outer join base_age_group i
            using (area_id)
        where location_level = 'neighborhood'
        """
    )

    pickle.dump(
        data, open(raw_data_path, 'wb')
    )

else:
    data = pickle.load(open(raw_data_path, 'rb'))

data['geometry'] = data['geometry'].apply(wkt.loads)
df = gpd.GeoDataFrame(data, geometry='geometry')
# the original crs (EPSG:4326, unit=degree)
# we need to declare this first
df.geometry.crs = {'init': 'epsg:4326'}

# to calculate the area in square kilometers
# change the projection to a Cartesian system (EPSG:3857, unit= m)
# and then divide by 10**6
df['size_sqkm'] = df.geometry.to_crs({'init': 'epsg:3857'}).area / 10 ** 6

for property_group in ['condo', 'hdb', 'landed']:
    num_col = f'num_of_completed_units_{property_group}'
    den_col = f'density_completed_units_{property_group}'
    df[den_col] = df[num_col] / df['size_sqkm']

for age_group in ['0_24', '25_54', '55_up']:
    num_col = f'num_of_residents_{age_group}'
    den_col = f'density_of_residents_{age_group}'
    df[den_col] = df[num_col] / df['size_sqkm']
