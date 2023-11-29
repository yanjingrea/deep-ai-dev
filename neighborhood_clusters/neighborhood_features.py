from datetime import datetime


import geopandas as gpd
from shapely import wkt


from constants.redshift import query_data
from constants.utils import OUTPUT_DIR

raw_data_path = f'{OUTPUT_DIR}neighborhood_features.plk'
if True:
    current_month = datetime.today().replace(day=1).date()

    property_type_mapping = {
        'public-stack': 'hdb',
        'private-stack': 'condo',
        'private-land': 'landed'
    }

    completed_units_scripts = lambda property_type: f"""
            select
                b.area_id,
                max(b.count_value) as num_of_completed_units_{property_type_mapping[property_type]},
                avg(b.average_psf) as average_psf_{property_type_mapping[property_type]}
            from ui_app.area_summary_prod_sg a
            full outer join ui_app.area_count_summary_prod_sg b
                on a.id = b.area_id
            where location_level = 'neighborhood'
                and activity_type = 'sale'
                and count_type = 'num-of-completed-units'
                and unit_mix = 'all'
                and index_date > dateadd(year, -1, '{current_month}') 
                and property_type_group = '{property_type}'
                and area_id is not null
            group by 1
    """

    data = query_data(
        f"""
        with base_hdb_count as (
            {completed_units_scripts('public-stack')}
        ), base_condo_count as (
            {completed_units_scripts('private-stack')}
        ), base_landed_count as (
            {completed_units_scripts('private-land')}
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
        )
        select
            a.label,
            a.geom as geometry,
            b.num_of_completed_units_condo,
            b.average_psf_condo,
            c.num_of_completed_units_hdb,
            c.average_psf_hdb,
            e.num_of_completed_units_landed,
            e.average_psf_landed,
            d.buyer_profile_company_percent,
            d.buyer_profile_foreigner_percent,
            d.buyer_profile_pr_percent,
            d.buyer_profile_sg_percent
        from ui_app.area_summary_prod_sg a
        full outer join base_condo_count b
            on a.id = b.area_id
        full outer join base_hdb_count c
            using (area_id)
        full outer join base_buyer_profile d
            using (area_id)
        full outer join base_landed_count e
            using (area_id)
        where location_level = 'neighborhood'
        """
    )

    # pickle.dump(
    #     data, open(raw_data_path, 'wb')
    # )

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

if False:
    neighborhood_data = query_data(
        f"""
        SELECT
            as2.id,
            as2.slug,
            as2.label,
            as2.geom as geometry
        from common_sg.area_summary as2
        left join translation_sg.ui_search_terms ust
            on ust.original_text = as2.location_level
            and ust.search_field ='location_level'
        where location_level = 'neighborhood'
        """
    )

    district_data = query_data(
        f"""
        SELECT
            as2.id,
            as2.slug,
            as2.label,
            as2.geom
        from common_sg.area_summary as2
        left join translation_sg.ui_search_terms ust
            on ust.original_text = as2.location_level
            and ust.search_field ='location_level'
        where location_level = 'district'
        """
    )

