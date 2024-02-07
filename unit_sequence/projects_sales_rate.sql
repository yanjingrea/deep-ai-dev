with
    base_static as (
                       select
                           project_dwid as dw_project_id,
                           project_display_name,
                           tenure,
                           latitude,
                           longitude,
                           expected_occupancy_quarter,
                           launch_date,
                           est_absd_deadline_quarter,
                           first_resale_date,
                           sold_out_date,
                           building_count,
                           unit_count,
                           residential_unit_count,
                           commercial_unit_count,
                           average_launch_psf,
                           average_launch_price,
                           project_property_types,
                           property_group,
                           property_type,
                           max_floor_count,
                           district_id,
                           neighborhood_id,
                           region_admin_subdistrict_code,
                           median_monthly_rent_yield,
                           project_age,
                           land_max_gfa,
                           land_size_sqft
                       from ui_app.project_summary_prod_sg
                       where property_group = 'condo'
                         and property_type != 'ec'
                         and launch_date is not null
                       order by launch_date desc
                   ),
    base_sales as (
                      select
                          dw_project_id,
                          count(*) as sales,
                          sales::float / residential_unit_count::float as sales_rate
                      from data_science.ui_master_sg_transactions_view_filled_features_condo a
                           join base_static b
                                using (dw_project_id)
                      where property_type_group = 'Condo'
                        and a.property_type != 'ec'
                        and transaction_sub_type = 'new sale'
                        and a.transaction_date <= dateadd(day, 7, launch_date)
                      group by 1, residential_unit_count, launch_date
                      order by launch_date desc
                  ),
    base_geo as (
                    select
                        dw_project_id,
                        max(km_to_sg_cbd) as km_to_sg_cbd,
                        max(num_of_bus_stops) as num_of_bus_stops,
                        max(meters_to_mrt) as meters_to_mrt,
                        max(num_of_good_schools) as num_of_good_schools
                    from data_science.ui_master_sg_properties_view_filled_static_features_condo p
                         join data_science.ui_master_sg_building_view_filled_features_condo b
                              using (dw_building_id)
                    group by 1
                )
select
    *
from base_static
     join base_sales
          using (dw_project_id)
     join base_geo
          using (dw_project_id)
order by launch_date desc
;
;