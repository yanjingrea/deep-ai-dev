with
    base_static as (
                       select
                           *
                       from (
                                select
                                    project_dwid as dw_project_id,
                                    project_display_name,
                                    launch_date,
                                    tenure,
                                    latitude,
                                    longitude,
                                    building_count,
                                    unit_count,
                                    residential_unit_count,
                                    commercial_unit_count,
                                    average_launch_psf,
                                    average_launch_price,
                                    max_floor_count,
                                    project_age,
                                    land_max_gfa,
                                    land_size_sqft
                                from ui_app.project_summary_prod_sg
                                where property_group = 'condo'
                                  and property_type != 'ec'
                                  and launch_date is not null
                            ) a
                            left join (
                                          select
                                              dw_project_id,
                                              project_units_zero_rm,
                                              project_units_one_rm,
                                              project_units_two_rm,
                                              project_units_three_rm,
                                              project_units_four_rm,
                                              project_units_five_rm,
                                              project_avg_size_of_zero_rm,
                                              project_avg_size_of_one_rm,
                                              project_avg_size_of_two_rm,
                                              project_avg_size_of_three_rm,
                                              project_avg_size_of_four_rm,
                                              project_avg_size_of_five_rm,
                                              region,
                                              zone,
                                              neighborhood,
                                              district
                                          from data_science.ui_master_sg_project_geo_view_filled_features_condo
                                      ) as b
                                      using (dw_project_id)

                   ),
    base_launch_data as (
                            select
                                dw_project_id,
                                min(transaction_date) as launch_date
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
                                       and property_type != 'ec'
                                 ) as "*2"
                            where seq = 1
                            group by 1
                        ),
    base_sales as (
                      select
                          a.dw_project_id,
                          count(dw_property_id) as sales
                      from base_launch_data a
                           left join data_science.ui_master_sg_transactions_view_filled_features_condo b
                                     on a.dw_project_id = b.dw_project_id
                                         and b.transaction_date::date <= dateadd(day, 7, launch_date::date)
                      group by 1
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
    a.*,
    left(launch_date, 4)::int as launch_year,
    case when sales is null then 0 else sales end as sales_quantity,
    sales_quantity:: float / residential_unit_count:: float as sales_rate,
    c.*
from base_static a
     left join base_sales b
               using (dw_project_id)
     left join base_geo c
               using (dw_project_id)
where left(launch_date, 4)::int >= 2010
    and residential_unit_count != 0
order by launch_date desc
;