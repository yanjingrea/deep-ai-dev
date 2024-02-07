with
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
                        )
        ,
    base_sales as (
                      select
                          a.dw_project_id,
                          num_of_bedrooms,
                          min(to_date(transaction_month_index, 'YYYYMM')) as transaction_month,
                          (
                              select
                                  non_landed_index
                              from developer_tool.sg_gov_residential_index
                              order by quarter_index desc
                              limit 1
                          ) as current_index,
                          count(dw_property_id) as sales,
                          avg(unit_price_psf / c.non_landed_index * current_index) as average_launch_psf,
                          avg(transaction_amount / c.non_landed_index * current_index) as average_launch_price
                      from base_launch_data a
                           left join data_science.ui_master_sg_transactions_view_filled_features_condo b
                                     on a.dw_project_id = b.dw_project_id
                                         and b.transaction_date::date <= dateadd(day, 7, launch_date::date)
                           join developer_tool.sg_gov_residential_index c
                                on b.transaction_quarter_index = c.quarter_index
                      group by 1, 2
                  )
        ,
    base as (
                select
                    dw_project_id,
                    project_launch_month,
                    transaction_month,
                    num_of_bedrooms,
                    case
                        when num_of_bedrooms = 0 then project_units_zero_rm
                        when num_of_bedrooms = 1 then project_units_one_rm
                        when num_of_bedrooms = 2 then project_units_two_rm
                        when num_of_bedrooms = 3 then project_units_three_rm
                        when num_of_bedrooms = 4 then project_units_four_rm
                        when num_of_bedrooms = 5 then project_units_five_rm
                        when num_of_bedrooms = 6 then proj_num_of_units - project_units_zero_rm - project_units_one_rm -
                                                      project_units_two_rm - project_units_three_rm -
                                                      project_units_four_rm - project_units_five_rm
                        end
                        as num_of_units,
                    case when sales is null then 0 else sales end as sales,
                    average_launch_psf,
                    neighborhood
                from (
                         select distinct
                             dw_project_id,
                             num_of_bedrooms
                         from data_science.ui_master_sg_properties_view_filled_static_features_condo
                     ) as a
                     left join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                               using (dw_project_id)
                     left join base_sales c
                               using (dw_project_id, num_of_bedrooms)
                where left(project_launch_month, 4)::int >= 2015
            )
        ,
    base_comparable as (
                           select distinct
                               p_base.dw_project_id as dw_project_id,
                               p_base.num_of_bedrooms,
                               case when p_base.transaction_month is null then to_date(p_base.project_launch_month, 'YYYYMM') else p_base.transaction_month end as transaction_month,
                               case when p_base.sales is null then 0 else p_base.sales end as sales,
                               p_base.average_launch_psf as average_launch_psf,
                               p_base.num_of_units as num_of_units,
                               p_base.sales / p_base.num_of_units as sales_rate,
                               c_base.dw_project_id as ref_project_id,
                               c_base.sales as ref_sales,
                               c_base.average_launch_psf as ref_average_launch_psf,
                               c_base.num_of_units as ref_num_of_units,
                               c_base.sales / c_base.num_of_units as ref_sales_rate,
                               comparison_order,
                                       dense_rank()
                                       over (
                                           partition by p_base.dw_project_id, p_base.num_of_bedrooms
                                           order by abs(p_base.average_launch_psf / c_base.average_launch_psf - 1), comparison_order
                                           ) as similarity_order
                           from base as p_base
                                left join ui_app.project_comparables_prod_sg p
                                          on p.project_dwid = p_base.dw_project_id
                                              and comparable_type = 'similiar-new'
                                left join base as c_base on
                                       p.comparable_project_dwid = c_base.dw_project_id
                                   and p_base.num_of_bedrooms = c_base.num_of_bedrooms
                                   and p_base.project_launch_month >= c_base.project_launch_month
                                   and c_base.num_of_units > 0
--                                    and abs(p_base.average_launch_psf / c_base.average_launch_psf - 1) <= 0.15
                           where p_base.num_of_units > 0
                           order by 1, num_of_bedrooms, p.comparison_order
                       )
        ,
    base_static as (
                       select
                           *
                       from (
                                select
                                    project_dwid as dw_project_id,
                                    project_display_name,
                                    launch_date,
                                    tenure_type,
                                    tenure_int,
                                    latitude,
                                    longitude,
                                    building_count,
                                    unit_count,
                                    residential_unit_count,
                                    commercial_unit_count,
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
                ),
    base_nearby_data as (
                            with
                                base_raw_transactions as (
                                                             select
                                                                 b.neighborhood,
                                                                 transaction_month_index,
                                                                 unit_price_psf,
                                                                         row_number()
                                                                         over (partition by dw_property_id order by transaction_date desc) as seq
                                                             from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                                                  join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                       using (dw_project_id)
                                                             where a.property_type_group = 'Condo'
                                                               and transaction_sub_type = 'new sale'
                                                               and project_launch_month::int > 200912
                                                         ),
                                base_nearby_num_of_trans as (
                                                                select
                                                                    neighborhood,
                                                                    to_date(transaction_month_index, 'YYYYMM') as transaction_month,
                                                                    count(*) as num_of_transactions,
--                                                                     percentile_cont(0.25) within group (order by unit_price_psf) as percentile_25th_psf,
--                                                                     percentile_cont(0.75) within group (order by unit_price_psf) as percentile_75th_psf,
                                                                    avg(unit_price_psf) as average_psf
                                                                from base_raw_transactions
                                                                where seq = 1
                                                                group by 1, 2
                                                                order by 1, 2
                                                            ),
                                nearby_num_of_units as (
                                                           select
                                                               neighborhood,
                                                               to_date(project_launch_month, 'YYYYMM') as transaction_month,
                                                               sum(proj_num_of_units) as num_of_new_units
                                                           from data_science.ui_master_sg_project_geo_view_filled_features_condo
                                                           where project_launch_month::int > 200912
                                                           group by 1, 2
                                                           order by 1, 2
                                                       )
                            select
                                neighborhood,
                                transaction_month,
                                        sum(num_of_new_units)
                                        over (partition by neighborhood order by transaction_month rows between unbounded preceding and 1 preceding) as cum_stock,
                                        sum(num_of_transactions)
                                        over (partition by neighborhood order by transaction_month rows between unbounded preceding and 1 preceding) as com_sales,
                                cum_stock - com_sales as nearby_num_of_remaining_units,
                                        sum(num_of_new_units)
                                        over (partition by neighborhood order by transaction_month rows between 7 preceding and 1 preceding) as latest_half_year_launched_units,
                                        sum(num_of_transactions)
                                        over (partition by neighborhood order by transaction_month rows between 7 preceding and 1 preceding) as latest_half_year_sold_units,
                                average_psf as nearby_average_psf
                            from (
                                     select *
                                     from base_nearby_num_of_trans
                                 ) as "rnot*"
                                 left outer join nearby_num_of_units
                                                 using (neighborhood, transaction_month)
                            order by 1, 2
                        )
select
    num_of_bedrooms,
    a.*,
    left(launch_date, 4)::int as launch_year,
    sales,
    average_launch_psf,
    num_of_units,
    sales_rate * 100 as sales_rate,
    ref_project_id,
    ref_sales,
    ref_average_launch_psf,
    ref_num_of_units,
    ref_sales_rate,
    similarity_order,
    km_to_sg_cbd,
    num_of_bus_stops,
    meters_to_mrt,
    num_of_good_schools,
    nearby_num_of_remaining_units,
    case
        when latest_half_year_launched_units is null then 0
        else latest_half_year_launched_units end as latest_half_year_launched_units,
    case
        when latest_half_year_sold_units is null then 0
        else latest_half_year_sold_units end as latest_half_year_sold_units
from base_static a
     left join base_comparable b
               using (dw_project_id)
     left join base_geo c
               using (dw_project_id)
     left join base_nearby_data d
               using (neighborhood, transaction_month)
where similarity_order = 1
  and transaction_month >= '2010-01-01'
order by launch_date desc, project_display_name, num_of_bedrooms