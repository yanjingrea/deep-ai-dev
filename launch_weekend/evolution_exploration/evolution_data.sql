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
                              min(to_date(transaction_month_index, 'YYYYMM')) as transaction_month,
                              launch_date,
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
                          group by 1, launch_date
                      )
            ,
        base as (
                    select
                        dw_project_id,
                        project_display_name,
                        case when project_display_name in ('Hillhaven', 'The Arcady At Boon Keng') then '2024-01-01' else launch_date end as launch_date,
                        transaction_month,
                        case when sales is null then 0 else sales end as sales,
                        proj_num_of_units,
                        average_launch_psf,
                        latitude,
                        longitude,
                        region_group
                    from data_science.ui_master_sg_project_geo_view_filled_features_condo b
                         left join base_sales c
                                   using (dw_project_id)
                         left join (
                                       select
                                           *
                                       from (
                                                select
                                                    project_dwid as dw_project_id,
                                                    project_display_name,
                                                    latitude,
                                                    longitude
                                                from ui_app.project_summary_prod_sg
                                                where property_group = 'condo'
                                                  and property_type != 'ec'
                                                  and launch_date is not null
                                            ) a
                                   ) as project_static
                                   using (dw_project_id)
                    where left(project_launch_month, 4)::int >= 2019 and proj_num_of_units >= 75
                ),
        base_com as (
                        select
                            p_base.project_display_name,
                            p_base.launch_date,
                            p_base.average_launch_psf,
                            p_base.sales,
                            p_base.proj_num_of_units as num_of_units,
                            p_base.sales::float / p_base.proj_num_of_units::float as sales_rate,
                            p_base.longitude,
                            p_base.latitude,
                            p_base.region_group,
                            c_base.project_display_name as ref_projects,
                            c_base.launch_date as ref_launch_date,
                            c_base.average_launch_psf as ref_launch_psf,
                            c_base.proj_num_of_units as ref_num_of_units,
                            c_base.sales::float / c_base.proj_num_of_units::float as ref_sales_rate,
                            c_base.longitude as ref_longitude,
                            c_base.latitude as ref_latitude,
                            ST_DistanceSphere(
                                    st_point(p_base.longitude, p_base.latitude),
                                    st_point(c_base.longitude, c_base.latitude)
                            ) as distance,
                            datediff(day, c_base.launch_date::date, p_base.launch_date::date) as days_gap,
                                    dense_rank()
                                    over (partition by p_base.project_display_name order by days_gap) as rank
                        from base as p_base
                             left join base as c_base
                                  on ST_DistanceSphere(
                                             st_point(p_base.longitude, p_base.latitude),
                                             st_point(c_base.longitude, c_base.latitude)
                                     ) <= 3000
                                      and p_base.launch_date > c_base.launch_date
                                      --and abs(p_base.average_launch_psf / c_base.average_launch_psf - 1) <= 0.15
                        order by p_base.launch_date, rank, distance
                    )
    select
        *
    from base_com
    where rank = 1 and project_display_name is not null