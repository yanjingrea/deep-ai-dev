with
    base_unit_count as (
                           select
                               a.project_dwid,
                               unit_mix as num_of_bedrooms,
                               count_value as num_of_units
                           from ui_app.project_unit_count_prod_sg a
                                join ui_app.project_summary_prod_sg b
                                     on a.project_dwid = b.project_dwid
                                         and to_date(left(b.launch_date, 7), 'YYYY-MM') = a.index_date::date
                           where count_type = 'num-of-units'
                             and num_of_bedrooms in (0, 1, 2, 3, 4, 5)
                             and project_display_name not ilike '%-%Inactive%'
                       )
        ,
    base_trans as (
                      select
                          neighborhood_id,
                          project_dwid,
                          property_dwid,
                          num_of_bedrooms,
                          activity_psf,
                          activity_date_index as activity_date,
                          days_on_market
                      from (
                               select
                                   neighborhood_id,
                                   project_dwid,
                                   property_dwid,
                                   unit_mix as num_of_bedrooms,
                                   activity_psf,
                                   to_date(left(activity_date, 7), 'YYYY-MM') as activity_date_index,
                                           min(activity_date)
                                           over (partition by project_dwid) as actual_launch_date,
                                   datediff(days, actual_launch_date, activity_date) as days_on_market,
                                           row_number()
                                           over (partition by project_dwid, property_dwid order by activity_date) as seq
                               from ui_app.transacted_summary_prod_sg t
                               where t.property_group = 'condo'
                                 and t.activity_type = 'new-sale'
                                 and t.property_type != 'executive condominium'
                                 and unit_mix in (0, 1, 2, 3, 4, 5, 6)
                           ) as "*2"
                      where seq = 1
                  ),
    base_market_info as (
                            with
                                base_panel as (
                                                  select
                                                      *
                                                  from (
                                                           select distinct
                                                               neighborhood_id
                                                           from ui_app.project_summary_prod_sg
                                                       ) as a
                                                       cross join
                                                  (
                                                      select distinct
                                                          index_date as activity_date
                                                      from ui_app.area_count_summary_prod_sg
                                                  ) as b
                                                       cross join
                                                  (
                                                      select distinct
                                                          unit_mix as num_of_bedrooms
                                                      from ui_app.area_count_summary_prod_sg
                                                      where unit_mix in (0, 1, 2, 3, 4, 5)
                                                  ) as c
                                              )
                                    ,
                                base_geo_launch_cum as (
                                                           select
                                                               neighborhood_id,
                                                               activity_date,
                                                               num_of_bedrooms,
                                                                       sum(num_of_launched_projects) over
                                                                   (partition by neighborhood_id order by activity_date rows between unbounded preceding and current row)
                                                                   as cumulative_num_of_launched_projects,
                                                                       sum(case
                                                                               when num_of_launched_units is null then 0
                                                                               else num_of_launched_units end) over
                                                                           (partition by neighborhood_id order by activity_date rows between unbounded preceding and current row)
                                                                   as cumulative_num_of_launched_units,
                                                                       sum(num_of_launched_projects) over
                                                                   (partition by neighborhood_id order by activity_date rows between 11 preceding and current row)
                                                                   as rolling_num_of_launched_projects_neighborhood,
                                                                       sum(case
                                                                               when num_of_launched_units is null then 0
                                                                               else num_of_launched_units end) over
                                                                           (partition by neighborhood_id order by activity_date rows between 11 preceding and current row)
                                                                   as rolling_num_of_launched_units_neighborhood
                                                           from (
                                                                    select
                                                                        a.neighborhood_id,
                                                                        a.activity_date,
                                                                        num_of_bedrooms,
                                                                        count(project_dwid) as num_of_launched_projects,
                                                                        sum(num_of_units) as num_of_launched_units
                                                                    from base_panel a
                                                                         left join ui_app.project_summary_prod_sg b
                                                                                   on a.neighborhood_id =
                                                                                      b.neighborhood_id
                                                                                       and
                                                                                      a.activity_date = b.launch_date
                                                                         left join base_unit_count
                                                                                   using (project_dwid, num_of_bedrooms)
                                                                    group by 1, 2, 3
                                                                    order by 1, 2
                                                                ) as ab
                                                       ),
                                base_nearby_panel as (
                                                         select
                                                             neighborhood_id,
                                                             activity_date,
                                                             num_of_bedrooms,
                                                             count(*) as units_sold_project,
                                                             avg(activity_psf) as units_price_project
                                                         from base_trans
                                                              right join base_panel
                                                                         using (neighborhood_id, activity_date, num_of_bedrooms)
                                                         group by 1, 2, 3
                                                     ),
                                base_geo_sales_cum as (
                                                          select
                                                              neighborhood_id,
                                                              activity_date,
                                                              num_of_bedrooms,
                                                              sum(
                                                                      case
                                                                          when cumulative_units_sold_project is null
                                                                              then 0
                                                                          else cumulative_units_sold_project end
                                                              ) as cumulative_units_sold_neighborhood
                                                          from (
                                                                   select
                                                                       neighborhood_id,
                                                                       activity_date,
                                                                       num_of_bedrooms,
                                                                               sum(case
                                                                                       when units_sold_project is null
                                                                                           then 0
                                                                                       else units_sold_project end)
                                                                               over (partition by neighborhood_id order by activity_date rows between unbounded preceding and 1 preceding) as cumulative_units_sold_project
                                                                   from base_nearby_panel
                                                               ) as base_proj_launch_monthly
                                                          group by 1, 2, 3
                                                      )
                            select
                                neighborhood_id,
                                activity_date,
                                num_of_bedrooms,
                                cumulative_num_of_launched_units,
                                cumulative_units_sold_neighborhood,
                                cumulative_num_of_launched_units - cumulative_units_sold_neighborhood as num_of_remaining_units_neighborhood,
                                        avg(
                                        cumulative_num_of_launched_units - cumulative_units_sold_neighborhood) over
                                            (
                                            partition by neighborhood_id
                                            order by activity_date rows between 11 preceding and current row
                                            ) as rolling_num_of_available_units_neighborhood,
                                rolling_num_of_launched_projects_neighborhood,
                                rolling_num_of_launched_units_neighborhood
                            from base_geo_launch_cum
                                 left join base_geo_sales_cum
                                           using (neighborhood_id, activity_date, num_of_bedrooms)
                            order by neighborhood_id, activity_date
                        )
        ,
    base_launch_sales as (
                             select
                                 *
                             from (
                                      select
                                          project_dwid,
                                          num_of_bedrooms,
                                          num_of_units,
                                          count(*) as sales,
                                          avg(activity_psf) as price_psf
                                      from base_trans
                                           join base_unit_count
                                                using (project_dwid, num_of_bedrooms)
                                      where days_on_market <= 7
                                      group by 1, 2, 3
                                  ) as btbuc
                             union
                             (
                                 select
                                     'dcb010254f6a58a773b85bdb1160dcdc':: varchar,
                                     1:: varchar,
                                     23,
                                     0::bigint,
                                     2050::double precision
                             )
                             union
                             (
                                 select
                                     'dcb010254f6a58a773b85bdb1160dcdc':: varchar,
                                     2:: varchar,
                                     120,
                                     0::bigint,
                                     2050::double precision
                             )
                             union
                             (
                                 select
                                     'dcb010254f6a58a773b85bdb1160dcdc':: varchar,
                                     3:: varchar,
                                     70,
                                     0::bigint,
                                     2050::double precision
                             )
                             union
                             (
                                 select
                                     'dcb010254f6a58a773b85bdb1160dcdc':: varchar,
                                     4:: varchar,
                                     54,
                                     0::bigint,
                                     2050::double precision
                             )
                             union
                             (
                                 select
                                     '682a48988db71cb2860888075a431fa0':: varchar,
                                     2:: varchar,
                                     72,
                                     0::bigint,
                                     2050::double precision
                             )
                             union
                             (
                                 select
                                     '682a48988db71cb2860888075a431fa0':: varchar,
                                     3:: varchar,
                                     64,
                                     0::bigint,
                                     2050::double precision
                             )
                             union
                             (
                                 select
                                     '682a48988db71cb2860888075a431fa0':: varchar,
                                     4:: varchar,
                                     6,
                                     0::bigint,
                                     2050::double precision
                             )
                         ),
    base_static as (
                       select
                           *
                       from (
                                select
                                    project_dwid,
                                    project_display_name,
                                    case
                                        when project_display_name in ('Lentoria', 'The Hill @ One North')
                                            then to_date('2024-02-01', 'YYYY-MM-DD')
                                        else to_date(left(launch_date, 7), 'YYYY-MM')
                                        end as activity_date,
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
                                    land_size_sqft,
                                    neighborhood_id
                                from ui_app.project_summary_prod_sg
                                where property_group = 'condo'
                                  and property_type != 'ec'
                                  and (launch_date is not null or
                                       project_display_name in ('Lentoria', 'The Hill @ One North'))
                                  and project_display_name not ilike '% - Inactive%'
                            ) a
                            left join (
                                          select
                                              dw_project_id as project_dwid,
                                              project_zero_rm_percentage,
                                              project_one_rm_percentage,
                                              project_two_rm_percentage,
                                              project_three_rm_percentage,
                                              project_four_rm_percentage,
                                              project_five_rm_percentage,
                                              project_avg_size_of_zero_rm,
                                              project_avg_size_of_one_rm,
                                              project_avg_size_of_two_rm,
                                              project_avg_size_of_three_rm,
                                              project_avg_size_of_four_rm,
                                              project_avg_size_of_five_rm,
                                              region,
                                              zone,
                                              district
                                          from data_science.ui_master_sg_project_geo_view_filled_features_condo
                                      ) as b
                                      using (project_dwid)
                   ),
    base_geo as (
                    with
                        base_proj as (
                                         select
                                             *
                                         from ui_app.project_summary_prod_sg proj
                                         where property_group = 'condo'
                                           and property_type != 'executive condominium'
                                           --and launch_date is not null
                                           and project_display_name not ilike '% - Inactive%'
                                     ),
                        base_mrt as (
                                        select distinct
                                            left(poi_name, charindex(' (', poi_name) - 1) as poi_name,
                                            poi_long,
                                            poi_lat
                                        from masterdata_sg.poi
                                        where poi_subtype = 'mrt station'
                                          and poi_is_active is true
                                    ),
                        base_proj_mrt as (
                                             select distinct
                                                 project_dwid,
                                                 min(
                                                         ST_DistanceSphere(
                                                                 st_point(proj.longitude, proj.latitude),
                                                                 st_point(school.poi_long, school.poi_lat)
                                                         )
                                                 ) as meters_to_mrt,
                                                 count(distinct poi_name) as num_of_mrt
                                             from base_proj proj
                                                  join base_mrt school
                                                       on ST_DistanceSphere(
                                                                  st_point(proj.longitude, proj.latitude),
                                                                  st_point(school.poi_long, school.poi_lat)
                                                          ) <= 2000

                                             group by 1
                                         ),
                        base_bus as (
                                        select
                                            *
                                        from masterdata_sg.poi
                                        where poi_subtype = 'bus stop'
                                          and poi_is_active is true
                                    ),
                        base_proj_bus as (
                                             select distinct
                                                 project_dwid,
                                                 min(
                                                         ST_DistanceSphere(
                                                                 st_point(proj.longitude, proj.latitude),
                                                                 st_point(school.poi_long, school.poi_lat)
                                                         )
                                                 ) as meters_to_bus_stops,
                                                 count(poi_name) as num_of_bus_stop
                                             from base_proj proj
                                                  join base_bus school
                                                       on ST_DistanceSphere(
                                                                  st_point(proj.longitude, proj.latitude),
                                                                  st_point(school.poi_long, school.poi_lat)
                                                          ) <= 200
                                             group by 1
                                         ),
                        base_school as (
                                           select
                                               *
                                           from masterdata_sg.poi
                                           where good_school_indicator = 1
                                              or poi_name in (
                                                              'henry park primary school',
                                                              'ai tong school',
                                                              'tao nan school',
                                                              'raffles girls\' primary school'
                                               )
                                       ),
                        base_proj_school as (
                                                select distinct
                                                    project_dwid,
                                                    count(poi_name) as num_of_good_schools
                                                from base_proj proj
                                                     join base_school school
                                                          on ST_DistanceSphere(
                                                                     st_point(proj.longitude, proj.latitude),
                                                                     st_point(school.poi_long, school.poi_lat)
                                                             ) <= 2000
                                                group by 1
                                            )

                    select
                        project_dwid,
                            round(
                                    st_distancesphere(
                                            ST_point(longitude, latitude),
                                            ST_point(103.851463066212, 1.2839332623453799)
                                    )
                            ) / 1000 as km_to_sg_cbd,
                        case when num_of_bus_stop is null then 0 else num_of_bus_stop end as num_of_bus_stops,
                        meters_to_bus_stops,
                        case when num_of_mrt is null then 0 else num_of_mrt end as num_of_mrt,
                        meters_to_mrt,
                        case when num_of_good_schools is null then 0 else num_of_good_schools end as num_of_good_schools
                    from base_proj proj
                         left join base_proj_bus using (project_dwid)
                         left join base_proj_mrt using (project_dwid)
                         left join base_proj_school using (project_dwid)
                    order by launch_date desc
                )
select
    *
from base_static as p_base
     left join base_launch_sales using (project_dwid)
     left join base_geo using (project_dwid)
     left join base_market_info using (activity_date, neighborhood_id, num_of_bedrooms)
     left join (
                   select
                       id as neighborhood_id,
                       label as neighborhood
                   from ui_app.area_summary_prod_sg
                   where location_level = 'neighborhood'
               ) as nini using (neighborhood_id)
where activity_date >= to_date('2010-01-01', 'YYYY-MM-DD')
  and residential_unit_count != 0
order by activity_date desc, project_display_name, num_of_bedrooms desc;

select * from data_science.ui_master_sg_properties_view_filled_static_features_condo;

select * from ui_app.project_summary_prod_sg