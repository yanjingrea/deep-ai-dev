with
    base as (
                select
                    project_dwid,
                    project_display_name,
                    case
                        when project_display_name in ('Lentoria', 'The Hill @ One North')
                            then '2024-02-01'
                        else launch_date
                        end as launch_date,
                    latitude,
                    longitude,
                    average_launch_psf,
                    residential_unit_count
                from ui_app.project_summary_prod_sg
                where property_group = 'condo'
                  and property_type != 'ec'
                  and (launch_date >= '2015-01-01' or project_display_name in ('Lentoria', 'The Hill @ One North'))
                  and residential_unit_count != 0
            )
select
    p_base.project_display_name,
    c_base.project_display_name as ref_project,
    ST_DistanceSphere(
            st_point(p_base.longitude, p_base.latitude),
            st_point(c_base.longitude, c_base.latitude)
    ) as distance,
    datediff(day, c_base.launch_date::date, p_base.launch_date::date) as days_gap,
            dense_rank()
            over (partition by p_base.project_display_name order by days_gap) as rank,
    count(*) over (partition by p_base.project_display_name) as num_of_comparables
from base as p_base
     left join base as c_base
               on ST_DistanceSphere(
                          st_point(p_base.longitude, p_base.latitude),
                          st_point(c_base.longitude, c_base.latitude)
                  ) <= 3000
                   and datediff(day, c_base.launch_date::date, p_base.launch_date::date) between 0 and 365 * 3
--                    and abs(p_base.average_launch_psf / c_base.average_launch_psf - 1) <= 0.15
                   and abs(c_base.residential_unit_count / p_base.residential_unit_count - 1) <= 0.5
                   and p_base.project_dwid != c_base.project_dwid
order by p_base.launch_date desc