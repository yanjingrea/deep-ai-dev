with
    base_final_listing_price as (
                                    with
                                        base_trans as (

                                                          select
                                                              *,
                                                              min(transaction_date::date) over (partition by dw_project_id) as project_launch_date,
                                                                      row_number()
                                                                      over (partition by dw_property_id order by transaction_date desc) as seq
                                                          from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                                               join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                    using (dw_project_id)
                                                          where a.property_type_group = 'Condo'
                                                            and transaction_sub_type = 'new sale'
                                                      ),
                                        base_index as (
                                                          select
                                                              transaction_month_index,
                                                              hi_avg_improved
                                                          from data_science.ui_master_daily_sg_index_sale
                                                          where property_group = 'condo'
                                                            and index_area_type = 'country'
                                                      ),
                                        base_trans_price as (
                                                                select
                                                                    dw_project_id,
                                                                    dw_property_id,
                                                                    transaction_date,
                                                                    project_launch_date,
                                                                            unit_price_psf /
                                                                            rebase_index.hi_avg_improved *
                                                                            adjust_index.hi_avg_improved as transaction_price_psf,
                                                                            unit_price_psf /
                                                                            rebase_index.hi_avg_improved *
                                                                            adjust_index.hi_avg_improved as transaction_price,
                                                                    floor_area_sqft
                                                                from base_trans a
                                                                     join base_index as rebase_index
                                                                          using (transaction_month_index)
                                                                     join data_science.ui_master_sg_project_geo_view_filled_features_condo c
                                                                          using (dw_project_id)
                                                                     join base_index as adjust_index
                                                                          on c.project_launch_month =
                                                                             adjust_index.transaction_month_index
                                                                              and seq = 1
                                                                order by c.project_name, unit
                                                            ),
                                        base_listing_price as (
                                                                  select
                                                                      dw_project_id,
                                                                      property_dwid as dw_property_id,
                                                                      developer_price,
                                                                      row_number() over (partition by property_dwid order by update_date desc) as seq
                                                                  from raw_reference.sg_new_launch_developer_price a
                                                                       join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                            on a.project_dwid = b.dw_project_id
                                                                                and a.update_date <=
                                                                                    to_date(b.project_launch_month, 'YYYYMM')
                                                                                and developer_price is not null
                                                              )
                                    select
                                        dw_property_id,
                                        transaction_date,
                                        project_launch_date,
                                        case
                                            when developer_price is null
                                                then transaction_price_psf
                                            else developer_price / floor_area_sqft
                                            end as listing_price_psf,
                                        case
                                            when developer_price is null
                                                then transaction_price
                                            else developer_price
                                            end as listing_price
                                    from base_trans_price
                                         full outer join base_listing_price
                                                         using (dw_property_id)
                                    where project_launch_date >= '2015-01-01'
                                ),
    base_final_avm_price as (
                                with
                                    base_avm_price as (
                                                          select
                                                              dw_property_id,
                                                              unit_price_psf as daily_avm_price_psf,
                                                              update_date,
                                                              row_number() over (partition by dw_property_id order by update_date desc) as seq
                                                          from data_science.master_daily_sale_valuation_sg_combined a
                                                               join data_science.ui_master_sg_properties_view_filled_static_features_condo
                                                                    using (dw_property_id)
                                                               join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                    using (dw_project_id)
                                                          where update_method = 'no_comparables_ml'
                                                            and property_group = 'condo'
                                                            and (
                                                                      to_date(update_date, 'YYYYMMDD') <=
                                                                      to_date(project_launch_month, 'YYYYMM')
                                                                  or dw_project_id = 'f7fea87e1a59592e817cd84c2b416bb4'
                                                              )
                                                      ),
                                    base_his_avm_price as (
                                                              select
                                                                  dw_property_id,
                                                                  xgb_avg_pred_psf
                                                              from developer_tool.sg_condo_properties_estimate_launch_price
                                                          )
                                select
                                    dw_property_id,
                                    case
                                        when daily_avm_price_psf is null then xgb_avg_pred_psf
                                        else daily_avm_price_psf end as avm_price_psf,
                                    avm_price_psf * floor_area_sqft as avm_price,
                                    update_date
                                from data_science.ui_master_sg_properties_view_filled_static_features_condo
                                left join (
                                         select
                                             *
                                         from base_avm_price
                                         where seq = 1
                                     ) as avm_price
                                    using (dw_property_id)
                                left join base_his_avm_price
                                    using (dw_property_id)
                                order by 1, 2
                            ),
    geo_features as (
                        select
                            dw_property_id,
                            case when p.latitude is null then m.latitude else p.latitude end as filled_latitude,
                            case when p.longitude is null then m.longitude else p.longitude end as filled_longitude,
                            case
                                when b.km_to_sg_cbd is null then round(
                                                                         st_distancesphere(
                                                                                 ST_point(filled_latitude, filled_longitude),
                                                                                 ST_point(103.851463066212, 1.2839332623453799)
                                                                         )
                                                                 ) / 1000
                                else b.km_to_sg_cbd end as filled_km_to_sg_cbd,
                            num_of_bus_stops,
                            num_of_good_schools
                        from data_science.ui_master_sg_properties_view_filled_static_features_condo p
                             join data_science.ui_master_sg_building_view_filled_features_condo b
                                  using (dw_building_id)
                             join (
                                      select
                                          project_dwid as dw_project_id,
                                          project_display_name,
                                          latitude,
                                          longitude
                                      from masterdata_sg.address
                                           join masterdata_sg.project
                                                using (address_dwid)
                                  ) as m
                                  using (dw_project_id)
                    )
select distinct
    project_display_name as project_name,
    dw_project_id,
    dw_property_id,
    unit,
    address_stack,
    address_floor_num,
    top_floor,
    num_of_bedrooms,
    floor_area_sqft,
    proj_num_of_units,
    case
        when num_of_bedrooms = 0 then project_units_zero_rm
        when num_of_bedrooms = 1 then project_units_one_rm
        when num_of_bedrooms = 2 then project_units_two_rm
        when num_of_bedrooms = 3 then project_units_three_rm
        when num_of_bedrooms = 4 then project_units_four_rm
        when num_of_bedrooms = 5 then project_units_five_rm
        end
        as num_of_units,
    transaction_date,
    project_launch_date,
    datediff(day, project_launch_date::date, transaction_date::date) as days_on_market,
    listing_price_psf,
    avm_price_psf,
    listing_price,
    avm_price,
    filled_latitude as latitude,
    filled_longitude as longitude,
    filled_km_to_sg_cbd as km_to_sg_cbd,
    num_of_bus_stops,
    num_of_good_schools
from data_science.ui_master_sg_properties_view_filled_static_features_condo
     left join (
                   select
                       project_dwid as dw_project_id,
                       project_display_name
                   from ui_app.project_summary_prod_sg
               ) as c
               using (dw_project_id)
     join (select * from  data_science.ui_master_sg_project_geo_view_filled_features_condo where project_launch_month::int > 201800) d
               using (dw_project_id)
     left join base_final_listing_price a
               using (dw_property_id)
     left join base_final_avm_price b
               using (dw_property_id)
     left join geo_features
               using (dw_property_id)
order by 1, 2, address_stack, address_floor_num;