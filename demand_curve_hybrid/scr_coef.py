from demand_curve_sep.cls_ds_partial_coef import FloorCoef, AreaCoef, TimeIndex, ZoneCoef

floor_coef = FloorCoef()
area_coef = AreaCoef()
time_index = TimeIndex()
zone_coef = ZoneCoef()


def query_adjust_coef(project_data):

    local_area_coef = area_coef.get_coef(project_data.floor_area_sqm.iloc[0] * 10.76)
    local_floor_coef = floor_coef.get_coef(max(project_data.proj_max_floor.iloc[0] // 2, 1))
    # local_zone_coef = zone_coef.get_coef(project_data.transaction_month.iloc[0].year, project_data.iloc[0].zone)
    # local_mrt_coef = mrt_coef.get_coef(project_data.meters_to_mrt.iloc[0])
    # coef_to_multiply = 1 / local_area_coef / local_floor_coef / local_zone_coef
    coef_to_multiply = 1 / local_area_coef / local_floor_coef
    # coef_to_multiply=1

    return coef_to_multiply


def query_time_rebase_index(launch_year_month):

    to_time_index = lambda ym: f'{ym.year}' + "{:02d}".format(ym.month)
    launch_year_month_index = to_time_index(launch_year_month)

    row = time_index.reference_table[
        time_index.reference_table['transaction_month_index'] == launch_year_month_index
        ]

    rebase_index = row['rebase_index'].iloc[0]
    current_index = row['current_index'].iloc[0]

    return 1 / current_index * rebase_index
