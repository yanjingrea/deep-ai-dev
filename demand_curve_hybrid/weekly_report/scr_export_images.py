import pickle

import pandas as pd

from demand_curve_hybrid.weekly_report.func_LaTex_code import *
from demand_curve_hybrid.weekly_report.scr_get_paths import dev_res_dir, report_dir

test_results_des = dev_res_dir + 'test_results.plk'
test_results = pickle.load(open(test_results_des, 'rb'))

image_paths_des = dev_res_dir + 'paths_df.plk'
paths_df = pickle.load(open(image_paths_des, 'rb'))

paths_df['page'] = paths_df.index // 2

# demand curve image
demand_images_codes = f''
for p in paths_df.page.unique():
    page_content = paths_df[paths_df['page'] == p]

    c = page_content.apply(
        lambda row: create_column(
            f'{int(row.num_of_bedrooms) if row.num_of_bedrooms != "all" else row.num_of_bedrooms} bedrooms',
            row.paths
        ), axis=1
    )

    demand_images_codes += create_page(
        page_title=', '.join(page_content['project_name'].unique()),
        column1=c.iloc[0],
        column2=c.iloc[1] if len(c) > 1 else ''
    )