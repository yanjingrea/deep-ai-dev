import os
from datetime import datetime
from os.path import dirname, realpath

# -------------------------------------------------

td = datetime.today()

dev_dir = dirname(realpath(__file__)) + f'/output/dev/{td.date()}/'
dev_figure_dir = dev_dir + f'figures/'
dev_data_dir = dev_dir + f'data/'
dev_res_dir = dev_dir + f'res/'

report_dir = f'/Users/wuyanjing/PycharmProjects/presentation/src/images/{td.date()}/'

for mode, directory_path in zip(
        ['dev_figures', 'dev_table', 'report', 'dev_results'],
        [dev_figure_dir, dev_data_dir, report_dir, dev_res_dir]
):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"{mode} directory created: {directory_path}")
        except OSError as error:
            print(f"Error creating {mode} directory: {error}")
    else:
        print(f"locate {mode} directory: {directory_path}")

u_curve_dir = report_dir + f'project_level_u_curve.png'
