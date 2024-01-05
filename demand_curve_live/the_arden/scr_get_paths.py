import os
from os.path import dirname, realpath

current_dir = dirname(realpath(__file__))
model_dir = current_dir + f'/output/model/'
figure_dir = current_dir + f'/output/figure/'
table_dir = current_dir + f'/output/table/'

for dir_path in [model_dir, figure_dir, table_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
