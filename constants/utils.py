import os
from os.path import dirname, realpath

OUTPUT_DIR = dirname(realpath(__file__)) + os.sep + 'local' + os.sep


# ---------------------------------------------------------------------------
def set_plot_format(plt):
    # -----------------------------------------------------
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.constrained_layout.use'] = True
    # ------------------------------------------------------


# ---------------------------------------------------------------------------
GREENS = ['#4F6962', '#B5C583', '#F3DF9B', '#CfB79D']

NatureD = {
    'purple': '#403990',
    'blue': '#2b6a99',
    'yellow': '#FBDD85',
    'orange': '#F46F43',
    'red': '#CF3D3E'
}
NatureL = {
    'yellow': '#FFEBAD',
    'grey': '#DCD7C1',
    'purple': '#BFB1D0',
    'skyblue': '#A7C0DE',
    'blue': '#6c91c2',
    'red': '#A4514F'
}

def print_in_green_bg(text):
    print('\x1b[6;30;42m' + text + '\x1b[0m')