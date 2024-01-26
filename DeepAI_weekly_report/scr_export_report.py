import pickle
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# from demand_curve_condo.DeepAI_weekly_report.scr_test import *
from DeepAI_weekly_report.func_LaTex_code import *
from DeepAI_weekly_report.scr_get_paths import dev_res_dir, report_dir

test_results_des = dev_res_dir + 'test_results.plk'
test_results = pickle.load(open(test_results_des, 'rb'))

image_paths_des = dev_res_dir + 'image_paths.plk'
image_paths = pickle.load(open(image_paths_des, 'rb'))

paths_df = pd.DataFrame(image_paths)
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

metrics_df = test_results

metrics_df['error_to_sales'] = metrics_df['pred_sales'] / metrics_df['sales'] - 1
metrics_df['error_to_stock'] = (
        (metrics_df['pred_sales'] - metrics_df['sales']) / metrics_df['num_of_units']
)
metrics_df['period_label'] = metrics_df['launching_period'].apply(
    lambda a: 'first' if a <= 3 else 'rest'
)

# prediction error
error_cols = ['error_to_sales', 'error_to_stock']

# distribution
distribution_cols = {}
for idx, col in enumerate(error_cols):
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics_df.groupby('period_label').apply(
        lambda df: sns.histplot(
            df,
            x=col,
            stat='percent',
            label=df['period_label'].iloc[0] + ' launching period',
            color='skyblue' if df['period_label'].iloc[0] == 'first' else 'gold',
            alpha=0.6
        )
    )

    title = f"distribution plot of {col.replace('_', ' ')}"
    ax.set_title(title)

    report_path = f'{title}.png'
    plt.legend()
    plt.savefig(report_dir + report_path, dpi=300)

    distribution_cols[f'column{idx + 1}'] = create_column(
        col.replace('_', ' '),
        report_path,
        width=0.8
    )

formula = r"""
        \footnotesize
        Error Metrics \\
        $\text{error to sales} = (\text{predictive sales} - \text{actual sales}) / \text{actual sales}$ \\
        $\text{error to stock} = (\text{predictive sales} - \text{actual sales}) / \text{actual stock}$
        \vspace{1em}
    """
error_codes = r"""\subsection{Distribution}""" + f"""
    {begin('frame')}{bracket('Error Distribution Plot')}
        {formula}
        {begin('columns')}
        {distribution_cols['column1']}
        {distribution_cols['column2']}
        {end('columns')}
    {end('frame')}
    """

n_sample = metrics_df.groupby(['num_of_bedrooms'])['project_name'].count()


def calculate_correct_rate(metric: Literal['error_to_sales', 'error_to_stock']):
    correct_rate_data = pd.DataFrame()

    for confidence_interval in np.arange(0.02, 0.14, 0.04):

        n_correct = metrics_df[
            (metrics_df[metric] <= confidence_interval)
        ].groupby(['num_of_bedrooms'])['project_name'].count()

        correct_rate = (n_correct / n_sample).rename(f'error <= {confidence_interval:.2f}')

        if correct_rate_data.empty:
            correct_rate_data = correct_rate
        else:
            correct_rate_data = pd.concat([correct_rate_data, correct_rate], axis=1)
    return correct_rate_data


format_func = lambda x: '{:.2f}'.format(x * 100) + r'\%'
to_stock_correct_rate = calculate_correct_rate('error_to_stock')

for col in to_stock_correct_rate.columns:
    to_stock_correct_rate[col] = to_stock_correct_rate[col].apply(format_func)

to_stock_correct_rate.insert(
    0,
    column='num of bedrooms',
    value=np.where(
        to_stock_correct_rate.index == -1, 'all', to_stock_correct_rate.index.astype(int)
    ),
)

caption = r'\caption' + '{Error compared to total number of available units}'
to_stock_correct_rate.reset_index(drop=True, inplace=True)

error_codes += f"""
{begin('frame')}{bracket(f'Error Distribution Summary')}
    {begin('table')}
    {to_stock_correct_rate.to_latex(index=False)}
    {caption}
    {end('table')}
{end('frame')}
"""

# project bedroom level error
projects_error = metrics_df.groupby(
    ['project_name', 'num_of_bedrooms'], as_index=False
)[error_cols].apply(lambda df: df[error_cols].abs().mean())
projects_error['num_of_bedrooms'] = projects_error['num_of_bedrooms'].astype(int)

sep_table = projects_error[projects_error.num_of_bedrooms != -1].describe().copy()
agg_table = projects_error[projects_error.num_of_bedrooms == -1].describe().copy()

for ec in error_cols:
    projects_error[ec] = projects_error[ec].apply(format_func)

for idx in [1, 2]:
    for sum_table in [sep_table, agg_table]:
        sum_table.iloc[1:, idx] = sum_table.iloc[1:, idx].apply(format_func)

rename_dict = {
    i: i.replace('_', ' ')
    for i in projects_error.columns
}

projects_error = projects_error.rename(columns=rename_dict)
projects_error['num of bedrooms'] = np.where(
    projects_error['num of bedrooms'] == -1, 'all', projects_error['num of bedrooms']
)
projects_error['page'] = projects_error.index // 10 + 1

# error summary
error_codes += r"""\subsection{Summary}"""
caption = r'\caption' + '{Error Metrics in percentage}'
for sum_table, page_name in zip(
        [agg_table, sep_table],
        ['Project Aggregate', 'Bedrooms Separate']
):
    error_codes += f"""
    {begin('frame')}{bracket(f'{page_name} Error Summary')}
        {begin('table')}
        {sum_table[sum_table.columns[1:]].rename(columns=rename_dict).set_axis(
        ['count', 'mean', 'std', 'min', '25th', '50th', '75th', 'max']
    ).to_latex(index=True)}
        {caption}
        {end('table')}
    {end('frame')}
    """

# error comparison
error_codes += r"""\subsection{Historical Error Comparison}"""
for sum_table, page_name, data_path in zip(
        [agg_table, sep_table],
        ['Project Aggregate', 'Bedrooms Separate'],
        ['agg', 'sep']
):
    his_path = (
        f'/Users/wuyanjing/PycharmProjects/app/demand_curve_sep'
        f'/weekly_report/output/{data_path}_historical_error.csv'
    )
    his_data = pd.read_csv(his_path, header=0)
    # his_data['report date'] = pd.to_datetime(his_data['report date'], dayfirst=True).dropna()
    # his_data['report date'] = pd.to_datetime(his_data['report date']).dropna()

    report_date = f'{pd.to_datetime(td).date()}'
    to_record_row = [report_date]
    for pct in ['mean', '25%', '50%', '75%']:
        to_record_row += sum_table.loc[pct][1:].apply(lambda s: s.replace(r'\%', '%')).reset_index(
            drop=True
        ).to_list()

    if his_data['report date'].iloc[-1] == report_date:
        his_data.iloc[-1, :] = to_record_row
    else:
        his_data.loc[-1] = to_record_row
    his_data.reset_index(inplace=True, drop=True)

    if td.isoweekday() == 5:
        his_data.to_csv(his_path, index=False)

    his_slide_cols = {}
    x = his_data['report date']
    for idx, error_type in enumerate(['error to sales', 'error to stock']):

        fig, ax = plt.subplots(figsize=(8, 6))

        for pct, l in zip(['', ' 50th'], ['mean', 'median']):

            col = error_type + pct

            y = his_data[col].str[:-1].astype(float)

            sns.lineplot(x=x, y=y, marker='o', label=l)

            for i in his_data.index:
                ax.text(x=x[i], y=y[i], s=his_data[col][i])

        ax.set_xticks(x)
        title = f"historical error of {error_type}"
        ax.set_title(title)

        report_path = f'{data_path} {title}.png'
        plt.legend()
        plt.savefig(report_dir + report_path, dpi=300)

        his_slide_cols[f'column{idx + 1}'] = create_column(
            error_type.replace('_', ' '),
            report_path,
            width=0.8
        )

    error_codes += f"""
        {begin('frame')}{bracket(f'{page_name} Historical Error Comparison')}
            {begin('columns')}
            {his_slide_cols['column1']}
            {his_slide_cols['column2']}
            {end('columns')}
        {end('frame')}
        """

# project's error
error_codes += r"""\subsection{Project's Average Error Analysis}"""
for p in projects_error.page.unique():
    page_content = projects_error[projects_error['page'] == p][rename_dict.values()]
    c = f"""
    {begin('frame')}{bracket('Table')}
        {begin('table')}
        {page_content.to_latex(index=False)}
        {caption}
        {end('table')}
    {end('frame')}
    """
    error_codes += c

latex_dir = '/Users/wuyanjing/PycharmProjects/presentation/src/'
file_name = f'weekly_report_template_{td}.tex'

with open(f"{latex_dir}{file_name}", "w") as file:

    latex_content = r"""
    %----------------------------------------------------------------------------------------
    %	PACKAGES AND THEMES
    %----------------------------------------------------------------------------------------
    \documentclass[aspectratio=169,xcolor=dvipsnames, t]{beamer}
    \usepackage{fontspec} % Allows using custom font. MUST be before loading the theme!
    \usetheme{SimplePlusREA}
    \usepackage{hyperref}
    \usepackage{graphicx} % Allows including images
    \usepackage{booktabs} % Allows the use of \toprule, \midrule and  \bottomrule in tables
    \usepackage{svg} %allows using svg figures
    \usepackage{tikz}
    \usepackage{makecell}
    \usepackage{wrapfig}
    % ADD YOUR PACKAGES BELOW

    %----------------------------------------------------------------------------------------
    %	TITLE PAGE CONFIGURATION
    %----------------------------------------------------------------------------------------

    \title[short title]{Weekly Report}
    \subtitle{New Launch Condo Demand Curve}

    \author[Surname]{Data Science}
    \institute[Real Estate Analytics]{Real Estate Analytics}
    % Your institution as it will appear on the bottom of every slide, maybe shorthand to save space


    \date{\today} % Date, can be changed to a custom date
    %----------------------------------------------------------------------------------------
    %	PRESENTATION SLIDES
    %----------------------------------------------------------------------------------------
    \begin{document}
        
        \maketitlepage
        \begin{frame}[t]{Overview} 
            \tableofcontents
        \end{frame}

        \makesection{Prediction Error}
        %----------------------------------------------------------------------------------------
        % Prediction Error
        %----------------------------------------------------------------------------------------
        """ + error_codes + r"""

        \makesection{Demand Curve}
        %----------------------------------------------------------------------------------------
        % Demand Curve
        %----------------------------------------------------------------------------------------
        """ + demand_images_codes + r""" 
        
        \makesection{Disclaimer}
        %----------------------------------------------------------------------------------------
        % Disclaimer Page
        %----------------------------------------------------------------------------------------
        \begin{frame}[fragile]
            \frametitle{Disclaimer}
            \footnotesize
            This report and all information and documents in any oral, 
            hardcopy, or electronic form prepared specifically for it (collectively, 'Information') have been 
            prepared by, or on behalf of, Real Estate Analytics. It is provided in summary form and Real Estate Analytics does not warrant or represent 
            that the Information is accurate, current, or complete. The Information is general information only and 
            is not, nor is intended to be, professional or legal advice to a user. Users requiring information beyond 
            that of a general nature in relation to, in connection with, or referred to in the Information, 
            should seek independent professional advice relevant to their own particular circumstances. The 
            Information may include views or recommendations of third parties and does not necessarily reflect the 
            views of Real Estate Analytics or indicate a commitment to a particular course of action. Real Estate 
            Analytics is not liable or responsible to any person for any injury, loss, or damage of any nature 
            whatsoever arising from or incurred by the use of, reliance on, or interpretation of the Information. Any 
            unauthorized use of the Information is strictly prohibited. Users are not authorized to copy, circulate, 
            disclose, disseminate, or distribute the Information, in whole or in part, to any third party, 
            unless explicitly agreed upon by Real Estate Analytics.
        \end{frame}

        %----------------------------------------------------------------------------------------
        % Final PAGE
        %----------------------------------------------------------------------------------------
        \finalpagetext{Thank you for your attention}
        \makefinalpage
    \end{document}
    """

    print(latex_content)

    file.write(
        latex_content
    )

print()
