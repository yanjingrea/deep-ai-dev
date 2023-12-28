from demand_curve_sep.weekly_report.scr_test import *
from demand_curve_sep.weekly_report.func_LaTex_code import *

file_3d_name = f'weekly_report_3d_{td}.py'

if False:
    with open(f"{dev_3d_dir}{file_3d_name}", "w") as file:
        paths_list = ',\n'.join(
            [
                i.__repr__() for i in demand_curve_model.image_3d_paths
            ]
        )

        loop_content = """
        for p in paths_list[:5]:
            figx = pickle.load(open(f'{dev_3d_dir}{p}', 'rb'))
            figx.show()
        """

        pickle_load_image = lambda path: f"pickle.load(open({path}, 'rb'))"

        py_content = f"""
        import pickle
        dev_3d_dir = '{dev_3d_dir}'
        paths_list = [{paths_list}]
        {loop_content}
        """

        file.write(
            py_content
        )

paths_df = pd.DataFrame(image_paths)
paths_df['page'] = paths_df.index // 2

# demand curve image
demand_images_codes = f''
for p in paths_df.page.unique():
    page_content = paths_df[paths_df['page'] == p]

    c = page_content.apply(
        lambda row: create_column(
            f'{int(row.num_of_bedrooms)} bedrooms',
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
    sns.set(style="darkgrid")
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
    {begin('frame')}{bracket('Error Distribution')}
        {formula}
        {begin('columns')}
        {distribution_cols['column1']}
        {distribution_cols['column2']}
        {end('columns')}
    {end('frame')}
    """

# project bedroom level error
projects_error = metrics_df.groupby(
    ['project_name', 'num_of_bedrooms'], as_index=False
)[error_cols].apply(lambda df: df[error_cols].abs().mean())
projects_error['num_of_bedrooms'] = projects_error['num_of_bedrooms'].astype(int)

summary_table = projects_error.describe().copy()
format_func = lambda x: '{:.2f}'.format(x * 100) + r'\%'
for ec in error_cols:
    projects_error[ec] = projects_error[ec].apply(format_func)

for idx in [1, 2]:
    summary_table.iloc[1:, idx] = summary_table.iloc[1:, idx].apply(format_func)

rename_dict = {
    i: i.replace('_', ' ')
    for i in projects_error.columns
}

projects_error = projects_error.rename(columns=rename_dict)

projects_error['page'] = projects_error.index // 10 + 1

# project error
error_codes += r"""\subsection{Project's Average Error Analysis}"""
for p in projects_error.page.unique():
    page_content = projects_error[projects_error['page'] == p][rename_dict.values()]
    caption = r'\caption' + '{Error Metrics in percentage}'
    c = f"""
    {begin('frame')}{bracket('Table')}
        {begin('table')}
        {page_content.to_latex(index=False)}
        {caption}
        {end('table')}
    {end('frame')}
    """
    error_codes += c

# error summary
error_codes += r"""\subsection{Summary}"""
error_codes += f"""
{begin('frame')}{bracket('Summary')}
        {begin('table')}
        {summary_table.rename(columns=rename_dict).set_axis(
    ['count', 'mean', 'std', 'min', '25th', '50th', '75th', 'max']
).to_latex(index=True)}
        {caption}
        {end('table')}
    {end('frame')}
"""

# error comparison
his_path = '/Users/wuyanjing/PycharmProjects/app/demand_curve_sep/weekly_report/output/historical_error.csv'
his_data = pd.read_csv(his_path, header=0)
his_data['report date'] = pd.to_datetime(his_data['report date'], dayfirst=True)


to_record_row = [pd.to_datetime(td)]
for pct in ['mean', '25%', '75%']:
    to_record_row += summary_table.loc[pct][1:].apply(lambda s: s.replace(r'\%', '%')).reset_index(
        drop=True
    ).to_list()
his_data.loc[-1] = to_record_row
his_data.reset_index(inplace=True)

his_slide_cols = {}
x = his_data['report date']
for idx, error_type in enumerate(['error to sales', 'error to stock']):

    fig, ax = plt.subplots(figsize=(8, 6))

    for pct in ['', ' 25th', ' 75th']:

        col = error_type + pct

        y = his_data[col].str[:-1].astype(float)

        sns.lineplot(x=x, y=y, marker='o', label=col)

        for i in his_data.index:
            ax.text(x[i], y[i], his_data[col][i])

    ax.set_xticks(x)
    title = f"historical error of {error_type}"
    ax.set_title(title)

    report_path = f'{title}.png'
    plt.legend()
    plt.savefig(report_dir + report_path, dpi=300)

    his_slide_cols[f'column{idx + 1}'] = create_column(
        error_type.replace('_', ' '),
        report_path,
        width=0.8
    )

error_codes += r"""\subsection{Historical Error Comparison}""" + f"""
    {begin('frame')}{bracket('Historical Error')}
        {begin('columns')}
        {his_slide_cols['column1']}
        {his_slide_cols['column2']}
        {end('columns')}
    {end('frame')}
    """

print(demand_images_codes)
print(error_codes)

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
        \begin{frame}{Overview} 
            \tableofcontents
        \end{frame}

        \makesection{Demand Curve}
        %----------------------------------------------------------------------------------------
        % Demand Curve
        %----------------------------------------------------------------------------------------
        """ + demand_images_codes + r"""

        \makesection{Prediction Error}
        %----------------------------------------------------------------------------------------
        % Prediction Error
        %----------------------------------------------------------------------------------------
        """ + error_codes + r"""
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

    if False:
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
        
        
        \makesection{Demand Curve}
        """ + f"""
        %----------------------------------------------------------------------------------------
        %Demand Curve
        %----------------------------------------------------------------------------------------
        \n
        {demand_images_codes}
        """ + r"""
        \makesection{Prediction Error}
        """ + f"""
        %----------------------------------------------------------------------------------------
        %Prediction Error
        %----------------------------------------------------------------------------------------
        \n
        {error_codes}
        """ + r"""
        \makesection{Disclaimer}
        %----------------------------------------------------------------------------------------
        % Disclaimer Page
        %----------------------------------------------------------------------------------------
        \begin{frame}[fragile] % Need to use the fragile option when verbatim is used in the slide
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
        %----------------------------------------------------------------------------------------
        \end{document}
        """

    print(latex_content)

    file.write(
        latex_content
    )

print()
