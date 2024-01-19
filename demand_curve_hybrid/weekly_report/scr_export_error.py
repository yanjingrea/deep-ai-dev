import seaborn as sns

from demand_curve_hybrid.weekly_report.func_LaTex_code import *
from demand_curve_hybrid.weekly_report.scr_error_metrics import *

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

    report_path = f'{title.replace(" ", "_")}.png'
    plt.legend()
    plt.savefig(report_dir + report_path, dpi=300)

    distribution_cols[f'column{idx + 1}'] = create_column(
        col.replace('_', ' '),
        report_path,
        width=1
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

# distribution_summary
distribution_summary_cols = {}
for idx, col in enumerate(error_cols):
    report_path = plot_correct_rate(col)
    distribution_summary_cols[f'column{idx + 1}'] = create_column(
        col.replace('_', ' '),
        report_path,
        width=1
    )

error_codes = f"""
    {begin('frame')}{bracket('Error Distribution Summary Plot')}
        {begin('columns')}
        {distribution_summary_cols['column1']}
        {distribution_summary_cols['column2']}
        {end('columns')}
    {end('frame')}
    """

format_func = lambda x: '{:.2f}'.format(x * 100) + r'\%'

# project bedroom level error
projects_error = metrics_df.groupby(
    ['project_name', 'num_of_bedrooms'], as_index=False
)[error_cols].apply(lambda df: df[error_cols].abs().mean())

projects_error['num_of_bedrooms'] = projects_error['num_of_bedrooms'].apply(
    lambda a: int(a) if isinstance(a, float) else a
)

sep_table = projects_error[projects_error.num_of_bedrooms != 'all'].describe().copy()
agg_table = projects_error[projects_error.num_of_bedrooms == 'all'].describe().copy()

for ec in error_cols:
    projects_error[ec] = projects_error[ec].apply(format_func)

for idx in [0, 1]:
    for sum_table in [sep_table, agg_table]:
        sum_table.iloc[1:, idx] = sum_table.iloc[1:, idx].apply(format_func)

rename_dict = {
    i: i.replace('_', ' ')
    for i in projects_error.columns
}

projects_error = projects_error.rename(columns=rename_dict)
projects_error['page'] = projects_error.index // 10 + 1

# error summary
error_codes += r"""\subsection{Error Summary}"""
caption = r'\caption' + '{Error Metrics in percentage}'
for sum_table, page_name in zip(
        [agg_table, sep_table],
        ['Project Aggregate', 'Bedrooms Separate']
):
    error_codes += f"""
    {begin('frame')}{bracket(f'{page_name} Error Summary')}
        {begin('table')}
        {sum_table.rename(columns=rename_dict).set_axis(
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

    his_slide_cols = {}

    paths_dict = save_historical_data(sum_table, data_path)

    for idx, error_type in enumerate(['error to sales', 'error to stock']):

        his_slide_cols[f'column{idx + 1}'] = create_column(
            error_type.replace('_', ' '),
            paths_dict[error_type],
            width=1
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
