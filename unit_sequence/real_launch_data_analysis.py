from os.path import dirname, realpath

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from unit_sequence.CoreRanker import RandomSelection
from constants.database import Database, get_hook

output_dir = dirname(realpath(__file__)) + f'/output/'

redshift = Database()
postgres = Database(get_hook('postgres'))

num_of_bedrooms_col = 'bedroom_count'
project_name_col = 'project_display_name'
stack_col = 'address_stack_num'
floor_col = 'address_floor_num'


def query_data(
    project_name
):
    avm_price_data = redshift.query_data(
        f"""
        with base_avm_price as (
                          select
                              dw_property_id,
                              unit_price_psf as avm_price_psf,
                              update_date,
                              row_number() over (partition by dw_property_id order by update_date desc) as seq
                          from data_science.master_daily_sale_valuation_sg_combined a
                               join data_science.ui_master_sg_properties_view_filled_static_features_condo
                                    using (dw_property_id)
                               join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                    using (dw_project_id)
                          where update_method = 'no_comparables_ml'
                            and property_group = 'condo'
                            --and to_date(update_date, 'YYYYMMDD') <= to_date(project_launch_month, 'YYYYMM')
                           and project_name = lower('{project_name.replace(" ", "-")}')
                      )
        select
            dw_property_id,
            avm_price_psf,
            update_date
        from base_avm_price
        where seq = 1
        order by 1, 2, update_date
        """
    )

    real_launch = postgres.query_data(
        f"""
        select
            project_display_name,
            property_dwid as dw_property_id,
            address_unit_text,
            floor_text:: int as address_floor_num,
            stack_text:: int as address_stack_num,
            bedroom_count,
            bathroom_count,
            sale_price as listing_price,
            sale_price/property_size_sqft as listing_price_psf,
            transacted_price as transacted_price,
            transacted_price/property_size_sqft as transacted_price_psf,
            sold_on,
            case when sold_on is not null then true else false end as sold_flag,
            row_number() over (partition by bedroom_count order by sold_on) as sequence
        from real_launch_sg.tenant_property_status
        inner join masterdata_sg.project p 
            using (project_dwid)
            where project_display_name = '{project_name}'
        order by bedroom_count, address_unit_text
        """
    )

    merged_data = real_launch.merge(avm_price_data, how='right', on='dw_property_id')
    merged_data['listing_price_psf'] = merged_data['listing_price_psf'].astype(float)
    # merged_data['transaction_time'] = pd.to_datetime(merged_data['sold_on'])

    merged_data = merged_data[merged_data['listing_price_psf'] != 0].copy()
    merged_data['price_diff'] = merged_data['avm_price_psf'] / merged_data['listing_price_psf'] - 1

    merged_data['pred_sequence'] = merged_data.groupby(num_of_bedrooms_col)['price_diff'].rank(
        ascending=False
    )
    merged_data['sales'] = merged_data.groupby(num_of_bedrooms_col)['sold_flag'].transform(
        lambda s: np.where(s == True, 1, 0).sum()
    )

    merged_data['pred_sold_flag'] = np.where(merged_data['pred_sequence'] <= merged_data['sales'], True, False)
    merged_data['correct_flag'] = np.where(
        merged_data['pred_sold_flag'] == merged_data['sold_flag'], True, False
    )

    return merged_data


def tower_view_comparison(project_trans):

    if project_trans.empty:
        return None

    project_name = project_trans[project_name_col].iloc[0]
    bed = project_trans[num_of_bedrooms_col].iloc[0]

    correct_rate = np.mean(
        np.where(
            project_trans['correct_flag'] == True, 1, 0
        )
    )

    q = project_trans['sales'].iloc[0]
    s = project_trans['address_unit_text'].count()
    q_percent = q / s

    random_score = RandomSelection().refer(q_percent)
    gain = pd.Series(correct_rate - random_score, name='gain').iloc[0]

    project_trans_sorted = project_trans.sort_values(by=stack_col).fillna(-100)

    fig, axs = plt.subplots(2, 3, figsize=(21, 14))

    for idx, col in enumerate(
            [
                'listing_price_psf',
                'avm_price_psf',
                'price_diff',
                'pred_sold_flag',
                'sold_flag',
                'correct_flag'
            ]
    ):

        ax = plt.subplot(2, 3, idx + 1)

        floor = project_trans_sorted[floor_col].astype(int)
        stack = project_trans_sorted[stack_col].astype(int)

        pivot_data = project_trans_sorted.pivot_table(
            index=floor,
            columns=stack,
            values=col,
            aggfunc='min'
        ).iloc[::-1]

        pivot_data.replace(True, 1, inplace=True)
        pivot_data.replace(False, 0, inplace=True)

        pivot_data[pivot_data == -100] = np.nan

        fmt = '.2f' if len(stack.unique()) < 10 else '.1f'
        cmap = sns.color_palette("coolwarm", as_cmap=True).copy()
        cmap.set_bad(color="lightgrey")
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap, annot=True,
            fmt=fmt if col not in ['sold_flag', 'pred_sold_flag', 'correct_flag'] else '.0f',
            annot_kws={"size": 8}, linewidth=0.5, linecolor='lightgrey'
        )

        if col == 'price_diff':
            ax_title = col + ' (avm_price_psf/listing_price_psf - 1)'
        else:
            ax_title = col

        ax.set_title(ax_title)
        ax.set_ylabel("Address Floor Number")
        ax.set_xlabel("Stack")

    title = f'flag heatmap {project_name} {bed}-bed'
    plt.suptitle(
        title +
        f'\n(correct rate {correct_rate * 100: .2f}%, gain {gain * 100: .2f}%)'
    )
    plt.savefig(output_dir + title + '.png', dpi=300)
    plt.close()

    return fig, axs


for project in [
    # 'The Arcady At Boon Keng',
    # 'Hillhaven',
    'Lumina Grand'
]:
    merged_data = query_data(project)

    cm = ConfusionMatrixDisplay.from_predictions(
        merged_data['sold_flag'],
        merged_data['pred_sold_flag'],
        cmap=plt.cm.Blues
    )
    title = f'confusion matrix {project}'
    plt.suptitle(title)
    plt.savefig(output_dir + title + '.png', dpi=300)

    for b in merged_data[num_of_bedrooms_col].unique():
        tower_view_comparison(merged_data[merged_data[num_of_bedrooms_col] == b])

print()
