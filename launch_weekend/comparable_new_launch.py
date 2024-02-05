import numpy as np
import plotly.graph_objects as go

from constants.redshift import query_data

data = query_data(
    f"""
    with
        base_launch_data as (
                                select
                                    dw_project_id,
                                    min(transaction_date) as launch_date
                                from (
                                         select
                                             *,
                                                     row_number()
                                                     over (partition by dw_property_id order by transaction_date desc) as seq
                                         from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                              join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                   using (dw_project_id)
                                         where a.property_type_group = 'Condo'
                                           and transaction_sub_type = 'new sale'
                                           and property_type != 'ec'
                                     ) as "*2"
                                where seq = 1
                                group by 1
                            ),
        base_sales as (
                          select
                              a.dw_project_id,
                              num_of_bedrooms,
                              (
                                  select non_landed_index
                                  from developer_tool.sg_gov_residential_index
                                  order by quarter_index desc
                                  limit 1
                              ) as current_index,
                              count(dw_property_id) as sales,
                              avg(unit_price_psf / c.non_landed_index * current_index) as average_launch_psf,
                              avg(transaction_amount / c.non_landed_index * current_index) as average_launch_price
                          from base_launch_data a
                               left join data_science.ui_master_sg_transactions_view_filled_features_condo b
                                         on a.dw_project_id = b.dw_project_id
                                             and b.transaction_date::date <= dateadd(day, 7, launch_date::date)
                               join developer_tool.sg_gov_residential_index c
                                    on b.transaction_quarter_index = c.quarter_index
                          group by 1, 2
                      )
            ,
        base as (
                    select
                        dw_project_id,
                        project_name,
                        project_launch_month,
                        num_of_bedrooms,
                        case
                            when num_of_bedrooms = 0 then project_units_zero_rm
                            when num_of_bedrooms = 1 then project_units_one_rm
                            when num_of_bedrooms = 2 then project_units_two_rm
                            when num_of_bedrooms = 3 then project_units_three_rm
                            when num_of_bedrooms = 4 then project_units_four_rm
                            when num_of_bedrooms = 5 then project_units_five_rm
                            end
                            as num_of_units,
                        sales,
                        average_launch_psf,
                        neighborhood
                    from data_science.ui_master_sg_project_geo_view_filled_features_condo
                         join base_sales
                              using (dw_project_id)
                    where left(project_launch_month, 4)::int >= 2015
                )
    select
        p_base.project_name as target_project_name,
        p_base.num_of_bedrooms,
        p_base.sales as target_sales,
        p_base.average_launch_psf as target_average_launch_psf,
        p_base.num_of_units as target_num_of_units,
        p_base.sales / p_base.num_of_units as target_sales_rate,
        c_base.project_name as ref_project_name,
        c_base.sales as ref_sales,
        c_base.average_launch_psf as ref_average_launch_psf,
        c_base.num_of_units as ref_num_of_units,
        c_base.sales / c_base.num_of_units as ref_sales_rate
    from ui_app.project_comparables_prod_sg p
         join base as p_base on p.project_dwid = p_base.dw_project_id
         join base as c_base on p.comparable_project_dwid = c_base.dw_project_id
    where p_base.project_launch_month >= c_base.project_launch_month
      and p_base.num_of_units > 0
      and c_base.num_of_units > 0
      and p_base.num_of_bedrooms = c_base.num_of_bedrooms
      and p_base.average_launch_psf / c_base.average_launch_psf - 1 <= 0.15
    order by 1, p.comparison_order
    """
)

for num_bed in np.arange(1, 6):

    node_traces_list = []

    temp_data = data[
        (data[f'target_num_of_units'] >= 50) &
        (data[f'ref_num_of_units'] >= 50) &
        (data['num_of_bedrooms'] == num_bed)
        ].copy()

    for mode in ['target', 'ref']:

        # data = data[
        #     (data[f'{mode}_project_stock'] >= 80) &
        #     (data[f'{mode}_project_sales_rate'] <= 1)
        #     ].copy()

        nodes_data = temp_data[[
            f'{mode}_project_name',
            f'num_of_bedrooms',
            f'{mode}_average_launch_psf',
            f'{mode}_sales',
            f'{mode}_num_of_units',
            f'{mode}_sales_rate'
        ]].drop_duplicates()

        hovertemplate = """
                    <b>%{hovertext}</b>
                        <br>
                            <br>project_name=%{customdata[0]}
                            <br>num_of_bedrooms=%{customdata[1]:.0f}
                            <br>mode_average_launch_psf=%{customdata[2]:.2f}
                            <br>mode_sales=%{customdata[3]:.0f}
                            <br>mode_num_of_units=%{customdata[4]:.0f}
                            <br>mode_sales_rate=%{customdata[5]:.2f}
                    <extra></extra>
                """.replace('mode', mode)

        node_trace = go.Scatter(
            customdata=nodes_data,
            x=nodes_data[f'{mode}_average_launch_psf'],
            y=nodes_data[f'{mode}_sales_rate'],
            mode='markers',
            hovertext=nodes_data[f'{mode}_project_name'],
            hovertemplate=hovertemplate,
            marker=dict(
                showscale=True,
                size=10,
                line_width=2
            )
        )

        node_traces_list += [node_trace]

    edge_x = []
    edge_y = []
    for idx, row in temp_data.iterrows():
        x0, y0 = row[f'target_average_launch_psf'], row[f'target_sales_rate']
        x1, y1 = row[f'ref_average_launch_psf'], row[f'ref_sales_rate']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        marker=dict(size=10, symbol="arrow-bar-up"),
        hoverinfo='none',
        mode='lines'
    )

    fig = go.Figure(
        data=node_traces_list + [edge_trace],
        layout=go.Layout(
            title=f'<br>New Launch Comparable {num_bed}-bedroom',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
            yaxis=dict(showgrid=True, zeroline=True, showticklabels=True)
        )
    )
    fig.show()

print()
