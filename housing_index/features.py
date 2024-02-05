from constants.redshift import query_data

data = query_data(
    f"""
    with
        condo_new_sale as (
                              select
                                  transaction_quarter_index::int as quarter_index,
                                  avg(unit_price_psf) as avg_price_condo_new_sale,
                                  count(*) as trans_num_condo_new_sale
                              from data_science.ui_master_sg_transactions_view_filled_features_condo a
                              where property_type_group = 'Condo'
                                and transaction_sub_type = 'new sale'
                              group by 1
                          ),
        condo_resale as (
                            select
                                transaction_quarter_index::int as quarter_index,
                                avg(unit_price_psf) as avg_price_condo_resale,
                                count(*) as trans_num_condo_resale
                            from data_science.ui_master_sg_transactions_view_filled_features_condo a
                            where property_type_group = 'Condo'
                              and transaction_sub_type = 'resale'
                            group by 1
                        ),
        hdb_resale as (
                          select
                              transaction_quarter_index::int as quarter_index,
                              avg(unit_price_psf) as avg_price_hdb_resale,
                              count(*) as trans_num_hdb_resale
                          from data_science.ui_master_sg_transactions_view_filled_features_hdb a
                          where property_type_group = 'HDB'
                            and transaction_sub_type = 'resale'
                          group by 1
                      ),
        gov_index as (
                         select
                             to_date(
                                     concat(
                                             left(quarter_index, 4)::varchar,
                                             (right(quarter_index, 1)::int * 3 - 2)::varchar
                                     ),
                                     'YYYYMM'
                             ) as quarter_date,
                             *
                         from developer_tool.sg_gov_residential_index
                         order by quarter_index desc
                     )
    
    select
        *
    from condo_new_sale
         join condo_resale
              using (quarter_index)
         join hdb_resale
              using (quarter_index)
         join gov_index using (quarter_index)
         join developer_tool.sg_gov_economic_feature_quarterly
              using (quarter_index, quarter)
         join developer_tool.sg_gov_economic_feature_yearly
              using (year)
    order by quarter_index
    """
)

data['covid'] = data['year'].apply(lambda a: 1 if a >= 2020 else 0)
data['crisis'] = data['year'].apply(lambda a: 1 if 2008 <= a <= 2009 else 0)
data['sc_percentage'] = data['population_sc'] / data['population']
