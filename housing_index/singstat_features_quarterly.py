import pandas as pd

from constants.redshift import upload_data
from housing_index.singstat import read_singstat_id

table_id_dict = {
    'gdp': 'M015941',
    'cpi': 'M213151',
    'num_of_vacant_properties': 'M400841',
    'num_of_developing_lands': 'M400391',
}

return_columns_dict = {
    'gdp': {
        'gdp_in_chained_(2015)_dollars': 'gdp',
        'accommodation_&_food_services,_real_estate,'
        '_administrative_&_support_services_and_other_services_industries':
            'housing_gdp'
    },
    'cpi': {'all_items': 'cpi'},
    'num_of_vacant_properties': {
        'available_non-landed_properties': 'available_condo',
        'vacant_non-landed_properties': 'vacant_condo'
    },
    'num_of_developing_lands': {'total_non-landed_properties': 'developing_condo'}
}

quarterly_data = pd.DataFrame()

for table_name, rename_dict in return_columns_dict.items():

    table_id = table_id_dict[table_name]

    queried_data = read_singstat_id(
        table_id,
        'quarter',
        return_columns=list(rename_dict.keys())
    )
    queried_data = queried_data.rename(columns=rename_dict)

    if quarterly_data.empty:
        quarterly_data = queried_data

    else:
        merge_key = 'quarter_index'
        if 'quarter_index' in queried_data.columns:
            queried_data['quarter_index'] = queried_data['quarter_index'].astype(int)

        # outer join might change the dtype of quarter ts_housing_index_forecasting from int to float
        quarterly_data = pd.merge(quarterly_data, queried_data, on=merge_key, how='left')

quarterly_data.insert(0, 'quarter', quarterly_data['quarter_index'].astype(str).apply(lambda a: f'{a[:4]} {a[-1]}Q'))
quarterly_data.insert(2, 'year', quarterly_data['quarter_index'].astype(str).apply(lambda a: int(a[:4])))

upload_data(quarterly_data, 'developer_tool.sg_gov_economic_feature_quarterly')
