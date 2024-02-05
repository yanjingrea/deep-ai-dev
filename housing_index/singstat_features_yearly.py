import pandas as pd

from constants.redshift import upload_data
from housing_index.singstat import read_singstat_id

table_id_dict = {
    'unemployment_rate': 'M182332',
    'household_income': 'M810361',
    'population': 'M810001',
    'num_of_hdb_units': 'M400351',
    'num_of_hdb_units_sold': 'M400151',
}

return_columns_dict = {
    'unemployment_rate': {'resident_unemployment_rate,_(sa)': 'unemployment_rate'},
    'household_income': {
        'median_monthly_household_income_from_work_per_household_member_('
        'including_employer_cpf_contributions)':
            'monthly_household_income',
        'gini_coefficient_based_on_household_income_from_work_per_household_member_('
        'including_employer_cpf_contributions)_after_accounting_for_government_transfers_and_taxes':
            'gini_coefficient'
    },
    'population': {
        'resident_population': 'population',
        'rate_of_natural_increase': 'nature_increase_rate',
        'singapore_citizen_population': 'population_sc'
    },
    'num_of_hdb_units_sold': {
        'flats_constructed': 'constructed_hdb',
        "flats_sold_under_'home_ownership_scheme'": 'sold_hdb'
    },
}

yearly_data = pd.DataFrame()

for table_name, rename_dict in return_columns_dict.items():

    table_id = table_id_dict[table_name]

    queried_data = read_singstat_id(
        table_id,
        'year',
        return_columns=list(rename_dict.keys())
    )
    queried_data = queried_data.rename(columns=rename_dict)

    if yearly_data.empty:
        yearly_data = pd.concat([yearly_data, queried_data])

    else:
        merge_key = 'year'
        yearly_data = pd.merge(yearly_data, queried_data, on=merge_key, how='left')

upload_data(yearly_data, 'developer_tool.sg_gov_economic_feature_yearly')