import pandas as pd
from housing_index.singstat import read_singstat_id
from constants.redshift import upload_data

resource_ids = ['M212161', 'M212261']
frequency = 'quarter'

private_data = read_singstat_id('M212261', frequency)
hdb_data = read_singstat_id('M212161', frequency)

columns = [
    'residential_properties',
    'landed',
    'non-landed'
]

private_data.rename(
    columns={
        i: i.replace('-', '_') + '_index'
        for i in columns
    },
    inplace=True
)

hdb_data.rename(
    columns={'total': 'hdb_resale_index'},
    inplace=True
)

queried_data = pd.merge(
    private_data,
    hdb_data,
    on=[frequency, 'quarter_index'],
    how='outer'
)

upload_data(
    queried_data,
    'developer_tool.sg_gov_residential_index'
)

