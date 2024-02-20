from constants.redshift import query_data

# -----------------------------------------------
# query data
with open(
        'classification_data.sql',
        'r'
) as sql_file:

    sql_script = sql_file.read()
    data = query_data(sql_script)

# comparable data
with open(
        'classification_comparable.sql',
        'r'
) as sql_file:
    sql_script = sql_file.read()
comparable_data = query_data(sql_script)

comp_features = ['sales', 'price_psf', 'residential_unit_count', 'meters_to_mrt']

comp_rename_dict = {
    i: "ref_" + i
    for i in comp_features
}
comp_rename_dict['project_display_name'] = 'ref_project'
# 'project_display_name': 'ref_project',

first_comp = comparable_data[comparable_data['rank'] == 1].copy()
first_comp = first_comp.merge(
    data[['project_display_name'] + comp_features].rename(columns=comp_rename_dict),
    on='ref_project'
)
data = data.merge(first_comp, on='project_display_name')

# -----------------------------------------------
# processing
data.loc[data['project_display_name'] == 'The Linq @ Beauty World', 'residential_unit_count'] = 120
data.loc[data['project_display_name'] == 'Royal Hallmark', 'residential_unit_count'] = 32

data['launch_year'] = data['activity_date'].str[:4].astype(int)
data['sales_rate'] = data['sales'] / data['residential_unit_count']
data['meters_to_mrt'] = data['meters_to_mrt'].fillna(3000)

for bed_text in [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five'
]:

    data[f'project_avg_size_of_{bed_text}_rm'] = data[f'project_avg_size_of_{bed_text}_rm'].replace(
        -1, 0
    )



