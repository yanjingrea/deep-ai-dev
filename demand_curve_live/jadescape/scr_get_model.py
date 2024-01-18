import pickle
from datetime import datetime
from demand_curve_live.jadescape.scr_get_paths import model_dir


project_name = 'Jadescape'
today = datetime.today().date()
# today = '2024-01-08'
models_path = model_dir + f'{project_name} {today}'.replace(' ', '_')
linear_models = pickle.load(open(models_path, 'rb'))

data_class_path = model_dir + f'dataclass {project_name} {today}'.replace(' ', '_')
training_data_class = pickle.load(open(data_class_path, 'rb'))

