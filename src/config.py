"""Константы проекта."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 5

TARGET = 'age_category'
ID_COL = 'user_id'

AGE_LABELS = ['<18', '18-25', '26-40', '41-55', '56+']

S3_BASE_URL = 'https://code.s3.yandex.net/datasets/'

# Логическое имя таблицы -> имя файла
DATA_FILES = {
    'users': 'ds_s13_users.csv',
    'visits': 'ds_s13_visits.csv',
    'ads': 'ads_activity.csv',
    'surf': 'surf_depth.csv',
    'device': 'primary_device.csv',
    'cloud': 'cloud_usage.csv',
}

EXPECTED_CATEGORIES = [f'Category {i:02d}' for i in range(1, 21)]
EXPECTED_DAYTIMES = ['вечер', 'день', 'ночь', 'утро']

CAT_FEATURES = ['ads_activity', 'surf_depth', 'primary_device', 'mode_daytime']
BOOL_FEATURES = ['cloud_usage']

# Лучшие гиперпараметры по результатам GridSearchCV (см. ноутбук)
BEST_PARAMS = {'kernel': 'poly', 'C': 1.0, 'degree': 2}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'models' / 'age_classifier.joblib'
