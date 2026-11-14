"""Пайплайн предобработки и модели."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import BEST_PARAMS, BOOL_FEATURES, CAT_FEATURES, RANDOM_STATE


def build_preprocessor(feature_columns) -> ColumnTransformer:
    """ColumnTransformer: числовые — импутация нулём и масштабирование,
    категориальные — 'unknown' и OHE, бинарные — импутация False."""
    num_features = [c for c in feature_columns if c not in CAT_FEATURES + BOOL_FEATURES]

    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    boolean = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=False)),
    ])

    return ColumnTransformer([
        ('num', numeric, num_features),
        ('cat', categorical, CAT_FEATURES),
        ('bool', boolean, BOOL_FEATURES),
    ], remainder='drop')


def build_pipeline(feature_columns, **model_params) -> Pipeline:
    """Полный пайплайн: предобработка + SVC с лучшими гиперпараметрами."""
    params = {**BEST_PARAMS, **model_params}
    return Pipeline([
        ('preprocessor', build_preprocessor(feature_columns)),
        ('model', SVC(random_state=RANDOM_STATE, **params)),
    ])
