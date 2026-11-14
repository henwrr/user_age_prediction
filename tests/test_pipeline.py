import joblib
import numpy as np

from src.config import ID_COL, TARGET
from src.features import build_features
from src.pipeline import build_pipeline


def test_pipeline_fits_predicts_and_survives_pickle(tables, tmp_path):
    t = tables
    df = build_features(t['users'], t['visits'], t['ads'], t['surf'], t['device'], t['cloud'])
    X, y = df.drop(columns=[ID_COL, TARGET]), df[TARGET]

    pipeline = build_pipeline(X.columns).fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == len(X)
    assert set(pred) <= set(range(5))

    path = tmp_path / 'model.joblib'
    joblib.dump(pipeline, path)
    assert np.array_equal(joblib.load(path).predict(X), pred)


def test_pipeline_handles_unseen_category(tables):
    t = tables
    df = build_features(t['users'], t['visits'], t['ads'], t['surf'], t['device'], t['cloud'])
    X, y = df.drop(columns=[ID_COL, TARGET]), df[TARGET]
    pipeline = build_pipeline(X.columns).fit(X, y)

    X_new = X.head(3).copy()
    X_new['primary_device'] = 'умные часы'
    assert len(pipeline.predict(X_new)) == 3
