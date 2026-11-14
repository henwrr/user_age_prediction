import numpy as np

from src.config import TARGET
from src.features import build_features


def _build(t, drop_target=False):
    users = t['users'].drop(columns=[TARGET]) if drop_target else t['users']
    return build_features(users, t['visits'], t['ads'], t['surf'], t['device'], t['cloud'])


def test_one_row_per_user_and_34_features(tables):
    df = _build(tables)
    assert df['user_id'].is_unique
    assert len(df) == tables['users']['user_id'].nunique()
    assert df.shape[1] == 34 + 2  # признаки + user_id + target


def test_shares_sum_to_one(tables):
    df = _build(tables)
    cat_cols = [c for c in df.columns if c.startswith('cat_share_')]
    day_cols = [c for c in df.columns if c.startswith('daytime_share_')]
    assert len(cat_cols) == 20 and len(day_cols) == 4
    assert np.allclose(df[cat_cols].sum(axis=1), 1)
    assert np.allclose(df[day_cols].sum(axis=1), 1)


def test_missing_categories_are_zero_columns(tables):
    # В синтетике нет Category 16–20, но колонки должны быть
    df = _build(tables)
    assert (df['cat_share_Category_20'] == 0).all()


def test_works_without_target(tables):
    df = _build(tables, drop_target=True)
    assert TARGET not in df.columns
    assert df.shape[1] == 34 + 1


def test_mode_daytime_matches_max_share(tables):
    df = _build(tables)
    day_cols = [c for c in df.columns if c.startswith('daytime_share_')]
    expected = df[day_cols].idxmax(axis=1).str.replace('daytime_share_', '')
    assert (df['mode_daytime'] == expected).all()
