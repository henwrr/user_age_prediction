"""Сборка признакового пространства из сырых таблиц."""
import pandas as pd

from .config import EXPECTED_CATEGORIES, EXPECTED_DAYTIMES, TARGET


def _shares(visits: pd.DataFrame, column: str, expected: list[str], prefix: str) -> pd.DataFrame:
    """Доли визитов пользователя по значениям ``column`` с фиксированным набором столбцов."""
    counts = visits.groupby(['user_id', column]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=expected, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    shares.columns = [f'{prefix}{str(c).replace(" ", "_")}' for c in shares.columns]
    return shares


def build_features(users_df, visits_df, ads_df, surf_df, device_df, cloud_df) -> pd.DataFrame:
    """Объединяет все таблицы и создаёт признаки на уровне пользователя.

    Возвращает датафрейм с ``user_id``, 34 признаками и, если она есть
    в ``users_df``, целевой колонкой ``age_category``.
    """
    users_df = users_df.drop_duplicates()
    visits_df = visits_df.drop_duplicates()
    ads_df = ads_df.drop_duplicates()
    surf_df = surf_df.drop_duplicates()
    device_df = device_df.drop_duplicates()
    cloud_df = cloud_df.drop_duplicates()

    cat_share = _shares(visits_df, 'website_category', EXPECTED_CATEGORIES, 'cat_share_')
    daytime_share = _shares(visits_df, 'daytime', EXPECTED_DAYTIMES, 'daytime_share_')

    by_user = visits_df.groupby('user_id')
    activity = pd.DataFrame({
        'avg_sessions_per_day': (
            visits_df.groupby(['user_id', 'date'])['session_id'].nunique()
            .groupby('user_id').mean()
        ),
        'total_sessions': by_user['session_id'].nunique(),
        'active_days': by_user['date'].nunique(),
        'total_visits': by_user.size(),
        'unique_categories': by_user['website_category'].nunique(),
    })

    mode_daytime = (
        daytime_share.idxmax(axis=1)
        .str.replace('daytime_share_', '', regex=False)
        .rename('mode_daytime')
    )

    visit_features = cat_share.join(daytime_share).join(activity).join(mode_daytime)

    keep_cols = ['user_id'] + ([TARGET] if TARGET in users_df.columns else [])
    df = users_df[keep_cols].merge(visit_features, left_on='user_id', right_index=True, how='left')
    for extra in (ads_df, surf_df, device_df, cloud_df):
        df = df.merge(extra, on='user_id', how='left')

    return df.reset_index(drop=True)
