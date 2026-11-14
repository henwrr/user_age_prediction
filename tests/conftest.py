import numpy as np
import pandas as pd
import pytest

from src.config import EXPECTED_CATEGORIES, EXPECTED_DAYTIMES


def make_tables(n_users: int = 60, n_visits: int = 3000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Синтетические таблицы той же структуры, что и в реальных данных."""
    rng = np.random.default_rng(seed)
    user_ids = [f'user_{i:03d}' for i in range(n_users)]
    users = pd.DataFrame({'user_id': user_ids, 'age_category': np.arange(n_users) % 5})

    visits = pd.DataFrame({
        'date': rng.choice(pd.date_range('2025-11-01', periods=14).strftime('%Y-%m-%d'), n_visits),
        'daytime': rng.choice(EXPECTED_DAYTIMES, n_visits),
        'session_id': [f's{i}' for i in rng.integers(0, n_visits // 2, n_visits)],
        'user_id': rng.choice(user_ids, n_visits),
        'website_category': rng.choice(EXPECTED_CATEGORIES[:15], n_visits),
    })

    def side(col, values, frac=0.9):
        ids = rng.choice(user_ids, int(n_users * frac), replace=False)
        return pd.DataFrame({'user_id': ids, col: rng.choice(values, len(ids))})

    return {
        'users': pd.concat([users, users.head(3)]),  # дубликаты
        'visits': visits,
        'ads': side('ads_activity', ['очень редко', 'редко', 'умеренно', 'часто', 'очень часто']),
        'surf': side('surf_depth', ['поверхностно', 'средне', 'глубоко']),
        'device': side('primary_device', ['смартфон', 'ПК', 'ноутбук', 'планшет']),
        'cloud': side('cloud_usage', [True, False]),
    }


@pytest.fixture
def tables():
    return make_tables()
