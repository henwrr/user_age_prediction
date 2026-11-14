"""Загрузка исходных таблиц."""
from pathlib import Path

import pandas as pd

from .config import DATA_FILES, S3_BASE_URL


def load_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Загружает все таблицы проекта.

    Если указан ``data_dir``, CSV читаются оттуда, иначе — из публичного
    хранилища Яндекс Практикума.
    """
    tables = {}
    for name, filename in DATA_FILES.items():
        source = Path(data_dir) / filename if data_dir else S3_BASE_URL + filename
        tables[name] = pd.read_csv(source)
    return tables
