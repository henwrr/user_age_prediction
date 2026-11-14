"""Предсказание возрастной категории для новых пользователей.

В папке ``--data-dir`` должны лежать CSV с теми же именами, что и обучающие.
Колонка ``age_category`` в таблице пользователей не нужна.

Пример:
    python -m src.predict --data-dir data/ --output predictions.csv
"""
import argparse
from pathlib import Path

import joblib

from .config import AGE_LABELS, ID_COL, MODEL_PATH, TARGET
from .data import load_data
from .features import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description='Предсказание возрастной категории')
    parser.add_argument('--data-dir', required=True, help='папка с CSV новых пользователей')
    parser.add_argument('--model-path', default=MODEL_PATH, type=Path)
    parser.add_argument('--output', default='predictions.csv', type=Path)
    args = parser.parse_args()

    pipeline = joblib.load(args.model_path)
    t = load_data(args.data_dir)
    users = t['users'].drop(columns=[TARGET], errors='ignore')

    df = build_features(users, t['visits'], t['ads'], t['surf'], t['device'], t['cloud'])
    predictions = df[[ID_COL]].copy()
    predictions[TARGET] = pipeline.predict(df.drop(columns=[ID_COL]))
    predictions['age_label'] = predictions[TARGET].map(dict(enumerate(AGE_LABELS)))

    predictions.to_csv(args.output, index=False)
    print(f'Предсказания для {len(predictions)} пользователей сохранены в {args.output}')


if __name__ == '__main__':
    main()
