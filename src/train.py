"""Обучение финальной модели.

Пример:
    python -m src.train                    # данные из хранилища Практикума
    python -m src.train --data-dir data/   # локальные CSV
"""
import argparse
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import AGE_LABELS, ID_COL, MODEL_PATH, N_SPLITS, RANDOM_STATE, TARGET, TEST_SIZE
from .data import load_data
from .features import build_features
from .pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Обучение классификатора возраста')
    parser.add_argument('--data-dir', help='папка с CSV (по умолчанию — S3 Практикума)')
    parser.add_argument('--model-path', default=MODEL_PATH, type=Path,
                        help='куда сохранить модель')
    args = parser.parse_args()

    t = load_data(args.data_dir)
    df = build_features(t['users'], t['visits'], t['ads'], t['surf'], t['device'], t['cloud'])
    print(f'Признаковое пространство: {df.shape[0]} пользователей, {df.shape[1] - 2} признаков')

    X = df.drop(columns=[ID_COL, TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(X.columns)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f'F1-macro CV:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f'F1-macro test: {f1_score(y_test, y_pred, average="macro"):.4f}\n')
    print(classification_report(y_test, y_pred, target_names=AGE_LABELS))

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f'Модель сохранена: {args.model_path}')


if __name__ == '__main__':
    main()
