.PHONY: install train predict test

install:
	pip install -r requirements.txt

train:
	python -m src.train

predict:
	python -m src.predict --data-dir data/ --output predictions.csv

test:
	pytest -q
