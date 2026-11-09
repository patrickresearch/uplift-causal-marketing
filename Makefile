.PHONY: setup test lint docs app

setup:
	uv sync
	uv run plotly_get_chrome -y

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

docs:
	uv run jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 notebooks/*.ipynb --output-dir docs

app:
	uv run streamlit run app/streamlit_app.py
