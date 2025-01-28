install-python:
	uv python install

install:
	uv venv
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml

indexing:
	uv run python src/indexing/embedding.py

retriver:
	uv run python src/retriever/hybrid_retriever.py

api:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

ui:
	uv run python -m streamlit run src/ui/app.py

all: 
	make indexing
	make retriver
	make start-api
	make start-ui

docker-start:
	docker-compose up --build