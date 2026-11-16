.PHONY: setup corpus serve test lint eval demo

setup:
	uv sync

corpus:            ## download the papers and build the index
	uv run plumbline fetch
	uv run plumbline ingest

serve:             ## dashboard at http://127.0.0.1:8000
	uv run plumbline serve

test:
	uv run pytest -q

lint:
	uv run ruff check .

eval:              ## run all evaluations and enforce the quality gate
	uv run plumbline eval --gate

demo: setup corpus serve
