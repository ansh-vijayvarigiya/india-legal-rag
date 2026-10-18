.PHONY: test test-all ingest eval inspect

test:
	python -m pytest -m "not slow"

test-all:
	python -m pytest

ingest:
	python -m legalrag.ingest --source rbi

eval:
	python -m legalrag.eval

inspect:
	python -m legalrag.inspector
