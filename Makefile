.PHONY: test test-all ingest ingest-offline eval inspect judge

# The interpreter to run everything with. Point it at the virtualenv on Windows:
#   make eval PYTHON=.venv/Scripts/python.exe
# The default matters: Python 3.14 is also installed on the development machine,
# and this project requires 3.11.
PYTHON ?= python

test:
	$(PYTHON) -m pytest -m "not slow"

test-all:
	$(PYTHON) -m pytest

ingest:
	$(PYTHON) -m legalrag.ingest --source rbi

# Rebuild the corpus from the committed snapshots, with no network.
ingest-offline:
	$(PYTHON) -m legalrag.ingest --source rbi --snapshots data/raw/rbi

eval:
	$(PYTHON) -m legalrag.eval

inspect:
	$(PYTHON) -m legalrag.inspector

# Grades the generated answers and writes answers.md. Slow: two model calls per
# question, and the free tier meters tokens per minute. LIMIT=3 for a quick look:
#   make judge LIMIT=3
LIMIT ?=
judge:
	$(PYTHON) -m legalrag.judge $(if $(LIMIT),--limit $(LIMIT),)
