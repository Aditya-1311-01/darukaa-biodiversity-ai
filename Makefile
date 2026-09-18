VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: setup venv install env index test eval run lint clean

setup: venv install env index test   ## full first-time setup

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PY) -m pip install --upgrade pip --quiet

install: venv
	$(PIP) install -r requirements.txt

env:
	@test -f .env || (cp .env.example .env && echo "Created .env - add your Groq key")

index:
	$(PY) -m knowledge.ingest || echo "index build skipped; keyword fallback active"

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m eval.evaluate

lint:
	$(PY) -m ruff check app knowledge eval tests

run:
	$(VENV)/bin/uvicorn app.main:app --reload

clean:
	rm -rf $(VENV) .chroma .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
