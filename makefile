# Makefile

# Define Python and environment variables
PYTHON := python3.12
VENV := .venv
REQUIREMENTS := mixed_rag/requirements.txt
FOLDER := mixed_rag

activate-and-install:
	$(PYTHON) -m venv $(VENV)

	@echo "installing requirements"

	$(VENV)/bin/pip install -r $(REQUIREMENTS)

run-pgvector:
	@echo "Launching pgvector"
	docker compose -f $(FOLDER)/docker/docker_compose.yml up --build -d pgvector

run-splitter:
	@echo "Running nlm-ingestor"
	docker compose -f $(FOLDER)/docker/docker_compose.yml up --build -d nlm-ingestor

run-dev:
	@echo "Running dev"
	docker compose -f $(FOLDER)/docker/docker_compose.yml up --build -d
