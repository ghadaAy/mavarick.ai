# Makefile

# Define Python and environment variables
PYTHON := python3.12
VENV := .venv
REQUIREMENTS := src/requirements.txt
FOLDER := src

activate-and-install:
	$(PYTHON) -m venv $(VENV)

	@echo "installing requirements"
	$(VENV)/bin/pip install -r $(REQUIREMENTS)


run-splitter:
	@echo "Running nlm-ingestor"
	docker compose -f $(FOLDER)/docker/docker_compose.yml up --build -d nlm-ingestor

run:
	@echo "Running dev"
	docker compose -f $(FOLDER)/docker/docker_compose.yml up --build -d
