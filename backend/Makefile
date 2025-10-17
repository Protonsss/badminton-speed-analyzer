.PHONY: help install dev test lint format clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev          - Run dev server"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean cache files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"

install:
	pip install -r requirements.txt

dev:
	python -m app.main

test:
	pytest

lint:
	flake8 app tests
	mypy app

format:
	black app tests
	isort app tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-logs:
	docker-compose logs -f api

docker-stop:
	docker-compose down

