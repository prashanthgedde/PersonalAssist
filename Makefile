.PHONY: help lint typecheck test test-all format clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

typecheck: ## Run pyright type checker
	uv run pyright

test: ## Run unit tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v

test-all: lint typecheck test ## Run all checks (lint, typecheck, tests)

pre-commit: lint typecheck test ## Run all checks before commit
