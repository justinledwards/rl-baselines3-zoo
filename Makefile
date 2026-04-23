LINT_PATHS = *.py tests/ scripts/ rl_zoo3/ hyperparams/python/*.py docs/conf.py

# Run pytest and coverage report
pytest:
	mise run test

# check all trained agents (slow)
check-trained-agents:
	python -m pytest -v tests/test_enjoy.py -k trained_agent --color=yes

mypy:
	mise run type

type: mypy

lint:
	mise run lint

format:
	mise run format

check-codestyle:
	mise run check-codestyle

commit-checks: format type lint

doc:
	mise run doc

spelling:
	cd docs && make spelling

clean:
	mise run clean

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	mise run build
	uv run twine upload dist/*

# Test PyPi package release
test-release:
	mise run build
	uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: lint format check-codestyle commit-checks doc spelling docker type pytest
