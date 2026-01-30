# Format
python -m black .
python -m isort .

# Check
python -m pylint --rcfile=pyproject.toml mace_jax tests scripts
python -m mypy --config-file=.mypy.ini mace_jax tests scripts

# Tests
python -m pytest tests
