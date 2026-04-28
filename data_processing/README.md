# Develop
```
python3 -m venv venv
```

## Install dependencies
```
venv/bin/pip install -e .[dev]
```

## Checks
```
venv/bin/pylint src
```

## Test
```
venv/bin/pytest -v src
```

# Release
```
venv/bin/python -m build
venv/bin/twine upload [--repository testpypi] dist/*
```
