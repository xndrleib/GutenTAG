# Contributing

## Code style and format

We use [black](https://black.readthedocs.io/) to automatically format our python files.
Please stick to the black code style.

The `synth-gen` branch also uses Pyright as the primary type-checking gate for
the `gutenTAG` package. Before publishing broad changes, run:

```bash
black gutenTAG tests
flake8 gutenTAG tests
pyright
pytest
```

When using the local project environment, prefer:

```bash
.venv/bin/python -m black gutenTAG tests
.venv/bin/python -m flake8 gutenTAG tests
npx --yes pyright
.venv/bin/python -m pytest
```

Please consider using the pre-commit hooks.
They automatically run i.a. black for you.
See next section.

### Black quick-installation guide

```bash
pip install black
black
```

## Running pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run some checks on your files before they are commited.
Find the configured hooks in [`.pre-commit-config.yaml`](./pre-commit-config.yaml).
If there are errors, you have to re-add the files to the index and commit the fixed files.

### Pre-commit quick-installation guide

```bash
pip install pre-commit
pre-commit install
```

Optionally, cou can then run the hooks against all files with:

```bash
pre-commit run --all-files
```
