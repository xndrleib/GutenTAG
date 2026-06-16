# Development Notes

Status: current guidance for the `synth-gen` development branch.

This page records the local engineering contract for changes in the GutenTAG
package repository. The broader `synth-gen` documentation remains the canonical
source for dataset semantics and architecture, especially the parent-repository
`docs/architecture/code-ownership.md` page.

## Architecture Boundaries

Public package entry points should remain thin adapters:

- `gutenTAG.TSDatasetGenerator`
- `gutenTAG.TSGeneratorConfig`
- `gutenTAG.generate_ts_dataset`
- the command line entry points under `gutenTAG/__main__.py`

These adapters may parse inputs, preserve compatibility, and delegate work.
They should not accumulate new planning, validation, output, or anomaly
semantics when a focused owner module exists.

Current owner families:

- generation orchestration: `gutenTAG/ts_dataset_generation.py` plus focused
  helpers under `gutenTAG/tsgen/`;
- runtime configuration: `gutenTAG/tsgen/config/`;
- variant planning and segment placement: `gutenTAG/tsgen/planning/`;
- parameter realization and sanitization: `gutenTAG/tsgen/parameters/`;
- runtime anomaly semantics: `gutenTAG/generator/`;
- generated artifacts, summaries, labels, and manifests:
  `gutenTAG/tsgen/output/`, `gutenTAG/tsgen/labels/`, and
  `gutenTAG/tsgen/manifest.py`;
- capability analysis, validation, admission, visual audit, and release
  evidence: `gutenTAG/tsgen/capabilities/`.

If a change does not clearly fit one of these owners, create the smallest
coherent owner module instead of extending a large facade.

## Type And Quality Gates

The branch uses Pyright as the primary type gate. Pyright is configured in
`pyproject.toml` with `typeCheckingMode = "standard"` over the `gutenTAG`
package.

Before committing code changes, run:

```bash
.venv/bin/python -m black gutenTAG tests
.venv/bin/python -m flake8 gutenTAG tests
npx --yes pyright
.venv/bin/python -m pytest
```

For narrow changes, a targeted pytest run is useful while iterating, but the
full test suite should pass before publishing a broad refactor.

## Typed Boundary Rules

Most typing friction in this codebase comes from `pandas`, `numpy`, CSV, YAML,
and JSON boundaries. Keep conversions explicit:

- use helper functions for `DataFrame`, `Series`, and groupby casts;
- normalize nullable table fields before calling `float` or `int`;
- keep `Any` and `cast(...)` near I/O or table boundaries;
- do not let `Any` flow into domain policy logic;
- prefer explicit `Protocol` or dataclass contracts for internal runtime
  boundaries.

The goal is not to make dynamic data disappear. The goal is to quarantine it so
that planning, generation, validation, and admission code can be read as normal
Python with clear invariants.

## Large-File Guardrail

Large files are allowed only as compatibility facades or aggregation points.
When adding behavior, first check whether the behavior belongs in a focused
owner under `tsgen/`, `generator/`, or `tsgen/capabilities/`.

Good facade behavior:

- assemble a request;
- call owner modules;
- preserve public APIs;
- publish a small result object or manifest.

Bad facade behavior:

- branch on anomaly semantics inline;
- hand-build table rows already owned by a profile module;
- parse config sections already owned by `tsgen/config/`;
- add another long private method because nearby code already has access to
  the needed state.

