"""Last checked source/type corrections; removed after the saved source commit."""
from pathlib import Path
r = Path(__file__).resolve().parents[1]


def patch(path, old, new):
    f = r/path
    s = f.read_text()
    if old not in s:
        raise ValueError(f"Patch precondition failed: {path}: {old[:80]}")
    f.write_text(s.replace(old, new))


p = r/'gutenTAG/tsgen/processes/interventions.py'
s = p.read_text(); i = s.index('    for event in metadata:')
s = s[:i]+s[i:].replace('for event in metadata:', 'for record in metadata:').replace('event[', 'record[')
p.write_text(s)
patch('gutenTAG/tsgen/capabilities/v13_evaluation.py', '    starts, ends = [], []', '    starts: list[int] = []\n    ends: list[int] = []')
patch('tests/test_corrected_detectability_partitions.py', '    instances = (SimpleNamespace(split="train"), SimpleNamespace(split="test"))', '    from typing import Any\n    instances: Any = (SimpleNamespace(split="train"), SimpleNamespace(split="test"))')
patch('gutenTAG/tsgen/fingerprint_audit.py', 'from typing import Mapping, Sequence', 'from typing import Any, Mapping, Sequence, cast')
patch('gutenTAG/tsgen/fingerprint_audit.py', 'float(row[key])', 'float(cast(Any, row[key]))')
patch('gutenTAG/tsgen/config/runtime_model.py', '        validate_raw_ts_config(dict(config))', '        if config.get("schema_version") == "synthgen.v13.1":\n            raise ValueError("Use BenchmarkConfig and generate_benchmark for v13.1 law datasets")\n        validate_raw_ts_config(dict(config))')
print('Type delta and legacy entrypoint guards applied')
