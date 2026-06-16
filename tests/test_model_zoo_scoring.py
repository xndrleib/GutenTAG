from types import SimpleNamespace

import math
import numpy as np

from gutenTAG.tsgen.capabilities.model_zoo_scoring import (
    context_score,
    model_null,
    model_scan_statistic,
    overlaps,
    scan_window_specs,
)
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


class ToyModel:
    model_id = "toy"
    family = "structural.toy"

    def score_windows(self, series, windows):
        values = []
        for start, end in np.asarray(windows, dtype=int):
            window = np.asarray(series[int(start) : int(end)], dtype=np.float64)
            values.append(float(np.mean(window)))
        return np.asarray(values, dtype=np.float64)


def test_model_null_builds_candidate_and_scan_nulls() -> None:
    null = model_null(
        model=ToyModel(),
        clean_instances=[np.arange(12, dtype=np.float64).reshape(6, 2)],
        event_length=2,
        protocol=CapabilityProtocol(max_scan_windows_per_length=4),
        windows=None,
    )

    assert null.candidate_null.family == "structural.toy"
    assert null.candidate_null.witness_or_model == "toy"
    assert null.raw_candidate_count > 0
    assert null.scan_null.scope == "model_zoo"
    assert math.isfinite(model_scan_statistic(null, raw_score=1.0))


def test_context_score_scores_window_around_group() -> None:
    group = SimpleNamespace(start=2, end=4, length=2)
    series = np.arange(10, dtype=np.float64).reshape(5, 2)

    score = context_score(ToyModel(), series, group, length=5)

    assert score == np.mean(series[0:5])


def test_scan_window_specs_and_overlap_helpers() -> None:
    windows = scan_window_specs(
        series_length=10,
        event_length=3,
        protocol=CapabilityProtocol(max_scan_windows_per_length=4),
        windows=None,
    )

    assert len(windows) <= 4
    assert overlaps(0, 3, 2, 5)
    assert not overlaps(0, 2, 2, 5)
