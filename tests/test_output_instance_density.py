import numpy as np
import pytest

from gutenTAG.tsgen.output.instance_density import (
    compute_density_metrics,
    validate_density,
)


def test_compute_density_metrics_uses_source_support_for_effective_mode() -> None:
    labels = np.zeros((6, 2), dtype=np.int8)
    labels[2:4, 1] = 1

    metrics = compute_density_metrics(
        labels=labels,
        events=[
            {
                "start": 2,
                "end": 4,
                "source_start": 1,
                "source_end": 5,
                "channel": 1,
            }
        ],
        support_label_mode="effective_support",
    )

    assert metrics["achieved_density_labeled"] == pytest.approx(2 / 6)
    assert metrics["achieved_density_source"] == pytest.approx(4 / 6)
    assert metrics["density_for_validation"] == pytest.approx(4 / 6)
    assert metrics["density_validation_mode"] == "source_support"


def test_validate_density_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="density_tolerance"):
        validate_density(
            target_density=0.5,
            active_density_range=(0.0, 1.0),
            active_density_tolerance=-0.1,
            achieved_density_labeled=0.5,
            achieved_density_source=0.5,
            density_for_validation=0.5,
            density_validation_mode="labeled_support",
        )


def test_validate_density_rejects_target_mismatch() -> None:
    with pytest.raises(ValueError, match="Density mismatch"):
        validate_density(
            target_density=0.5,
            active_density_range=(0.0, 1.0),
            active_density_tolerance=0.01,
            achieved_density_labeled=0.2,
            achieved_density_source=0.2,
            density_for_validation=0.2,
            density_validation_mode="labeled_support",
        )
