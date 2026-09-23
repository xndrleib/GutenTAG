import numpy as np
import pytest

from gutenTAG.tsgen.fingerprint_audit import nearest_centroid_fingerprint_accuracy
from gutenTAG.tsgen.ood_splits import parameter_range_overlap, validate_disjoint_holdouts
from gutenTAG.tsgen.sensor_artifacts import apply_sensor_artifact


@pytest.mark.parametrize("kind", ["clipping", "quantization", "stuck-at", "dropout", "timestamp-jitter"])
def test_sensor_artifacts_are_deterministic(kind):
    values = np.linspace(-1.0, 1.0, 64)
    first = apply_sensor_artifact(values, kind=kind, rng=np.random.default_rng(7), severity=0.6)
    second = apply_sensor_artifact(values, kind=kind, rng=np.random.default_rng(7), severity=0.6)
    np.testing.assert_allclose(first.values, second.values, equal_nan=True)
    np.testing.assert_array_equal(first.observed_mask, second.observed_mask)


def test_ood_contract_rejects_family_and_mechanism_leakage():
    with pytest.raises(ValueError, match="family-OOD leakage"):
        validate_disjoint_holdouts(
            train_families=["sine", "gp-mixture"],
            heldout_families=["gp-mixture"],
            train_mechanisms=["mean"],
            heldout_mechanisms=["copula-switch"],
        )


def test_parameter_overlap_audit():
    result = parameter_range_overlap(
        {"length_scale": (0.01, 0.1), "snr": (2.0, 5.0)},
        {"length_scale": (0.2, 0.5), "snr": (4.0, 8.0)},
    )
    assert result == {"length_scale": False, "snr": True}


def test_fingerprint_classifier_detects_obvious_nuisance_leakage():
    rows = []
    labels = []
    for label, length in [("mean", 10.0), ("variance", 100.0)]:
        for delta in range(8):
            rows.append({"source_length": length + delta * 0.01})
            labels.append(label)
    assert nearest_centroid_fingerprint_accuracy(rows, labels) > 0.9
