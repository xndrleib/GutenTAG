import pytest

from gutenTAG.tsgen.capabilities.admission import (
    AdmissionPolicy,
    admission_policy_from_yaml,
)
from gutenTAG.tsgen.capabilities.admission_policy import NUMERIC_GATE_NAMES


def test_admission_policy_from_mapping_accepts_nested_numeric_gate_statuses() -> None:
    policy = AdmissionPolicy.from_mapping(
        {
            "admission_policy": {
                "admission_policy_version": "synthgen.admission.test",
                "mode": "observation",
                "numeric_gates": {
                    "enforcement": "observation",
                    "promote_after_full_runs": 7,
                    "max_debug_event_share": {
                        "value": 0.25,
                        "status": "observation",
                    },
                    "custom_gate": {"value": 1.0, "status": "experimental"},
                },
            }
        }
    )

    payload = policy.to_dict()

    assert policy.version == "synthgen.admission.test"
    assert policy.mode == "observation"
    assert policy.numeric_gate_enforcement == "observation"
    assert policy.promote_after_full_runs == 7
    assert policy.max_debug_event_share == 0.25
    assert payload["numeric_gates"]["max_debug_event_share"]["status"] == "observation"
    assert payload["numeric_gates"]["custom_gate"] == {
        "value": 1.0,
        "status": "experimental",
    }
    assert set(NUMERIC_GATE_NAMES).issubset(payload["numeric_gates"])


def test_admission_policy_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unknown admission policy keys"):
        AdmissionPolicy.from_mapping({"unexpected": True})


def test_admission_policy_from_yaml_loads_default_when_path_is_missing() -> None:
    assert isinstance(admission_policy_from_yaml(None), AdmissionPolicy)
