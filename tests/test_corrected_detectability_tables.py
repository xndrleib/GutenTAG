import pandas as pd

from gutenTAG.tsgen.capabilities.corrected_detectability_tables import (
    as_bool,
    blind_scan_columns,
    common_columns,
    false_alerts_by_alpha,
    frontier_columns,
    normalize_blind_scan_frame,
    ordered_rows,
    resolution_columns,
)
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


def test_corrected_detectability_column_contracts_are_stable() -> None:
    frontier = frontier_columns()
    blind_scan = blind_scan_columns()
    resolution = resolution_columns()

    assert frontier[: len(common_columns())] == common_columns()
    assert "candidate_level_p_value" in frontier
    assert "scan_level_p_value" in frontier
    assert "best_canonical_detected" in frontier
    assert "window_start" in blind_scan
    assert "false_alert" in blind_scan
    assert "calibration_resolution_min_p" in resolution


def test_ordered_rows_projects_columns_and_preserves_extra_fields() -> None:
    rows = (
        {"b": 2, "extra": "kept"},
        {"a": 1},
    )

    assert ordered_rows(rows, ("a", "b")) == (
        {"a": None, "b": 2, "extra": "kept"},
        {"a": 1, "b": None},
    )


def test_normalize_blind_scan_frame_restores_schema_and_bool_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "alpha": 0.10,
                "overlaps_oracle_event": "0",
                "false_alert": "true",
                "detected": "yes",
                "extra": "dropped",
            }
        ]
    )

    normalized = normalize_blind_scan_frame(frame)

    assert list(normalized.columns) == blind_scan_columns()
    assert normalized["false_alert"].tolist() == [True]
    assert normalized["overlaps_oracle_event"].tolist() == [False]
    assert normalized["detected"].tolist() == [True]
    assert "extra" not in normalized.columns


def test_normalize_blind_scan_frame_handles_empty_frames() -> None:
    normalized = normalize_blind_scan_frame(pd.DataFrame())

    assert list(normalized.columns) == blind_scan_columns()
    assert normalized.empty


def test_false_alerts_by_alpha_counts_only_false_alert_rows() -> None:
    protocol = CapabilityProtocol(alpha_grid=(0.10, 0.05))
    frame = pd.DataFrame(
        [
            {"alpha": 0.10, "false_alert": "true"},
            {"alpha": 0.10, "false_alert": False},
            {"alpha": 0.05, "false_alert": "yes"},
        ]
    )

    assert false_alerts_by_alpha(frame, protocol) == {0.10: 1, 0.05: 1}


def test_false_alerts_by_alpha_returns_protocol_grid_for_empty_frames() -> None:
    protocol = CapabilityProtocol(alpha_grid=(0.10, 0.05))

    assert false_alerts_by_alpha(pd.DataFrame(), protocol) == {0.10: 0, 0.05: 0}


def test_as_bool_normalizes_serialized_values() -> None:
    assert as_bool("true")
    assert as_bool(" YES ")
    assert as_bool("1")
    assert not as_bool("false")
    assert not as_bool("0")
    assert not as_bool("")
