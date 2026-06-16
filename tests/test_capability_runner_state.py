from pathlib import Path

import pandas as pd
import pytest

from gutenTAG.tsgen.capabilities.runner_state import CapabilityRunState


class DummyOutput:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str, int]] = []

    def write_table(self, name: str, filename: str, frame: pd.DataFrame) -> None:
        self.writes.append((name, filename, int(len(frame))))


def _state(tmp_path: Path) -> tuple[CapabilityRunState, DummyOutput]:
    output = DummyOutput()
    state = CapabilityRunState(
        output_dir=tmp_path,
        cache_dir=None,
        output=output,
        output_filenames={"table": "table.csv"},
    )
    return state, output


def test_publish_tables_writes_each_known_table_once(tmp_path: Path) -> None:
    state, output = _state(tmp_path)
    state.tables["table"] = pd.DataFrame({"value": [1, 2]})

    state.publish_tables(("missing", "table", "table"))

    assert output.writes == [("table", "table.csv", 2)]
    assert state.written_tables == {"table"}


def test_profile_stage_records_success_details_and_table_rows(tmp_path: Path) -> None:
    state, output = _state(tmp_path)
    state.tables["table"] = pd.DataFrame({"value": [1, 2, 3]})

    with state.profile_stage("unit_profile", label_export="diagnostics"):
        state.publish_tables(("table",))

    assert output.writes == [("table", "table.csv", 3)]
    record = state.runtime["profiles"][0]
    assert record["profile"] == "unit_profile"
    assert record["status"] == "complete"
    assert record["written_tables"] == ["table"]
    assert record["table_rows"] == {"table": 3}
    assert record["details"] == {"label_export": "diagnostics"}


def test_profile_stage_records_failure_before_reraising(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)

    with pytest.raises(RuntimeError):
        with state.profile_stage("failing_profile"):
            raise RuntimeError("boom")

    record = state.runtime["profiles"][0]
    assert record["profile"] == "failing_profile"
    assert record["status"] == "failed"
    assert record["error"] == {"type": "RuntimeError", "message": "boom"}


def test_finish_runtime_adds_aggregate_metrics(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    (tmp_path / "artifact.txt").write_text("payload", encoding="utf-8")

    state.finish_runtime()

    assert state.runtime["total_elapsed_seconds"] >= 0.0
    assert state.runtime["output_dir_size_bytes"] >= 7
    assert state.runtime["cache_dir_size_bytes"] is None
