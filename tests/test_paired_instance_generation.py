from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import gutenTAG.tsgen.paired_instance_generation as paired_module
from gutenTAG.tsgen.config import TSGeneratorConfig
from gutenTAG.tsgen.paired_instance_generation import generate_paired_instance
from gutenTAG.tsgen.variants import VariantSpec


@dataclass(frozen=True)
class _PairedFakes:
    clean_base: np.ndarray
    clean_observed: np.ndarray
    anomalous_base: np.ndarray
    anomalous_observed: np.ndarray
    labels: np.ndarray
    events: list[dict[str, int]]
    base_instance: Any
    anomalous_series: Any
    planning: Any


def test_generate_paired_instance_writes_artifacts_summary_and_events(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fakes = _paired_fakes()
    _install_paired_fakes(monkeypatch, calls, fakes)

    instance_dir = tmp_path / "instance_000"
    summary = _generate_test_paired_instance(tmp_path, instance_dir)

    assert summary == {"instance_id": "instance_000", "paired": True}
    _assert_generation_inputs(calls)
    _assert_runtime_wiring(calls, fakes)
    _assert_artifact_and_summary_writes(calls, fakes, instance_dir)


def _paired_fakes() -> _PairedFakes:
    clean_base = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    clean_observed = np.asarray([[1.1], [2.1], [3.1]], dtype=np.float64)
    anomalous_base = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)
    anomalous_observed = np.asarray([[11.0], [21.0], [31.0]], dtype=np.float64)
    labels = np.asarray([[0], [1], [0]], dtype=np.int8)
    events = [{"start": 1, "end": 2, "channel": 0}]
    return _PairedFakes(
        clean_base=clean_base,
        clean_observed=clean_observed,
        anomalous_base=anomalous_base,
        anomalous_observed=anomalous_observed,
        labels=labels,
        events=events,
        base_instance=_base_instance(clean_base, clean_observed),
        anomalous_series=SimpleNamespace(
            base_values=anomalous_base,
            channel_bos=["channel-bo"],
        ),
        planning=_planning_result(),
    )


def _base_instance(clean_base: np.ndarray, clean_observed: np.ndarray) -> Any:
    return SimpleNamespace(
        series=SimpleNamespace(
            base_values=clean_base,
            observed_values=clean_observed,
        ),
        base_parameters={"frequency": 1.0},
        base_channel_parameters=[{"phase": 0.0}],
        base_parameters_per_channel=[{"frequency": 1.0, "phase": 0.0}],
        split_phase_shift_info={"split": "train"},
    )


def _planning_result() -> Any:
    return SimpleNamespace(
        segment_plan=("segment",),
        target_density=0.1,
        active_density_range=(0.1, 0.2),
        active_density_tolerance=0.01,
        active_channel_policy="single-random",
        active_overlap_policy="global",
        planner_cfg={"strategy": "uniform"},
        special_policy={"policy": "variant"},
    )


def _install_paired_fakes(
    monkeypatch, calls: dict[str, object], fakes: _PairedFakes
) -> None:
    for name, fake in _paired_fake_functions(calls, fakes).items():
        monkeypatch.setattr(paired_module, name, fake)


def _paired_fake_functions(
    calls: dict[str, object],
    fakes: _PairedFakes,
) -> dict[str, Any]:
    def fake_prepare_base_instance(**kwargs):
        calls["prepare"] = kwargs
        return fakes.base_instance

    def fake_generate_base_instance_series(**kwargs):
        calls["base_series"] = kwargs
        return fakes.anomalous_series

    def fake_plan_instance_segments(**kwargs):
        calls["plan"] = kwargs
        return fakes.planning

    def fake_resolve_runtime_anomaly_parameters_for_segments(**kwargs):
        calls["runtime_params"] = kwargs
        return [{"amplitude": 2.0}]

    def fake_build_anomalies(**kwargs):
        calls["anomalies"] = kwargs
        return ("anomaly",)

    def fake_apply_runtime_anomalies(**kwargs):
        calls["apply"] = kwargs
        return fakes.labels, fakes.events

    def fake_apply_variations(base, channel_bos):
        calls["variations"] = {"base": base, "channel_bos": channel_bos}
        return fakes.anomalous_observed

    def fake_annotate_segment_local_stats(segment_plan, clean_values):
        calls["local_stats"] = {
            "segment_plan": segment_plan,
            "clean_values": clean_values,
        }

    def fake_write_instance_artifacts(**kwargs):
        calls["artifacts"] = kwargs

    def fake_build_paired_instance_summary(**kwargs):
        calls["summary_kwargs"] = kwargs
        return {"instance_id": kwargs["instance_id"], "paired": True}

    def fake_write_json(path: Path, payload, **kwargs):
        calls["json"] = {"path": path, "payload": payload, "kwargs": kwargs}

    return {
        "prepare_base_instance": fake_prepare_base_instance,
        "generate_base_instance_series": fake_generate_base_instance_series,
        "plan_instance_segments": fake_plan_instance_segments,
        "resolve_runtime_anomaly_parameters_for_segments": (
            fake_resolve_runtime_anomaly_parameters_for_segments
        ),
        "build_anomalies": fake_build_anomalies,
        "apply_runtime_anomalies": fake_apply_runtime_anomalies,
        "apply_variations": fake_apply_variations,
        "annotate_segment_local_stats": fake_annotate_segment_local_stats,
        "write_instance_artifacts": fake_write_instance_artifacts,
        "build_paired_instance_summary": fake_build_paired_instance_summary,
        "write_json": fake_write_json,
    }


def _generate_test_paired_instance(
    tmp_path: Path, instance_dir: Path
) -> dict[str, Any]:
    return generate_paired_instance(
        config=TSGeneratorConfig(
            output_root=tmp_path / "dataset",
            master_seed=1,
            length=3,
            channels=1,
            generate_plots=False,
        ),
        variant=VariantSpec("sine", "mean", "p00"),
        split="train",
        instance_dir=instance_dir,
        seeds=_paired_seeds(),
        base_parameter_template={},
        base_channel_parameter_template={},
        anomaly_parameter_template={"amplitude": 2.0},
        fixed_base_parameters={"frequency": 1.0},
        fixed_base_channel_parameters=({"phase": 0.0},),
        fixed_anomaly_parameters={"amplitude": 2.0},
        effective_base_channel_correlation={"shared_noise_weight": 0.0},
        variant_anomaly_policy={"policy": "variant"},
        variant_segment_planner={"strategy": "uniform"},
        logger=SimpleNamespace(),
    )


def _paired_seeds() -> dict[str, int]:
    return {
        "base_seed": 1,
        "base_shared_noise_seed": 2,
        "params_seed": 3,
        "plan_seed": 4,
        "anomaly_seed": 5,
        "zoom_seed": 6,
    }


def _assert_generation_inputs(calls: dict[str, object]) -> None:
    assert calls["prepare"]["base_kind"] == "sine"
    assert calls["prepare"]["base_channel_correlation"] == {"shared_noise_weight": 0.0}
    assert calls["base_series"]["base_kind"] == "sine"
    assert calls["plan"]["anomaly_type"] == "mean"
    assert calls["plan"]["variant_segment_planner"] == {"strategy": "uniform"}


def _assert_runtime_wiring(calls: dict[str, object], fakes: _PairedFakes) -> None:
    assert calls["runtime_params"]["segment_plan"] == ("segment",)
    assert calls["anomalies"]["anomaly_parameters_per_segment"] == [{"amplitude": 2.0}]
    np.testing.assert_array_equal(
        calls["local_stats"]["clean_values"], fakes.clean_base
    )
    np.testing.assert_array_equal(calls["apply"]["base"], fakes.anomalous_base)
    assert calls["apply"]["channel_bos"] == ["channel-bo"]
    np.testing.assert_array_equal(calls["variations"]["base"], fakes.anomalous_base)


def _assert_artifact_and_summary_writes(
    calls: dict[str, object],
    fakes: _PairedFakes,
    instance_dir: Path,
) -> None:
    artifacts = calls["artifacts"]
    np.testing.assert_array_equal(artifacts["clean"], fakes.clean_observed)
    np.testing.assert_array_equal(artifacts["anomalous"], fakes.anomalous_observed)
    np.testing.assert_array_equal(artifacts["labels"], fakes.labels)
    assert artifacts["events"] == fakes.events
    assert calls["summary_kwargs"]["variant_id"] == "sine__mean__p00"
    assert calls["summary_kwargs"]["base_channel_correlation"] == {
        "shared_noise_weight": 0.0
    }
    assert calls["json"]["path"] == instance_dir / "instance_summary.json"
