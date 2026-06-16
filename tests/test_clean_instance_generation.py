from pathlib import Path
from types import SimpleNamespace

import numpy as np

import gutenTAG.tsgen.clean_instance_generation as clean_module
from gutenTAG.tsgen.clean_instance_generation import generate_clean_only_instance
from gutenTAG.tsgen.config import TSGeneratorConfig
from gutenTAG.tsgen.variants import VariantSpec


def _clean_base_instance(observed: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        series=SimpleNamespace(observed_values=observed),
        base_parameters={"frequency": 1.0},
        base_channel_parameters=[{"phase": 0.0}, {"phase": 1.0}],
        base_parameters_per_channel=[
            {"frequency": 1.0, "phase": 0.0},
            {"frequency": 1.0, "phase": 1.0},
        ],
        split_phase_shift_info={"split": "train"},
    )


def _install_clean_generation_fakes(
    monkeypatch, calls: dict[str, object], base_instance
) -> None:
    def fake_prepare_base_instance(**kwargs):
        calls["prepare"] = kwargs
        return base_instance

    def fake_write_instance_artifacts(**kwargs):
        calls["artifacts"] = kwargs

    def fake_build_clean_instance_summary(**kwargs):
        calls["summary_kwargs"] = kwargs
        return {"instance_id": kwargs["instance_id"], "clean": True}

    def fake_write_json(path: Path, payload, **kwargs):
        calls["json"] = {"path": path, "payload": payload, "kwargs": kwargs}

    monkeypatch.setattr(
        clean_module,
        "prepare_base_instance",
        fake_prepare_base_instance,
    )
    monkeypatch.setattr(
        clean_module,
        "write_instance_artifacts",
        fake_write_instance_artifacts,
    )
    monkeypatch.setattr(
        clean_module,
        "build_clean_instance_summary",
        fake_build_clean_instance_summary,
    )
    monkeypatch.setattr(clean_module, "write_json", fake_write_json)


def _clean_generation_config(tmp_path: Path) -> TSGeneratorConfig:
    return TSGeneratorConfig(
        output_root=tmp_path / "dataset",
        master_seed=1,
        length=2,
        channels=2,
        generate_plots=False,
    )


def _assert_clean_generation_calls(
    calls: dict[str, object],
    observed: np.ndarray,
    instance_dir: Path,
) -> None:
    assert calls["prepare"]["base_kind"] == "sine"
    assert calls["prepare"]["base_channel_correlation"] == {"shared_noise_weight": 0.0}
    artifacts = calls["artifacts"]
    np.testing.assert_array_equal(artifacts["clean"], observed)
    np.testing.assert_array_equal(artifacts["anomalous"], observed)
    np.testing.assert_array_equal(artifacts["labels"], np.zeros((2, 2), dtype=np.int8))
    assert artifacts["events"] == []
    assert calls["summary_kwargs"]["base_parameters"] == {"frequency": 1.0}
    assert calls["json"]["path"] == instance_dir / "instance_summary.json"


def test_generate_clean_only_instance_writes_artifacts_and_summary(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    _install_clean_generation_fakes(
        monkeypatch,
        calls,
        _clean_base_instance(observed),
    )
    instance_dir = tmp_path / "instance_000"

    summary = generate_clean_only_instance(
        config=_clean_generation_config(tmp_path),
        variant=VariantSpec("sine", "mean", "p00"),
        split="train",
        instance_dir=instance_dir,
        seeds={"base_seed": 1, "base_shared_noise_seed": 2},
        base_parameter_template={},
        base_channel_parameter_template={},
        fixed_base_parameters={"frequency": 1.0},
        fixed_base_channel_parameters=({"phase": 0.0}, {"phase": 1.0}),
        effective_base_channel_correlation={"shared_noise_weight": 0.0},
    )

    assert summary == {"instance_id": "instance_000", "clean": True}
    _assert_clean_generation_calls(calls, observed, instance_dir)
