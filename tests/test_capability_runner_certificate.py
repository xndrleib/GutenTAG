from pathlib import Path

import pandas as pd

from gutenTAG.tsgen.capabilities.dataset import DatasetIndex, InstanceRecord
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol
from gutenTAG.tsgen.capabilities.runner_certificate import (
    OUTPUT_FILENAMES,
    build_capability_certificate,
    family_capability_summary,
    frame_to_markdown,
    write_markdown_report,
)


def _dataset(root: Path) -> DatasetIndex:
    instance = InstanceRecord(
        dataset_root=root,
        variant_id="variant-a",
        split="train",
        instance_id="instance_000",
        instance_dir=root,
        clean_path=root / "clean.csv",
        anomalous_path=root / "anomalous.csv",
        events_path=root / "events.json",
        summary_path=root / "instance_summary.json",
        base_oscillation="sine",
        anomaly_type="mean",
        channels=1,
        length=64,
        event_groups=(),
    )
    return DatasetIndex(
        root=root,
        manifest={
            "dataset_schema_version": "synthgen.dataset.test",
            "normalized_config_hash": "hash-123",
        },
        instances=(instance,),
    )


def _tables() -> dict[str, pd.DataFrame]:
    event_summary = pd.DataFrame(
        [
            {
                "variant_id": "variant-a",
                "anomaly_type": "mean",
                "constraint_tag": "location.mean",
                "semantic_scope": "univariate",
                "best_distance": 0.20,
                "D_s1": 0.10,
                "D_s2": 0.30,
                "D_canonical_s1": 0.15,
                "D_canonical_s2": 0.35,
                "witness_sufficiency_proxy": 0.80,
                "support_concentration_l2_max": 0.40,
            }
        ]
    )
    return {
        "event_summary": event_summary,
        "arity": pd.DataFrame(
            [
                {
                    "variant_id": "variant-a",
                    "delta": 0.20,
                    "is_observable_at_delta": True,
                    "observed_arity": 1,
                    "canonical_is_observable_at_delta": True,
                    "canonical_observed_arity": 1,
                }
            ]
        ),
        "detectability_summary": pd.DataFrame(
            [{"variant_id": "variant-a", "alpha": 0.10, "detected_rate": 0.75}]
        ),
        "description_summary": pd.DataFrame(
            [
                {
                    "variant_id": "variant-a",
                    "median_witness_sufficiency": 0.90,
                    "median_repair_gain": 0.25,
                    "median_description_risk_proxy": 0.05,
                }
            ]
        ),
    }


def test_family_capability_summary_combines_optional_profile_tables() -> None:
    summary = family_capability_summary(_tables())

    assert len(summary) == 1
    row = summary[0]
    assert row["variant_id"] == "variant-a"
    assert row["event_count"] == 1
    assert row["observable_share_at_min_delta"] == 1.0
    assert row["detected_rate_alpha_0.1"] == 0.75
    assert row["median_witness_sufficiency"] == 0.90


def test_build_capability_certificate_uses_output_contract_and_extra_artifacts(
    tmp_path: Path,
) -> None:
    certificate = build_capability_certificate(
        dataset=_dataset(tmp_path),
        protocol=CapabilityProtocol(alpha_grid=(0.10,)),
        output_dir=tmp_path / "analysis",
        tables=_tables(),
        profile_names=("observability",),
        run_manifest={"runtime": {"profiles": []}},
        extra_artifacts={"profile_run_manifest": "manifests/profile_run_manifest.json"},
    )

    assert certificate["summary"] == {
        "instance_count": 1,
        "event_group_count": 1,
        "variant_count": 1,
    }
    assert certificate["dataset_schema_version"] == "synthgen.dataset.test"
    assert certificate["dataset_normalized_config_hash"] == "hash-123"
    assert (
        certificate["artifacts"]["event_summary"] == OUTPUT_FILENAMES["event_summary"]
    )
    assert (
        certificate["artifacts"]["profile_run_manifest"]
        == "manifests/profile_run_manifest.json"
    )


def test_write_markdown_report_renders_summary_sections(tmp_path: Path) -> None:
    certificate = build_capability_certificate(
        dataset=_dataset(tmp_path),
        protocol=CapabilityProtocol(alpha_grid=(0.10,)),
        output_dir=tmp_path / "analysis",
        tables=_tables(),
        profile_names=("observability",),
    )
    path = tmp_path / "capability_report.md"

    write_markdown_report(
        path,
        certificate,
        {"identifiability_summary": pd.DataFrame([{"metric": "x", "value": 1.0}])},
    )

    text = path.read_text(encoding="utf-8")
    assert "# Synth-gen capability analysis report" in text
    assert "## Dataset coverage" in text
    assert "## Identifiability summary" in text


def test_frame_to_markdown_renders_pipe_table() -> None:
    rendered = frame_to_markdown(pd.DataFrame([{"name": "alpha", "value": 0.1234567}]))

    assert "| name  | value" in rendered
    assert "0.123457" in rendered
