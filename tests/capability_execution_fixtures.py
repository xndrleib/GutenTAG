import importlib.util
from pathlib import Path

import pandas as pd

from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord


def _event_group(
    group_id: str,
    start: int,
    end: int,
    *,
    source_start: int | None = None,
    source_end: int | None = None,
) -> EventGroup:
    return EventGroup(
        group_id=group_id,
        start=start,
        end=end,
        source_start=start if source_start is None else source_start,
        source_end=end if source_end is None else source_end,
        anomaly_type="mean",
        constraint_tag="location.mean",
        repair_operator="additive_offset",
        semantic_scope="channel",
        intervention_channels=(0,),
        context_channels=(0,),
        group_channels=(0,),
        primary_channels=(0,),
        event_scope="unit_test",
        purity_hint="pure",
        raw_events=(),
    )


def _support_instance_record(
    root: Path,
    instance_dir: Path,
    event_groups: tuple[EventGroup, ...],
    *,
    length: int = 100,
) -> InstanceRecord:
    return InstanceRecord(
        dataset_root=root,
        variant_id="sine__mean__p00",
        split="train",
        instance_id="instance_000",
        instance_dir=instance_dir,
        clean_path=instance_dir / "clean.csv",
        anomalous_path=instance_dir / "anomalous.csv",
        events_path=instance_dir / "events.json",
        summary_path=instance_dir / "instance_summary.json",
        base_oscillation="sine",
        anomaly_type="mean",
        channels=1,
        length=length,
        event_groups=event_groups,
    )


def _relation_event_group(group_id: str, start: int, end: int) -> EventGroup:
    return EventGroup(
        group_id=group_id,
        start=start,
        end=end,
        source_start=start,
        source_end=end,
        anomaly_type="mode-correlation",
        constraint_tag="regime.mode_correlation",
        repair_operator="restore_regime_conditioned_dependence",
        semantic_scope="regime_relation",
        intervention_channels=(1,),
        context_channels=(0, 1),
        group_channels=(0, 1),
        primary_channels=(1,),
        event_scope="unit_test",
        purity_hint="pure",
        raw_events=(),
    )


def _instance_record_for_groups(
    event_groups: tuple[EventGroup, ...],
    *,
    channels: int,
    length: int,
) -> InstanceRecord:
    root = Path("/tmp/synth-gen-capability-test")
    return InstanceRecord(
        dataset_root=root,
        variant_id="random-mode-jump__mode-correlation__p00",
        split="train",
        instance_id="instance_000",
        instance_dir=root,
        clean_path=root / "clean.csv",
        anomalous_path=root / "anomalous.csv",
        events_path=root / "events.json",
        summary_path=root / "instance_summary.json",
        base_oscillation="random-mode-jump",
        anomaly_type="mode-correlation",
        channels=channels,
        length=length,
        event_groups=event_groups,
    )


def _small_generation_config(output_root: Path) -> dict:
    return {
        "generator": {
            "output_root": str(output_root),
            "master_seed": 1729,
            "overwrite_output": True,
            "log_level": "WARNING",
            "on_variant_failure": "skip",
        },
        "dataset": {
            "length": 120,
            "channels": 2,
            "splits": ["train"],
            "instances_per_split": 1,
        },
        "anomaly_policy": {
            "density_range": [0.05, 0.08],
            "density_tolerance": 0.05,
            "segment_count_range": [1, 2],
            "placement_policy": "uniform",
            "channel_policy": "single-random",
            "overlap_policy": "global",
            "length_normalization": "resample",
        },
        "variants": {
            "base_oscillations": ["sine"],
            "anomaly_types": ["mean"],
            "profiles_per_pair": 1,
            "pair_profiles": {},
            "base_parameter_policy": "random_per_instance",
            "base_channel_parameter_policy": "random_per_instance",
            "anomaly_parameter_policy": "fixed_per_variant",
            "skip_base_oscillations": [],
            "skip_anomaly_types": [],
            "disabled_anomaly_types": [],
        },
        "plot": {"enabled": False},
    }


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    if frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def _release_csv_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.csv"):
        relative_path = path.relative_to(root)
        if "tables_csv" in relative_path.parts:
            continue
        paths.append(relative_path)
    return sorted(paths)


def _parquet_engine_available() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )
