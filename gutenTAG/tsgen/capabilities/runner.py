"""End-to-end capability analysis runner."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - Unix-only optional telemetry.
    resource = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

from ..io import write_json
from ..manifest import canonical_json_hash
from .admission import AdmissionPolicy, admission_policy_from_yaml, compute_admission_profiles
from .annotation import LABEL_TABLE_KEYS, compute_annotation_profiles
from .corrected_detectability import compute_corrected_detectability_frontier
from .dataset import DatasetIndex, discover_dataset
from .describability import compute_describability_profiles
from .detectability import compute_detectability_frontier
from .diagnosis import compute_diagnosis_profiles, merge_identifiability_summary
from .execution import build_run_context
from .identifiability import compute_identifiability_profiles
from .law_observability import compute_law_observability_profiles
from .model_zoo import compute_model_zoo_frontier
from .observability import compute_observability_profiles
from .protocol import CapabilityProtocol
from .repair import compute_repair_profiles
from .release import write_generator_feedback
from .validation import compute_implementation_validity
from .visualization import compute_visual_audit


OUTPUT_FILENAMES: Mapping[str, str] = {
    "observability": "observability_profile.csv",
    "arity": "arity_profile.csv",
    "event_summary": "event_capability_summary.csv",
    "detectability_frontier": "detectability_frontier.csv",
    "detectability_summary": "detectability_summary.csv",
    "corrected_detectability_frontier": "corrected_detectability_frontier.csv",
    "oracle_window_diagnostic_frontier": "oracle_window_diagnostic_frontier.csv",
    "blind_scan_events": "blind_scan_events.csv",
    "calibration_resolution": "calibration_resolution.csv",
    "model_zoo_frontier": "model_zoo_frontier.csv",
    "model_zoo_event_scores": "model_zoo_event_scores.csv",
    "law_observability_profile": "law_observability_profile.csv",
    "law_observability_summary": "law_observability_summary.csv",
    "identifiability_embeddings": "identifiability_event_embeddings.csv",
    "identifiability_summary": "identifiability_summary.csv",
    "identifiability_pairwise": "identifiability_pairwise.csv",
    "diagnosis_confusion_matrix": "diagnosis_confusion_matrix.csv",
    "identifiability_quotient": "identifiability_quotient.csv",
    "description_profile": "description_profile.csv",
    "description_summary": "description_summary.csv",
    "description_stability": "description_stability.csv",
    "repair_profile": "repair_profile.csv",
    "support_integrity": "support_integrity.csv",
    "boundary_audit": "boundary_audit.csv",
    "shortcut_audit": "shortcut_audit.csv",
    "realized_effects": "realized_effects.csv",
    "detector_attribution": "detector_attribution.csv",
    "negative_controls": "negative_controls.csv",
    "implementation_validity": "implementation_validity.csv",
    "labels_oracle_any": "labels/labels_oracle_any.csv",
    "labels_oracle_intervention": "labels/labels_oracle_intervention.csv",
    "labels_oracle_context": "labels/labels_oracle_context.csv",
    "labels_event_only": "labels/labels_event_only.csv",
    "labels_delayed": "labels/labels_delayed.csv",
    "labels_weak_point": "labels/labels_weak_point.csv",
    "labels_visible_only": "labels/labels_visible_only.csv",
    "labels_noisy_boundary": "labels/labels_noisy_boundary.csv",
    "labels_censored": "labels/labels_censored.csv",
    "annotation_alignment": "annotation_alignment.csv",
    "annotation_robustness": "annotation_robustness.csv",
    "admission_events": "admission_events.csv",
    "admission_variants": "admission_variants.csv",
    "visual_audit_selection": "visual_audit_selection.csv",
}


def run_capability_analysis(
    *,
    dataset_root: Path,
    output_dir: Path | None = None,
    protocol: CapabilityProtocol | None = None,
    profiles: Sequence[str] | None = None,
    profile_preset: str | None = None,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    materialize_arrays: bool = False,
    resume: bool = True,
    output_format: str = "csv",
    release_csv: bool = True,
    allow_parquet_fallback: bool = False,
    label_export: Literal["full", "diagnostics"] = "full",
    admission_policy: AdmissionPolicy | None = None,
    admission_policy_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full capability analysis and write output artifacts."""

    if admission_policy is not None and admission_policy_path is not None:
        raise ValueError("Pass either admission_policy or admission_policy_path, not both")
    if label_export not in {"full", "diagnostics"}:
        raise ValueError(f"Unsupported label_export mode: {label_export!r}")
    active_protocol = protocol or CapabilityProtocol()
    active_admission_policy = admission_policy
    if active_admission_policy is None and admission_policy_path is not None:
        active_admission_policy = admission_policy_from_yaml(admission_policy_path)
    admission_policy_hash = (
        canonical_json_hash(active_admission_policy.to_dict())
        if active_admission_policy is not None
        else None
    )
    dataset = discover_dataset(Path(dataset_root))
    if output_dir is None:
        output_dir = dataset.root / "analysis" / "capability_v1"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = build_run_context(
        dataset=dataset,
        protocol=active_protocol,
        output_dir=output_dir,
        profiles=profiles,
        profile_preset=profile_preset,
        n_jobs=n_jobs,
        cache_dir=cache_dir,
        materialize_arrays=materialize_arrays,
        resume=resume,
        output_format=output_format,
        release_csv=release_csv,
        allow_parquet_fallback=allow_parquet_fallback,
        label_export=label_export,
        admission_policy_hash=admission_policy_hash,
    )

    tables: dict[str, pd.DataFrame] = {}
    extra_artifacts: dict[str, str] = {}
    written_tables: set[str] = set()
    runtime = _initial_runtime_record()
    context.run_manifest["runtime"] = runtime

    def publish_tables(keys: Sequence[str]) -> None:
        for key in keys:
            if key in written_tables or key not in tables:
                continue
            context.output.write_table(key, OUTPUT_FILENAMES[key], tables[key])
            written_tables.add(key)

    @contextmanager
    def profile_stage(profile_name: str, **details: object) -> Iterator[None]:
        before_tables = set(written_tables)
        before_perf = time.perf_counter()
        before_wall = time.time()
        status = "complete"
        error: dict[str, str] | None = None
        try:
            yield
        except Exception as exc:
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            elapsed = max(0.0, time.perf_counter() - before_perf)
            written_delta = sorted(written_tables - before_tables)
            record: dict[str, object] = {
                "profile": profile_name,
                "status": status,
                "started_at_unix": before_wall,
                "finished_at_unix": time.time(),
                "elapsed_seconds": elapsed,
                "written_tables": written_delta,
                "table_count_delta": len(written_delta),
                "table_rows": {key: int(len(tables[key])) for key in written_delta if key in tables},
                "output_dir_size_bytes": _directory_size_bytes(output_dir),
                "cache_dir_size_bytes": _directory_size_bytes(context.cache_dir),
                "current_rss_bytes": _current_rss_bytes(),
                "peak_rss_bytes": _peak_rss_bytes(),
            }
            if details:
                record["details"] = dict(details)
            if error is not None:
                record["error"] = error
            runtime["profiles"].append(record)

    observability = None
    detectability = None
    corrected_detectability = None
    if "observability" in context.profile_names:
        with profile_stage("observability"):
            observability = compute_observability_profiles(
                dataset,
                active_protocol,
                arrays=context.arrays,
            )
            tables.update(
                {
                    "observability": observability.observability,
                    "arity": observability.arity,
                    "event_summary": observability.event_summary,
                }
            )
            publish_tables(("observability", "arity", "event_summary"))
    if "detectability" in context.profile_names:
        with profile_stage("detectability"):
            detectability = compute_detectability_frontier(
                dataset,
                active_protocol,
                arrays=context.arrays,
                windows=context.windows,
            )
            tables.update(
                {
                    "detectability_frontier": detectability.event_frontier,
                    "detectability_summary": detectability.summary,
                }
            )
            publish_tables(("detectability_frontier", "detectability_summary"))
    if "corrected_detectability" in context.profile_names:
        with profile_stage("corrected_detectability"):
            corrected_detectability = compute_corrected_detectability_frontier(
                dataset,
                active_protocol,
                arrays=context.arrays,
                windows=context.windows,
                cache=context.cache,
                n_jobs=context.n_jobs,
            )
            tables.update(
                {
                    "corrected_detectability_frontier": corrected_detectability.frontier,
                    "oracle_window_diagnostic_frontier": corrected_detectability.oracle_window_diagnostic,
                    "blind_scan_events": corrected_detectability.blind_scan_events,
                    "calibration_resolution": corrected_detectability.calibration_resolution,
                }
            )
            publish_tables(
                (
                    "corrected_detectability_frontier",
                    "oracle_window_diagnostic_frontier",
                    "blind_scan_events",
                    "calibration_resolution",
                )
            )
            write_json(
                output_dir / "candidate_nulls_manifest.json",
                corrected_detectability.candidate_nulls_manifest,
            )
            write_json(
                output_dir / "scan_nulls_manifest.json",
                corrected_detectability.scan_nulls_manifest,
            )
            extra_artifacts.update(
                {
                    "candidate_nulls_manifest": "candidate_nulls_manifest.json",
                    "scan_nulls_manifest": "scan_nulls_manifest.json",
                }
            )
    if "implementation_validity" in context.profile_names:
        if observability is None:
            raise ValueError("Implementation validity requires the observability profile")
        if detectability is None:
            raise ValueError("Implementation validity requires the detectability frontier")
        with profile_stage("implementation_validity"):
            validity = compute_implementation_validity(
                dataset,
                active_protocol,
                observability.observability,
                observability.event_summary,
                corrected_detectability.frontier
                if corrected_detectability is not None
                else detectability.event_frontier,
                arrays=context.arrays,
                cache=context.cache,
                n_jobs=context.n_jobs,
            )
            tables.update(
                {
                    "support_integrity": validity.support_integrity,
                    "boundary_audit": validity.boundary_audit,
                    "shortcut_audit": validity.shortcut_audit,
                    "realized_effects": validity.realized_effects,
                    "detector_attribution": validity.detector_attribution,
                    "negative_controls": validity.negative_controls,
                    "implementation_validity": validity.implementation_validity,
                }
            )
            publish_tables(
                (
                    "support_integrity",
                    "boundary_audit",
                    "shortcut_audit",
                    "realized_effects",
                    "detector_attribution",
                    "negative_controls",
                    "implementation_validity",
                )
            )
    if "model_zoo" in context.profile_names:
        with profile_stage("model_zoo"):
            model_zoo = compute_model_zoo_frontier(
                dataset,
                active_protocol,
                arrays=context.arrays,
                windows=context.windows,
                cache=context.cache,
                n_jobs=context.n_jobs,
            )
            tables.update(
                {
                    "model_zoo_frontier": model_zoo.frontier,
                    "model_zoo_event_scores": model_zoo.event_scores,
                }
            )
            publish_tables(("model_zoo_frontier", "model_zoo_event_scores"))
            write_json(output_dir / "model_zoo_model_manifest.json", model_zoo.model_manifest)
            extra_artifacts["model_zoo_model_manifest"] = "model_zoo_model_manifest.json"
    if "law_observability" in context.profile_names:
        with profile_stage("law_observability"):
            law_observability = compute_law_observability_profiles(
                dataset,
                active_protocol,
                arrays=context.arrays,
                cache=context.cache,
                n_jobs=context.n_jobs,
            )
            tables.update(
                {
                    "law_observability_profile": law_observability.profile,
                    "law_observability_summary": law_observability.summary,
                }
            )
            publish_tables(("law_observability_profile", "law_observability_summary"))
    if "identifiability" in context.profile_names:
        if observability is None:
            raise ValueError("Identifiability requires the observability profile")
        with profile_stage("identifiability"):
            identifiability = compute_identifiability_profiles(
                observability.observability,
                observability.event_summary,
                active_protocol,
            )
            diagnosis = compute_diagnosis_profiles(
                identifiability.event_embeddings,
                active_protocol,
            )
            tables.update(
                {
                    "identifiability_embeddings": identifiability.event_embeddings,
                    "identifiability_summary": merge_identifiability_summary(
                        identifiability.summary,
                        diagnosis.summary,
                    ),
                    "identifiability_pairwise": identifiability.pairwise,
                    "diagnosis_confusion_matrix": diagnosis.confusion_matrix,
                    "identifiability_quotient": diagnosis.quotient,
                }
            )
            publish_tables(
                (
                    "identifiability_embeddings",
                    "identifiability_summary",
                    "identifiability_pairwise",
                    "diagnosis_confusion_matrix",
                    "identifiability_quotient",
                )
            )
    if "describability" in context.profile_names:
        if observability is None:
            raise ValueError("Describability requires the observability profile")
        with profile_stage("describability"):
            describability = compute_describability_profiles(
                dataset,
                active_protocol,
                observability.observability,
                observability.event_summary,
                arrays=context.arrays,
            )
            repair = compute_repair_profiles(
                dataset,
                active_protocol,
                arrays=context.arrays,
            )
            tables.update(
                {
                    "description_profile": describability.description_profile,
                    "description_summary": describability.summary,
                    "description_stability": describability.description_stability,
                    "repair_profile": repair.repair_profile,
                }
            )
            publish_tables(
                (
                    "description_profile",
                    "description_summary",
                    "description_stability",
                    "repair_profile",
                )
            )
    if "annotation_alignment" in context.profile_names:
        with profile_stage(
            "annotation_alignment",
            label_export=label_export,
            label_tables_requested=label_export == "full",
            label_table_count=len(LABEL_TABLE_KEYS) if label_export == "full" else 0,
        ):
            annotation = compute_annotation_profiles(
                dataset,
                active_protocol,
                cache=context.cache,
                n_jobs=context.n_jobs,
                emit_label_tables=label_export == "full",
            )
            if label_export == "full":
                for key in LABEL_TABLE_KEYS:
                    tables[key] = annotation.label_tables[key]
            tables.update(
                {
                    "annotation_alignment": annotation.alignment,
                    "annotation_robustness": annotation.robustness,
                }
            )
            annotation_publish_keys = (
                tuple(LABEL_TABLE_KEYS) if label_export == "full" else ()
            ) + ("annotation_alignment", "annotation_robustness")
            publish_tables(annotation_publish_keys)
            write_json(output_dir / "annotation_channel_manifest.json", annotation.manifest)
            extra_artifacts["annotation_channel_manifest"] = "annotation_channel_manifest.json"
    if "admission" in context.profile_names:
        required = (
            "event_summary",
            "arity",
            "implementation_validity",
            "detector_attribution",
            "corrected_detectability_frontier",
        )
        missing = [key for key in required if key not in tables]
        if missing:
            raise ValueError(f"Admission requires prerequisite tables: {missing}")
        with profile_stage("admission"):
            admission = compute_admission_profiles(
                dataset,
                active_protocol,
                event_summary=tables["event_summary"],
                arity=tables["arity"],
                implementation_validity=tables["implementation_validity"],
                detector_attribution=tables["detector_attribution"],
                corrected_detectability=tables["corrected_detectability_frontier"],
                repair_profile=tables.get("repair_profile"),
                description_summary=tables.get("description_summary"),
                annotation_robustness=tables.get("annotation_robustness"),
                model_zoo_frontier=tables.get("model_zoo_frontier"),
                policy=active_admission_policy,
            )
            tables.update(
                {
                    "admission_events": admission.events,
                    "admission_variants": admission.variants,
                }
            )
            publish_tables(("admission_events", "admission_variants"))
            write_json(output_dir / "admission_policy_evaluation.json", admission.evaluation)
            write_json(output_dir / "release_summary.json", admission.release_summary)
            (output_dir / "release_summary.md").write_text(
                admission.release_summary_markdown,
                encoding="utf-8",
            )
            extra_artifacts.update(
                {
                    "admission_policy_evaluation": "admission_policy_evaluation.json",
                    "release_summary": "release_summary.json",
                    "release_summary_markdown": "release_summary.md",
                }
            )
    if "visual_audit" in context.profile_names:
        if "event_summary" not in tables:
            raise ValueError("Visual audit requires the event summary table")
        with profile_stage("visual_audit"):
            visual = compute_visual_audit(
                dataset,
                active_protocol,
                output_dir=output_dir,
                event_summary=tables["event_summary"],
                admission_events=tables.get("admission_events"),
                implementation_validity=tables.get("implementation_validity"),
                detector_attribution=tables.get("detector_attribution"),
                boundary_audit=tables.get("boundary_audit"),
                shortcut_audit=tables.get("shortcut_audit"),
                arrays=context.arrays,
            )
            tables["visual_audit_selection"] = visual.selection
            publish_tables(("visual_audit_selection",))
            write_json(output_dir / "visual_audit" / "visual_audit_manifest.json", visual.manifest)
            extra_artifacts.update(
                {
                    "visual_audit_index": "visual_audit/index.md",
                    "visual_audit_html_index": "visual_audit/index.html",
                    "visual_audit_manifest": "visual_audit/visual_audit_manifest.json",
                }
            )
    publish_tables(tuple(tables.keys()))
    if "admission" in context.profile_names:
        with profile_stage("generator_feedback"):
            write_generator_feedback(output_dir)
            extra_artifacts.update(
                {
                    "generator_feedback_report": "generator_feedback_report.json",
                    "generator_feedback_report_markdown": "generator_feedback_report.md",
                }
            )
    context.output.write_manifests()
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runtime["finished_at_unix"] = time.time()
    runtime["total_elapsed_seconds"] = max(
        0.0,
        float(runtime["finished_at_unix"]) - float(runtime["started_at_unix"]),
    )
    runtime["output_dir_size_bytes"] = _directory_size_bytes(output_dir)
    runtime["cache_dir_size_bytes"] = _directory_size_bytes(context.cache_dir)
    runtime["current_rss_bytes"] = _current_rss_bytes()
    runtime["peak_rss_bytes"] = _peak_rss_bytes()
    write_json(manifest_dir / "profile_run_manifest.json", context.run_manifest)
    extra_artifacts["profile_run_manifest"] = "manifests/profile_run_manifest.json"

    certificate = build_capability_certificate(
        dataset=dataset,
        protocol=active_protocol,
        output_dir=output_dir,
        tables=tables,
        profile_names=context.profile_names,
        cache_dir=context.cache_dir,
        run_manifest=context.run_manifest,
        extra_artifacts=extra_artifacts,
    )
    write_json(output_dir / "capability_certificate.json", certificate)
    _write_markdown_report(output_dir / "capability_report.md", certificate, tables)
    return certificate


def _initial_runtime_record() -> dict[str, object]:
    return {
        "runtime_manifest_version": "synthgen.capability.runtime.v1",
        "started_at_unix": time.time(),
        "profiles": [],
        "current_rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _directory_size_bytes(path: Path | None) -> int | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(current_root) / filename).stat().st_size
            except OSError:
                continue
    return int(total)


def _current_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            rss_pages = int(statm.read_text(encoding="utf-8").split()[1])
            return int(rss_pages * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, IndexError, ValueError):
            pass
    return None


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    if value <= 0:
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


def build_capability_certificate(
    *,
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    profile_names: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    run_manifest: Mapping[str, Any] | None = None,
    extra_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compact machine-readable capability certificate."""

    protocol_dict = protocol.to_dict()
    event_summary = tables.get("event_summary", pd.DataFrame())
    summary = {
        "instance_count": len(dataset.instances),
        "event_group_count": _dataset_event_count(dataset) if event_summary.empty else int(len(event_summary)),
        "variant_count": len({instance.variant_id for instance in dataset.instances})
        if event_summary.empty
        else int(event_summary["variant_id"].nunique()),
    }
    family_summary = _family_capability_summary(tables)
    payload: dict[str, Any] = {
        "capability_certificate_version": protocol.protocol_version,
        "dataset_root": str(dataset.root),
        "analysis_output_dir": str(output_dir),
        "analysis_cache_dir": str(cache_dir) if cache_dir is not None else None,
        "profiles": list(profile_names or []),
        "protocol": protocol_dict,
        "protocol_hash": canonical_json_hash(protocol_dict),
        "run_manifest": dict(run_manifest or {}),
        "dataset_schema_version": dataset.manifest.get("dataset_schema_version") if dataset.manifest else None,
        "dataset_normalized_config_hash": dataset.manifest.get("normalized_config_hash") if dataset.manifest else None,
        "summary": summary,
        "family_summary": family_summary,
        "artifacts": {
            key: OUTPUT_FILENAMES[key]
            for key, frame in tables.items()
            if key in OUTPUT_FILENAMES and (not frame.empty or key in OUTPUT_FILENAMES)
        },
    }
    if extra_artifacts:
        payload["artifacts"].update(dict(extra_artifacts))
    return payload


def _family_capability_summary(tables: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    event_summary = tables.get("event_summary", pd.DataFrame())
    if event_summary.empty:
        return []
    rows: list[dict[str, Any]] = []
    detectability = tables.get("detectability_summary", pd.DataFrame())
    descriptions = tables.get("description_summary", pd.DataFrame())
    arity = tables.get("arity", pd.DataFrame())
    grouped = event_summary.groupby(["variant_id", "anomaly_type", "constraint_tag", "semantic_scope"], dropna=False)
    for key, frame in grouped:
        variant_id, anomaly_type, constraint_tag, semantic_scope = key
        record: dict[str, Any] = {
            "variant_id": variant_id,
            "anomaly_type": anomaly_type,
            "constraint_tag": constraint_tag,
            "semantic_scope": semantic_scope,
            "event_count": int(len(frame)),
            "median_best_distance": float(frame["best_distance"].median()),
            "median_D_s1": float(frame.get("D_s1", pd.Series(dtype=float)).median()) if "D_s1" in frame else float("nan"),
            "median_D_s2": float(frame.get("D_s2", pd.Series(dtype=float)).median()) if "D_s2" in frame else float("nan"),
            "median_D_canonical_s1": float(frame.get("D_canonical_s1", pd.Series(dtype=float)).median()) if "D_canonical_s1" in frame else float("nan"),
            "median_D_canonical_s2": float(frame.get("D_canonical_s2", pd.Series(dtype=float)).median()) if "D_canonical_s2" in frame else float("nan"),
            "median_witness_sufficiency_proxy": float(frame["witness_sufficiency_proxy"].median()),
            "median_support_concentration_l2": float(frame["support_concentration_l2_max"].median()),
        }
        if not arity.empty:
            arity_frame = arity[(arity["variant_id"] == variant_id) & (arity["delta"] == arity["delta"].min())]
            if not arity_frame.empty:
                record["observable_share_at_min_delta"] = float(arity_frame["is_observable_at_delta"].mean())
                observed = arity_frame["observed_arity"].dropna()
                canonical_observed = arity_frame["canonical_observed_arity"].dropna() if "canonical_observed_arity" in arity_frame else pd.Series(dtype=float)
                record["median_observed_arity_at_min_delta"] = float(observed.median()) if not observed.empty else float("nan")
                record["canonical_observable_share_at_min_delta"] = float(arity_frame["canonical_is_observable_at_delta"].mean()) if "canonical_is_observable_at_delta" in arity_frame else float("nan")
                record["median_canonical_observed_arity_at_min_delta"] = float(canonical_observed.median()) if not canonical_observed.empty else float("nan")
        if not detectability.empty:
            det_frame = detectability[detectability["variant_id"] == variant_id]
            for alpha in sorted(det_frame["alpha"].unique()) if not det_frame.empty else []:
                alpha_frame = det_frame[det_frame["alpha"] == alpha]
                record[f"detected_rate_alpha_{float(alpha):g}"] = float(alpha_frame["detected_rate"].median())
        if not descriptions.empty:
            desc_frame = descriptions[descriptions["variant_id"] == variant_id]
            if not desc_frame.empty:
                record["median_witness_sufficiency"] = float(desc_frame["median_witness_sufficiency"].median())
                record["median_repair_gain"] = float(desc_frame["median_repair_gain"].median())
                record["median_description_risk_proxy"] = float(desc_frame["median_description_risk_proxy"].median())
        rows.append(record)
    return rows


def _dataset_event_count(dataset: DatasetIndex) -> int:
    return int(sum(len(instance.event_groups) for instance in dataset.instances))


def _write_markdown_report(path: Path, certificate: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]) -> None:
    lines: list[str] = []
    lines.append("# Synth-gen capability analysis report")
    lines.append("")
    lines.append(f"Certificate version: `{certificate.get('capability_certificate_version')}`")
    lines.append(f"Protocol hash: `{certificate.get('protocol_hash')}`")
    lines.append("")
    summary = certificate.get("summary", {})
    lines.append("## Dataset coverage")
    lines.append("")
    lines.append(f"- Instances: {summary.get('instance_count', 0)}")
    lines.append(f"- Event groups: {summary.get('event_group_count', 0)}")
    lines.append(f"- Variants: {summary.get('variant_count', 0)}")
    lines.append("")
    lines.append("## Family capability summary")
    lines.append("")
    family_summary = certificate.get("family_summary", [])
    if family_summary:
        frame = pd.DataFrame(family_summary)
        preferred = [
            "variant_id",
            "event_count",
            "median_best_distance",
            "median_D_s1",
            "median_D_s2",
            "median_D_canonical_s1",
            "median_D_canonical_s2",
            "observable_share_at_min_delta",
            "median_observed_arity_at_min_delta",
            "canonical_observable_share_at_min_delta",
            "median_canonical_observed_arity_at_min_delta",
            "detected_rate_alpha_0.01",
            "median_witness_sufficiency",
            "median_repair_gain",
        ]
        columns = [column for column in preferred if column in frame.columns]
        lines.append(_frame_to_markdown(frame[columns]))
    else:
        lines.append("No event groups were available for analysis.")
    lines.append("")
    lines.append("## Identifiability summary")
    lines.append("")
    identifiability = tables.get("identifiability_summary", pd.DataFrame())
    if identifiability is not None and not identifiability.empty:
        lines.append(_frame_to_markdown(identifiability))
    else:
        lines.append("Identifiability was not estimable for this run.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact pipe table without optional pandas dependencies."""

    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    rows = [[_format_markdown_cell(value) for value in record] for record in frame.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_markdown_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
