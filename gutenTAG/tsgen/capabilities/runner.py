"""End-to-end capability analysis runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from ..io import write_json
from ..manifest import canonical_json_hash
from .admission import AdmissionPolicy, admission_policy_from_yaml
from .dataset import discover_dataset
from .execution import build_run_context
from .protocol import CapabilityProtocol
from .runner_certificate import (
    OUTPUT_FILENAMES,
    build_capability_certificate,
    write_markdown_report,
)
from .runner_profiles import CapabilityProfileExecutor
from .runner_state import CapabilityRunState


@dataclass(frozen=True)
class _ResolvedAdmissionPolicy:
    policy: AdmissionPolicy | None
    policy_hash: str | None


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
    output_format: Literal["csv", "parquet", "both"] = "csv",
    release_csv: bool = True,
    allow_parquet_fallback: bool = False,
    label_export: Literal["full", "diagnostics"] = "full",
    admission_policy: AdmissionPolicy | None = None,
    admission_policy_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full capability analysis and write output artifacts."""

    _validate_run_options(
        admission_policy=admission_policy,
        admission_policy_path=admission_policy_path,
        label_export=label_export,
    )
    active_protocol = protocol or CapabilityProtocol()
    resolved_policy = _resolve_admission_policy(
        admission_policy=admission_policy,
        admission_policy_path=admission_policy_path,
    )
    dataset = discover_dataset(Path(dataset_root))
    output_path = _resolve_output_dir(dataset_root=dataset.root, output_dir=output_dir)
    context = build_run_context(
        dataset=dataset,
        protocol=active_protocol,
        output_dir=output_path,
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
        admission_policy_hash=resolved_policy.policy_hash,
    )
    state = _build_run_state(output_dir=output_path, context=context)
    _execute_profiles(
        dataset=dataset,
        protocol=active_protocol,
        context=context,
        state=state,
        output_dir=output_path,
        label_export=label_export,
        admission_policy=resolved_policy.policy,
    )
    return _finalize_capability_run(
        dataset=dataset,
        protocol=active_protocol,
        output_dir=output_path,
        context=context,
        state=state,
    )


def _validate_run_options(
    *,
    admission_policy: AdmissionPolicy | None,
    admission_policy_path: Path | None,
    label_export: Literal["full", "diagnostics"],
) -> None:
    if admission_policy is not None and admission_policy_path is not None:
        raise ValueError(
            "Pass either admission_policy or admission_policy_path, not both"
        )
    if label_export not in {"full", "diagnostics"}:
        raise ValueError(f"Unsupported label_export mode: {label_export!r}")


def _resolve_admission_policy(
    *,
    admission_policy: AdmissionPolicy | None,
    admission_policy_path: Path | None,
) -> _ResolvedAdmissionPolicy:
    policy = admission_policy
    if policy is None and admission_policy_path is not None:
        policy = admission_policy_from_yaml(admission_policy_path)
    policy_hash = canonical_json_hash(policy.to_dict()) if policy is not None else None
    return _ResolvedAdmissionPolicy(policy=policy, policy_hash=policy_hash)


def _resolve_output_dir(*, dataset_root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = dataset_root / "analysis" / "capability_v1"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _build_run_state(*, output_dir: Path, context: Any) -> CapabilityRunState:
    state = CapabilityRunState(
        output_dir=output_dir,
        cache_dir=context.cache_dir,
        output=context.output,
        output_filenames=OUTPUT_FILENAMES,
    )
    state.attach_to_manifest(context.run_manifest)
    return state


def _execute_profiles(
    *,
    dataset: Any,
    protocol: CapabilityProtocol,
    context: Any,
    state: CapabilityRunState,
    output_dir: Path,
    label_export: Literal["full", "diagnostics"],
    admission_policy: AdmissionPolicy | None,
) -> None:
    executor = CapabilityProfileExecutor(
        dataset=dataset,
        protocol=protocol,
        context=context,
        state=state,
        output_dir=output_dir,
        label_export=label_export,
        admission_policy=admission_policy,
    )
    executor.run_requested_profiles()


def _finalize_capability_run(
    *,
    dataset: Any,
    protocol: CapabilityProtocol,
    output_dir: Path,
    context: Any,
    state: CapabilityRunState,
) -> dict[str, Any]:
    tables = state.tables
    extra_artifacts = state.extra_artifacts
    context.output.write_manifests()
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    state.finish_runtime()
    write_json(manifest_dir / "profile_run_manifest.json", context.run_manifest)
    extra_artifacts["profile_run_manifest"] = "manifests/profile_run_manifest.json"

    certificate = build_capability_certificate(
        dataset=dataset,
        protocol=protocol,
        output_dir=output_dir,
        tables=tables,
        profile_names=context.profile_names,
        cache_dir=context.cache_dir,
        run_manifest=context.run_manifest,
        extra_artifacts=extra_artifacts,
    )
    write_json(output_dir / "capability_certificate.json", certificate)
    write_markdown_report(output_dir / "capability_report.md", certificate, tables)
    return certificate
