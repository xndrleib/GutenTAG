"""Profile execution handlers for capability analysis runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from ..io import write_json
from .admission import AdmissionPolicy, compute_admission_profiles
from .annotation import LABEL_TABLE_KEYS, compute_annotation_profiles
from .corrected_detectability import (
    CorrectedDetectabilityResult,
    compute_corrected_detectability_frontier,
)
from .dataset import DatasetIndex
from .describability import compute_describability_profiles
from .detectability import DetectabilityResult, compute_detectability_frontier
from .diagnosis import compute_diagnosis_profiles, merge_identifiability_summary
from .execution import CapabilityRunContext
from .identifiability import compute_identifiability_profiles
from .law_observability import compute_law_observability_profiles
from .model_zoo import compute_model_zoo_frontier
from .observability import ObservabilityResult, compute_observability_profiles
from .protocol import CapabilityProtocol
from .repair import compute_repair_profiles
from .release import write_generator_feedback
from .runner_state import CapabilityRunState
from .validation import compute_implementation_validity
from .visualization import compute_visual_audit


@dataclass
class CapabilityProfileExecutor:
    """Execute requested capability profiles against shared run state."""

    dataset: DatasetIndex
    protocol: CapabilityProtocol
    context: CapabilityRunContext
    state: CapabilityRunState
    output_dir: Path
    label_export: Literal["full", "diagnostics"]
    admission_policy: AdmissionPolicy | None = None
    observability: ObservabilityResult | None = None
    detectability: DetectabilityResult | None = None
    corrected_detectability: CorrectedDetectabilityResult | None = None

    def run_requested_profiles(self) -> None:
        """Run requested profiles in dependency order."""

        if self._requested("observability"):
            self.run_observability()
        if self._requested("detectability"):
            self.run_detectability()
        if self._requested("corrected_detectability"):
            self.run_corrected_detectability()
        if self._requested("implementation_validity"):
            self.run_implementation_validity()
        if self._requested("model_zoo"):
            self.run_model_zoo()
        if self._requested("law_observability"):
            self.run_law_observability()
        if self._requested("identifiability"):
            self.run_identifiability()
        if self._requested("describability"):
            self.run_describability()
        if self._requested("annotation_alignment"):
            self.run_annotation_alignment()
        if self._requested("admission"):
            self.run_admission()
        if self._requested("visual_audit"):
            self.run_visual_audit()
        self.state.publish_tables(tuple(self.state.tables.keys()))
        if self._requested("admission"):
            self.run_generator_feedback()

    def run_observability(self) -> None:
        """Run observed separability and arity profiles."""

        with self.state.profile_stage("observability"):
            self.observability = compute_observability_profiles(
                self.dataset,
                self.protocol,
                arrays=self.context.arrays,
            )
            self.state.tables.update(
                {
                    "observability": self.observability.observability,
                    "arity": self.observability.arity,
                    "event_summary": self.observability.event_summary,
                }
            )
            self.state.publish_tables(("observability", "arity", "event_summary"))

    def run_detectability(self) -> None:
        """Run legacy clean-calibrated detectability."""

        with self.state.profile_stage("detectability"):
            self.detectability = compute_detectability_frontier(
                self.dataset,
                self.protocol,
                arrays=self.context.arrays,
                windows=self.context.windows,
            )
            self.state.tables.update(
                {
                    "detectability_frontier": self.detectability.event_frontier,
                    "detectability_summary": self.detectability.summary,
                }
            )
            self.state.publish_tables(
                ("detectability_frontier", "detectability_summary")
            )

    def run_corrected_detectability(self) -> None:
        """Run scan-corrected detectability and null manifests."""

        with self.state.profile_stage("corrected_detectability"):
            self.corrected_detectability = compute_corrected_detectability_frontier(
                self.dataset,
                self.protocol,
                arrays=self.context.arrays,
                windows=self.context.windows,
                cache=self.context.cache,
                n_jobs=self.context.n_jobs,
            )
            self.state.tables.update(
                {
                    "corrected_detectability_frontier": self.corrected_detectability.frontier,
                    "oracle_window_diagnostic_frontier": (
                        self.corrected_detectability.oracle_window_diagnostic
                    ),
                    "blind_scan_events": self.corrected_detectability.blind_scan_events,
                    "calibration_resolution": (
                        self.corrected_detectability.calibration_resolution
                    ),
                }
            )
            self.state.publish_tables(
                (
                    "corrected_detectability_frontier",
                    "oracle_window_diagnostic_frontier",
                    "blind_scan_events",
                    "calibration_resolution",
                )
            )
            write_json(
                self.output_dir / "candidate_nulls_manifest.json",
                self.corrected_detectability.candidate_nulls_manifest,
            )
            write_json(
                self.output_dir / "scan_nulls_manifest.json",
                self.corrected_detectability.scan_nulls_manifest,
            )
            self.state.extra_artifacts.update(
                {
                    "candidate_nulls_manifest": "candidate_nulls_manifest.json",
                    "scan_nulls_manifest": "scan_nulls_manifest.json",
                }
            )

    def run_implementation_validity(self) -> None:
        """Run implementation-validity audits."""

        if self.observability is None:
            raise ValueError(
                "Implementation validity requires the observability profile"
            )
        if self.detectability is None:
            raise ValueError(
                "Implementation validity requires the detectability frontier"
            )
        frontier = (
            self.corrected_detectability.frontier
            if self.corrected_detectability is not None
            else self.detectability.event_frontier
        )
        with self.state.profile_stage("implementation_validity"):
            validity = compute_implementation_validity(
                self.dataset,
                self.protocol,
                self.observability.observability,
                self.observability.event_summary,
                frontier,
                arrays=self.context.arrays,
                cache=self.context.cache,
                n_jobs=self.context.n_jobs,
            )
            self.state.tables.update(
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
            self.state.publish_tables(
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

    def run_model_zoo(self) -> None:
        """Run model-zoo detectability profiles."""

        with self.state.profile_stage("model_zoo"):
            model_zoo = compute_model_zoo_frontier(
                self.dataset,
                self.protocol,
                arrays=self.context.arrays,
                windows=self.context.windows,
                cache=self.context.cache,
                n_jobs=self.context.n_jobs,
            )
            self.state.tables.update(
                {
                    "model_zoo_frontier": model_zoo.frontier,
                    "model_zoo_event_scores": model_zoo.event_scores,
                }
            )
            self.state.publish_tables(("model_zoo_frontier", "model_zoo_event_scores"))
            write_json(
                self.output_dir / "model_zoo_model_manifest.json",
                model_zoo.model_manifest,
            )
            self.state.extra_artifacts["model_zoo_model_manifest"] = (
                "model_zoo_model_manifest.json"
            )

    def run_law_observability(self) -> None:
        """Run law-observability profiles."""

        with self.state.profile_stage("law_observability"):
            law_observability = compute_law_observability_profiles(
                self.dataset,
                self.protocol,
                arrays=self.context.arrays,
                cache=self.context.cache,
                n_jobs=self.context.n_jobs,
            )
            self.state.tables.update(
                {
                    "law_observability_profile": law_observability.profile,
                    "law_observability_summary": law_observability.summary,
                }
            )
            self.state.publish_tables(
                ("law_observability_profile", "law_observability_summary")
            )

    def run_identifiability(self) -> None:
        """Run identifiability and diagnosis profiles."""

        if self.observability is None:
            raise ValueError("Identifiability requires the observability profile")
        with self.state.profile_stage("identifiability"):
            identifiability = compute_identifiability_profiles(
                self.observability.observability,
                self.observability.event_summary,
                self.protocol,
            )
            diagnosis = compute_diagnosis_profiles(
                identifiability.event_embeddings,
                self.protocol,
            )
            self.state.tables.update(
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
            self.state.publish_tables(
                (
                    "identifiability_embeddings",
                    "identifiability_summary",
                    "identifiability_pairwise",
                    "diagnosis_confusion_matrix",
                    "identifiability_quotient",
                )
            )

    def run_describability(self) -> None:
        """Run describability and repair profiles."""

        if self.observability is None:
            raise ValueError("Describability requires the observability profile")
        with self.state.profile_stage("describability"):
            describability = compute_describability_profiles(
                self.dataset,
                self.protocol,
                self.observability.observability,
                self.observability.event_summary,
                arrays=self.context.arrays,
            )
            repair = compute_repair_profiles(
                self.dataset,
                self.protocol,
                arrays=self.context.arrays,
            )
            self.state.tables.update(
                {
                    "description_profile": describability.description_profile,
                    "description_summary": describability.summary,
                    "description_stability": describability.description_stability,
                    "repair_profile": repair.repair_profile,
                }
            )
            self.state.publish_tables(
                (
                    "description_profile",
                    "description_summary",
                    "description_stability",
                    "repair_profile",
                )
            )

    def run_annotation_alignment(self) -> None:
        """Run annotation alignment and optional label table profiles."""

        with self.state.profile_stage(
            "annotation_alignment",
            label_export=self.label_export,
            label_tables_requested=self.label_export == "full",
            label_table_count=(
                len(LABEL_TABLE_KEYS) if self.label_export == "full" else 0
            ),
        ):
            annotation = compute_annotation_profiles(
                self.dataset,
                self.protocol,
                cache=self.context.cache,
                n_jobs=self.context.n_jobs,
                emit_label_tables=self.label_export == "full",
            )
            if self.label_export == "full":
                for key in LABEL_TABLE_KEYS:
                    self.state.tables[key] = annotation.label_tables[key]
            self.state.tables.update(
                {
                    "annotation_alignment": annotation.alignment,
                    "annotation_robustness": annotation.robustness,
                }
            )
            publish_keys = (
                tuple(LABEL_TABLE_KEYS) if self.label_export == "full" else ()
            ) + ("annotation_alignment", "annotation_robustness")
            self.state.publish_tables(publish_keys)
            write_json(
                self.output_dir / "annotation_channel_manifest.json",
                annotation.manifest,
            )
            self.state.extra_artifacts["annotation_channel_manifest"] = (
                "annotation_channel_manifest.json"
            )

    def run_admission(self) -> None:
        """Run provisional admission profiles."""

        missing = missing_required_tables(
            self.state.tables,
            (
                "event_summary",
                "arity",
                "implementation_validity",
                "detector_attribution",
                "corrected_detectability_frontier",
            ),
        )
        if missing:
            raise ValueError(f"Admission requires prerequisite tables: {missing}")
        with self.state.profile_stage("admission"):
            admission = compute_admission_profiles(
                self.dataset,
                self.protocol,
                event_summary=self.state.tables["event_summary"],
                arity=self.state.tables["arity"],
                implementation_validity=self.state.tables["implementation_validity"],
                detector_attribution=self.state.tables["detector_attribution"],
                corrected_detectability=self.state.tables[
                    "corrected_detectability_frontier"
                ],
                repair_profile=self.state.tables.get("repair_profile"),
                description_summary=self.state.tables.get("description_summary"),
                annotation_robustness=self.state.tables.get("annotation_robustness"),
                model_zoo_frontier=self.state.tables.get("model_zoo_frontier"),
                policy=self.admission_policy,
            )
            self.state.tables.update(
                {
                    "admission_events": admission.events,
                    "admission_variants": admission.variants,
                }
            )
            self.state.publish_tables(("admission_events", "admission_variants"))
            write_json(
                self.output_dir / "admission_policy_evaluation.json",
                admission.evaluation,
            )
            write_json(
                self.output_dir / "release_summary.json", admission.release_summary
            )
            (self.output_dir / "release_summary.md").write_text(
                admission.release_summary_markdown,
                encoding="utf-8",
            )
            self.state.extra_artifacts.update(
                {
                    "admission_policy_evaluation": "admission_policy_evaluation.json",
                    "release_summary": "release_summary.json",
                    "release_summary_markdown": "release_summary.md",
                }
            )

    def run_visual_audit(self) -> None:
        """Run visual audit selection and gallery output."""

        if "event_summary" not in self.state.tables:
            raise ValueError("Visual audit requires the event summary table")
        with self.state.profile_stage("visual_audit"):
            visual = compute_visual_audit(
                self.dataset,
                self.protocol,
                output_dir=self.output_dir,
                event_summary=self.state.tables["event_summary"],
                admission_events=self.state.tables.get("admission_events"),
                implementation_validity=self.state.tables.get(
                    "implementation_validity"
                ),
                detector_attribution=self.state.tables.get("detector_attribution"),
                boundary_audit=self.state.tables.get("boundary_audit"),
                shortcut_audit=self.state.tables.get("shortcut_audit"),
                arrays=self.context.arrays,
            )
            self.state.tables["visual_audit_selection"] = visual.selection
            self.state.publish_tables(("visual_audit_selection",))
            write_json(
                self.output_dir / "visual_audit" / "visual_audit_manifest.json",
                visual.manifest,
            )
            self.state.extra_artifacts.update(
                {
                    "visual_audit_index": "visual_audit/index.md",
                    "visual_audit_html_index": "visual_audit/index.html",
                    "visual_audit_manifest": "visual_audit/visual_audit_manifest.json",
                }
            )

    def run_generator_feedback(self) -> None:
        """Write generator-feedback artifacts after admission profiles."""

        with self.state.profile_stage("generator_feedback"):
            write_generator_feedback(self.output_dir)
            self.state.extra_artifacts.update(
                {
                    "generator_feedback_report": "generator_feedback_report.json",
                    "generator_feedback_report_markdown": "generator_feedback_report.md",
                }
            )

    def _requested(self, profile_name: str) -> bool:
        return profile_name in self.context.profile_names


def missing_required_tables(
    tables: Mapping[str, object],
    required: tuple[str, ...],
) -> list[str]:
    """Return required table keys missing from a mutable table registry."""

    return [key for key in required if key not in tables]


__all__ = [
    "CapabilityProfileExecutor",
    "missing_required_tables",
]
