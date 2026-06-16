import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis
from gutenTAG.tsgen.capabilities.dataset import (
    DatasetIndex,
    InstanceRecord,
)
from gutenTAG.tsgen.capabilities.release import write_release_certificate
from gutenTAG.tsgen.capabilities.visualization import compute_visual_audit

from tests.capability_execution_fixtures import (
    _relation_event_group,
    _small_generation_config,
)

ADMISSION_PROFILE_SEQUENCE = [
    "observability",
    "detectability",
    "corrected_detectability",
    "implementation_validity",
    "identifiability",
    "describability",
    "annotation_alignment",
    "admission",
]
ADMISSION_OUTPUT_FILES = (
    "admission_events.csv",
    "admission_variants.csv",
    "admission_policy_evaluation.json",
    "release_summary.json",
    "release_summary.md",
    "generator_feedback_report.json",
    "generator_feedback_report.md",
)
OBSERVATION_POLICY_YAML = (
    "admission_policy_version: synthgen.admission.test",
    "mode: observation_mode",
    "numeric_gates:",
    "  enforcement: observation_mode",
    "  promote_after_full_runs: 5",
    "  max_boundary_artifact_fail_share:",
    "    value: 1.0",
    "    status: observation",
    "  max_debug_event_share:",
    "    value: 1.0",
    "    status: observation",
    "  max_needs_repair_share:",
    "    value: 1.0",
    "    status: observation",
    "  min_canonical_observable_share:",
    "    value: 0.0",
    "    status: observation",
    "  min_oracle_annotation_pass_rate:",
    "    value: 0.0",
    "    status: observation",
    "",
)


def _fast_capability_protocol() -> CapabilityProtocol:
    return CapabilityProtocol(
        delta_grid=(0.20,),
        alpha_grid=(0.10,),
        max_scan_windows_per_length=8,
        bootstrap_samples=0,
    )


def _run_small_capability_profile(
    dataset_root: Path,
    output_dir: Path,
    *,
    profile_preset: str,
    **kwargs: Any,
) -> dict[str, Any]:
    TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()
    return run_capability_analysis(
        dataset_root=dataset_root,
        output_dir=output_dir,
        protocol=_fast_capability_protocol(),
        profile_preset=profile_preset,
        **kwargs,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_output_files_exist(
    testcase: unittest.TestCase,
    output_dir: Path,
    filenames: tuple[str, ...],
) -> None:
    for filename in filenames:
        testcase.assertTrue((output_dir / filename).exists(), filename)


def _load_admission_outputs(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        pd.read_csv(output_dir / "admission_events.csv"),
        pd.read_csv(output_dir / "admission_variants.csv"),
        _load_json(output_dir / "admission_policy_evaluation.json"),
        _load_json(output_dir / "generator_feedback_report.json"),
        _load_json(output_dir / "manifests" / "profile_run_manifest.json"),
    )


def _assert_admission_tables_and_policy(
    testcase: unittest.TestCase,
    events: pd.DataFrame,
    variants: pd.DataFrame,
    evaluation: dict[str, Any],
    feedback: dict[str, Any],
) -> None:
    testcase.assertIn("admission_status", events.columns)
    testcase.assertIn("failure_reasons", events.columns)
    testcase.assertIn("provisional_gate_hits", variants.columns)
    testcase.assertEqual(evaluation["mode"], "provisional")
    testcase.assertIn("recommendations", feedback)
    testcase.assertFalse(events.empty)
    testcase.assertFalse(variants.empty)


def _assert_profile_runtime_records(
    testcase: unittest.TestCase,
    certificate: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    runtime = manifest["runtime"]
    testcase.assertGreater(runtime["total_elapsed_seconds"], 0.0)
    testcase.assertGreaterEqual(runtime["output_dir_size_bytes"], 1)
    profile_records = {record["profile"]: record for record in runtime["profiles"]}
    for profile in certificate["profiles"]:
        testcase.assertIn(profile, profile_records)
        testcase.assertEqual(profile_records[profile]["status"], "complete")
        testcase.assertGreaterEqual(profile_records[profile]["elapsed_seconds"], 0.0)
    testcase.assertIn("generator_feedback", profile_records)
    testcase.assertIn(
        "observability", profile_records["observability"]["written_tables"]
    )
    testcase.assertGreater(
        profile_records["observability"]["table_rows"]["observability"], 0
    )
    testcase.assertEqual(
        profile_records["annotation_alignment"]["details"]["label_export"],
        "full",
    )


def _write_observation_policy(path: Path) -> None:
    path.write_text("\n".join(OBSERVATION_POLICY_YAML), encoding="utf-8")


def _assert_observation_policy_outputs(
    testcase: unittest.TestCase,
    output_dir: Path,
) -> None:
    evaluation = _load_json(output_dir / "admission_policy_evaluation.json")
    events = pd.read_csv(output_dir / "admission_events.csv")
    variants = pd.read_csv(output_dir / "admission_variants.csv")
    manifest = _load_json(output_dir / "manifests" / "profile_run_manifest.json")

    testcase.assertEqual(
        evaluation["admission_policy_version"], "synthgen.admission.test"
    )
    testcase.assertEqual(evaluation["mode"], "observation_mode")
    testcase.assertEqual(evaluation["numeric_gate_enforcement"], "observation_mode")
    testcase.assertEqual(
        evaluation["policy"]["numeric_gates"]["promote_after_full_runs"], 5
    )
    testcase.assertEqual(
        evaluation["policy"]["numeric_gates"]["max_debug_event_share"]["value"], 1.0
    )
    testcase.assertEqual(
        evaluation["policy"]["numeric_gates"]["max_debug_event_share"]["status"],
        "observation",
    )
    testcase.assertEqual(
        evaluation["policy"]["numeric_gates"]["max_needs_repair_share"]["value"], 1.0
    )
    testcase.assertEqual(
        evaluation["policy"]["numeric_gates"]["max_needs_repair_share"]["status"],
        "observation",
    )
    testcase.assertEqual(set(events["policy_mode"].astype(str)), {"observation_mode"})
    testcase.assertEqual(set(variants["policy_mode"].astype(str)), {"observation_mode"})
    testcase.assertIn("admission_policy_hash", manifest["fingerprints"])


def _relation_visual_instance_dir(root: Path) -> Path:
    return (
        root
        / "dataset"
        / "variants"
        / "rmj__mode-correlation__p00"
        / "train"
        / "instances"
        / "instance_000"
    )


def _write_relation_visual_arrays(instance_dir: Path) -> None:
    instance_dir.mkdir(parents=True)
    time = np.linspace(0.0, 4.0 * np.pi, 160)
    clean = np.column_stack([np.sin(time), np.sin(time) + 0.05 * np.cos(3 * time)])
    anomalous = clean.copy()
    anomalous[60:100, 1] = -anomalous[60:100, 1]
    pd.DataFrame(clean, columns=["value-0", "value-1"]).to_csv(
        instance_dir / "clean.csv",
        index=False,
    )
    pd.DataFrame(anomalous, columns=["value-0", "value-1"]).to_csv(
        instance_dir / "anomalous.csv",
        index=False,
    )


def _relation_visual_dataset(root: Path, instance_dir: Path) -> DatasetIndex:
    group = _relation_event_group("0", 60, 100)
    instance = InstanceRecord(
        dataset_root=root / "dataset",
        variant_id="rmj__mode-correlation__p00",
        split="train",
        instance_id="instance_000",
        instance_dir=instance_dir,
        clean_path=instance_dir / "clean.csv",
        anomalous_path=instance_dir / "anomalous.csv",
        events_path=instance_dir / "events.json",
        summary_path=instance_dir / "instance_summary.json",
        base_oscillation="rmj",
        anomaly_type="mode-correlation",
        channels=2,
        length=160,
        event_groups=(group,),
    )
    return DatasetIndex(root=root / "dataset", manifest={}, instances=(instance,))


def _relation_visual_event_summary(instance: InstanceRecord) -> pd.DataFrame:
    group = instance.event_groups[0]
    return pd.DataFrame(
        [
            {
                "event_id": "rmj__mode-correlation__p00/train/instance_000/g0",
                "variant_id": instance.variant_id,
                "split": instance.split,
                "instance_id": instance.instance_id,
                "base_oscillation": instance.base_oscillation,
                "anomaly_type": group.anomaly_type,
                "constraint_tag": group.constraint_tag,
                "semantic_scope": group.semantic_scope,
                "start": group.start,
                "end": group.end,
                "length": group.length,
                "group_channels": "0|1",
                "best_canonical_distance": 2.5,
            }
        ]
    )


def _assert_relation_visual_panels_written(
    testcase: unittest.TestCase,
    output_dir: Path,
    result: Any,
) -> None:
    testcase.assertTrue((output_dir / "visual_audit" / "index.html").exists())
    written = result.selection[result.selection["plot_status"] == "written"]
    testcase.assertFalse(written.empty)
    first = written.iloc[0]
    for column in (
        "relation_scatter_path",
        "rolling_correlation_path",
        "pca_residual_path",
    ):
        panel_path = output_dir / "visual_audit" / str(first[column])
        testcase.assertTrue(panel_path.exists(), panel_path)
    testcase.assertGreaterEqual(result.manifest["relation_scatter_count"], 1)
    testcase.assertGreaterEqual(result.manifest["rolling_correlation_count"], 1)
    testcase.assertGreaterEqual(result.manifest["pca_residual_count"], 1)
    testcase.assertIn("Visual Audit", result.index_html)


class TestCapabilityExecutionProfiles(unittest.TestCase):
    def test_run_can_write_p4_diagnosis_and_repair_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            config = _small_generation_config(dataset_root)
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="maturity",
            )

            self.assertEqual(
                certificate["profiles"],
                ["observability", "identifiability", "describability"],
            )
            for filename in (
                "diagnosis_confusion_matrix.csv",
                "identifiability_quotient.csv",
                "identifiability_summary.csv",
                "repair_profile.csv",
                "description_stability.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            diagnosis = pd.read_csv(output_dir / "diagnosis_confusion_matrix.csv")
            quotient = pd.read_csv(output_dir / "identifiability_quotient.csv")
            summary = pd.read_csv(output_dir / "identifiability_summary.csv")
            repair = pd.read_csv(output_dir / "repair_profile.csv")
            stability = pd.read_csv(output_dir / "description_stability.csv")
            self.assertIn("predicted_label", diagnosis.columns)
            self.assertIn("mean_local_impurity", quotient.columns)
            self.assertIn("descriptor_diagnosis_loss", summary.columns)
            self.assertIn("repair_gain", repair.columns)
            self.assertIn("description_stability_status", stability.columns)
            self.assertFalse(repair.empty)

    def test_run_can_write_p5_annotation_channel_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="annotation",
            )

            self.assertEqual(certificate["profiles"], ["annotation_alignment"])
            expected_label_files = (
                "labels_oracle_any.csv",
                "labels_oracle_intervention.csv",
                "labels_oracle_context.csv",
                "labels_event_only.csv",
                "labels_delayed.csv",
                "labels_weak_point.csv",
                "labels_visible_only.csv",
                "labels_noisy_boundary.csv",
                "labels_censored.csv",
            )
            for filename in expected_label_files:
                self.assertTrue((output_dir / "labels" / filename).exists(), filename)
            for filename in (
                "annotation_channel_manifest.json",
                "annotation_alignment.csv",
                "annotation_robustness.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            oracle_any = pd.read_csv(output_dir / "labels" / "labels_oracle_any.csv")
            intervention = pd.read_csv(
                output_dir / "labels" / "labels_oracle_intervention.csv"
            )
            alignment = pd.read_csv(output_dir / "annotation_alignment.csv")
            robustness = pd.read_csv(output_dir / "annotation_robustness.csv")
            self.assertIn("label_any", oracle_any.columns)
            self.assertIn("label-0", intervention.columns)
            self.assertIn("annotation_channel", alignment.columns)
            self.assertIn("jaccard", alignment.columns)
            self.assertIn("robustness_status", robustness.columns)
            self.assertFalse(alignment.empty)

    def test_annotation_diagnostics_mode_skips_bulk_label_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="annotation",
                label_export="diagnostics",
            )

            self.assertEqual(certificate["profiles"], ["annotation_alignment"])
            self.assertFalse((output_dir / "labels" / "labels_oracle_any.csv").exists())
            self.assertTrue((output_dir / "annotation_alignment.csv").exists())
            self.assertTrue((output_dir / "annotation_robustness.csv").exists())
            manifest = json.loads(
                (output_dir / "annotation_channel_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["label_export"], "diagnostics")
            self.assertFalse(manifest["label_tables_emitted"])
            self.assertEqual(manifest["table_paths"], {})
            output_manifest = json.loads(
                (output_dir / "manifests" / "output_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            table_names = {record["name"] for record in output_manifest["tables"]}
            self.assertNotIn("labels_oracle_any", table_names)
            self.assertIn("annotation_alignment", table_names)
            self.assertIn("annotation_robustness", table_names)
            self.assertEqual(
                certificate["run_manifest"]["output"]["label_export"],
                "diagnostics",
            )

    def test_run_can_write_provisional_admission_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            certificate = _run_small_capability_profile(
                dataset_root=dataset_root,
                output_dir=output_dir,
                profile_preset="admission",
            )

            self.assertEqual(certificate["profiles"], ADMISSION_PROFILE_SEQUENCE)
            _assert_output_files_exist(self, output_dir, ADMISSION_OUTPUT_FILES)
            events, variants, evaluation, feedback, manifest = _load_admission_outputs(
                output_dir
            )
            _assert_admission_tables_and_policy(
                self,
                events,
                variants,
                evaluation,
                feedback,
            )
            _assert_profile_runtime_records(self, certificate, manifest)

    def test_run_uses_external_admission_policy_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            policy_path = tmp_path / "admission_policy.yaml"
            _write_observation_policy(policy_path)

            _run_small_capability_profile(
                dataset_root=dataset_root,
                output_dir=output_dir,
                profile_preset="admission",
                admission_policy_path=policy_path,
            )

            _assert_observation_policy_outputs(self, output_dir)

    def test_run_can_write_visual_audit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="visual_audit",
            )

            self.assertEqual(certificate["profiles"][-1], "visual_audit")
            self.assertTrue((output_dir / "visual_audit" / "index.md").exists())
            self.assertTrue((output_dir / "visual_audit" / "index.html").exists())
            self.assertTrue(
                (output_dir / "visual_audit" / "visual_audit_manifest.json").exists()
            )
            self.assertTrue((output_dir / "visual_audit_selection.csv").exists())
            selection = pd.read_csv(output_dir / "visual_audit_selection.csv")
            self.assertIn("bucket", selection.columns)
            self.assertIn("plot_path", selection.columns)
            written = selection[selection["plot_status"] == "written"]
            self.assertFalse(written.empty)
            first_plot = output_dir / "visual_audit" / str(written.iloc[0]["plot_path"])
            self.assertTrue(first_plot.exists(), first_plot)

            certificate = write_release_certificate(
                dataset_root=dataset_root,
                analysis_dir=output_dir,
                admission_policy_path=Path("protocols/v12/admission_policy.yaml"),
            )
            self.assertEqual(
                certificate["release_certificate_version"],
                "synthgen.release_certificate.v12.1",
            )
            self.assertEqual(certificate["certificate_status"], "complete")
            self.assertTrue((output_dir / "release_certificate.json").exists())
            self.assertTrue((output_dir / "release_certificate.md").exists())

    def test_visual_audit_writes_relation_panels_and_html_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "analysis"
            instance_dir = _relation_visual_instance_dir(root)
            _write_relation_visual_arrays(instance_dir)
            dataset = _relation_visual_dataset(root, instance_dir)
            instance = dataset.instances[0]

            result = compute_visual_audit(
                dataset,
                CapabilityProtocol(alpha_grid=(0.10,)),
                output_dir=output_dir,
                event_summary=_relation_visual_event_summary(instance),
            )

            _assert_relation_visual_panels_written(self, output_dir, result)


if __name__ == "__main__":
    unittest.main()
