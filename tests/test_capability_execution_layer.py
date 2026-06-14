import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
import gutenTAG.tsgen.capabilities.corrected_detectability as corrected_detectability_module
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis
from gutenTAG.tsgen.capabilities.admission import AdmissionPolicy, _event_admission, _variant_status
from gutenTAG.tsgen.capabilities.array_store import ArrayStore
from gutenTAG.tsgen.capabilities.corrected_detectability import compute_corrected_detectability_frontier
from gutenTAG.tsgen.capabilities.dataset import DatasetIndex, EventGroup, InstanceRecord, discover_dataset
from gutenTAG.tsgen.capabilities.diagnosis import (
    _feature_matrix as _diagnosis_feature_matrix,
    _nearest_centroid_predictions,
    compute_diagnosis_profiles,
)
from gutenTAG.tsgen.capabilities.execution import resolve_profile_names
from gutenTAG.tsgen.capabilities.identifiability import _leave_one_out_nearest_labels
from gutenTAG.tsgen.capabilities.model_zoo import compute_model_zoo_frontier
from gutenTAG.tsgen.capabilities.models import LowRankResidualModel, TargetRegressionResidualModel
from gutenTAG.tsgen.capabilities.output_store import ParquetOutputError
from gutenTAG.tsgen.capabilities.release import write_release_certificate
from gutenTAG.tsgen.capabilities.repair.operators import repair_segment
from gutenTAG.tsgen.capabilities.validation.boundary import compute_boundary_audit
from gutenTAG.tsgen.capabilities.validation.attribution import compute_detector_attribution
from gutenTAG.tsgen.capabilities.validation.implementation_validity import _implementation_validity
from gutenTAG.tsgen.capabilities.validation.negative_controls import _event_negative_controls
from gutenTAG.tsgen.capabilities.validation.realized_effects import compute_realized_effects
from gutenTAG.tsgen.capabilities.validation.support import compute_support_integrity
from gutenTAG.tsgen.capabilities.visualization import compute_visual_audit
from gutenTAG.tsgen.capabilities.windows import WindowLibrary, WindowSpec
from gutenTAG.tsgen.contracts import write_v12_metadata_registries


class TestCapabilityExecutionLayer(unittest.TestCase):
    def test_identifiability_leave_one_out_nearest_labels_excludes_self(self) -> None:
        matrix = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
        labels = np.asarray(["a", "a", "b", "b"], dtype=object)

        self.assertEqual(_leave_one_out_nearest_labels(matrix, labels), ["a", "a", "b", "b"])

        empty_features = np.zeros((3, 0), dtype=float)
        empty_labels = np.asarray(["x", "y", "z"], dtype=object)
        self.assertEqual(_leave_one_out_nearest_labels(empty_features, empty_labels), ["y", "x", "x"])

    def test_diagnosis_leave_group_out_centroid_predictions_are_vectorized(self) -> None:
        embeddings = pd.DataFrame(
            {
                "variant_id": ["v0", "v1", "v0", "v1"],
                "anomaly_type": ["a", "a", "b", "b"],
                "witness_mean_delta": [0.0, 0.1, 10.0, 10.1],
            }
        )
        matrix, _ = _diagnosis_feature_matrix(embeddings)

        predictions = _nearest_centroid_predictions(
            embeddings,
            matrix,
            descriptor="anomaly_type",
            holdout_column="variant_id",
        )

        self.assertEqual(predictions, ["a", "a", "b", "b"])

    def test_diagnosis_profiles_run_without_dense_pairwise_distance_matrix(self) -> None:
        embeddings = pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(8)],
                "variant_id": ["v0", "v1"] * 4,
                "base_oscillation": ["sine", "sine", "cosine", "cosine"] * 2,
                "anomaly_type": ["mean", "mean", "variance", "variance"] * 2,
                "constraint_tag": ["location.mean", "location.mean", "scale.variance", "scale.variance"] * 2,
                "operator_family": ["location", "location", "scale", "scale"] * 2,
                "support_type": ["segment"] * 8,
                "semantic_scope": ["channel"] * 8,
                "repair_operator": ["additive_offset", "additive_offset", "match_local_variance", "match_local_variance"] * 2,
                "witness_mean_delta": [0.0, 0.1, 0.2, 0.3, 10.0, 10.1, 10.2, 10.3],
                "witness_variance_delta": [10.0, 10.1, 10.2, 10.3, 0.0, 0.1, 0.2, 0.3],
            }
        )

        diagnosis = compute_diagnosis_profiles(embeddings, CapabilityProtocol())

        self.assertFalse(diagnosis.summary.empty)
        self.assertFalse(diagnosis.quotient.empty)
        self.assertIn("epsilon_quotient_impurity", diagnosis.summary.columns)

    def test_array_store_materializes_and_reuses_source_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__mean__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
            pd.DataFrame(values, columns=["ch_0", "ch_1"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(values + 1.0, columns=["ch_0", "ch_1"]).to_csv(instance_dir / "anomalous.csv", index=False)
            record = InstanceRecord(
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
                channels=2,
                length=2,
                event_groups=(),
            )

            store = ArrayStore(cache_dir=root / "cache", mmap=True)
            store.materialize((record,))
            clean = store.get(record, "clean")
            anomalous = store.get(record, "anomalous")

            self.assertTrue(
                (
                    root
                    / "cache"
                    / "arrays"
                    / "sine__mean__p00"
                    / "train"
                    / "instance_000"
                    / "clean.npy"
                ).exists()
            )
            self.assertTrue(np.allclose(np.asarray(clean), values))
            self.assertTrue(np.allclose(np.asarray(anomalous), values + 1.0))

    def test_support_integrity_excludes_other_known_event_supports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__mean__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            clean = np.zeros((100, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 1.0
            anomalous[70:80, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(instance_dir / "anomalous.csv", index=False)
            instance = InstanceRecord(
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
                length=100,
                event_groups=(
                    _event_group("0", 10, 20),
                    _event_group("1", 70, 80),
                ),
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            support = compute_support_integrity(dataset)

            self.assertEqual(set(support["support_status"]), {"valid"})
            self.assertTrue((support["inside_mass_share"] == 1.0).all())
            self.assertTrue((support["far_field_mass_share"] == 0.0).all())
            self.assertTrue((support["known_other_event_l2_mass"] == 10.0).all())
            self.assertTrue((support["known_other_event_mass_share"] == 0.5).all())

    def test_support_integrity_treats_near_field_residual_as_boundary_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__mean__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            clean = np.zeros((100, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 1.0
            anomalous[20:30, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(instance_dir / "anomalous.csv", index=False)
            instance = _support_instance_record(root, instance_dir, (_event_group("0", 10, 20),))
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            support = compute_support_integrity(dataset)

            self.assertEqual(support.iloc[0]["support_status"], "boundary_uncertain")
            self.assertAlmostEqual(float(support.iloc[0]["inside_mass_share"]), 0.5)
            self.assertAlmostEqual(float(support.iloc[0]["support_or_near_field_mass_share"]), 1.0)
            self.assertAlmostEqual(float(support.iloc[0]["far_field_mass_share"]), 0.0)

    def test_support_integrity_keeps_far_field_residual_leaky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__mean__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            clean = np.zeros((100, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 1.0
            anomalous[80:90, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(instance_dir / "anomalous.csv", index=False)
            instance = _support_instance_record(root, instance_dir, (_event_group("0", 10, 20),))
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            support = compute_support_integrity(dataset)

            self.assertEqual(support.iloc[0]["support_status"], "leaky")
            self.assertAlmostEqual(float(support.iloc[0]["inside_mass_share"]), 0.5)
            self.assertAlmostEqual(float(support.iloc[0]["support_or_near_field_mass_share"]), 0.5)
            self.assertAlmostEqual(float(support.iloc[0]["far_field_mass_share"]), 0.5)

    def test_boundary_audit_uses_source_edges_not_trimmed_effective_support_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__variance__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            clean = np.zeros((40, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[12:14, 0] = 1.0
            anomalous[16:18, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(instance_dir / "anomalous.csv", index=False)
            event = _event_group("0", 12, 18, source_start=10, source_end=20)
            instance = _support_instance_record(root, instance_dir, (event,), length=40)
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            boundary = compute_boundary_audit(dataset)

            row = boundary.iloc[0]
            self.assertEqual(row["boundary_status"], "valid_interior_or_mixed")
            self.assertEqual(int(row["boundary_audit_start"]), 10)
            self.assertEqual(int(row["boundary_audit_end"]), 20)
            self.assertAlmostEqual(float(row["boundary_energy_share"]), 0.0)

    def test_boundary_audit_keeps_true_source_edge_artifact_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__variance__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            clean = np.zeros((40, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:12, 0] = 1.0
            anomalous[18:20, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(instance_dir / "anomalous.csv", index=False)
            event = _event_group("0", 12, 18, source_start=10, source_end=20)
            instance = _support_instance_record(root, instance_dir, (event,), length=40)
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            boundary = compute_boundary_audit(dataset)

            row = boundary.iloc[0]
            self.assertEqual(row["boundary_status"], "boundary_primary_detection_cause")
            self.assertAlmostEqual(float(row["boundary_energy_share"]), 1.0)

    def test_mode_repair_restores_sign_flipped_regime_segment(self) -> None:
        base = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
        clean = np.column_stack([base, base])
        anomalous = np.column_stack([base, -base])

        repaired = repair_segment(
            clean,
            anomalous,
            constraint_tag="regime.mode_correlation",
            repair_operator="restore_regime_conditioned_dependence",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - clean))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - clean))))
        self.assertGreater(raw_rmse, 0.5)
        self.assertLess(repaired_rmse, raw_rmse * 0.05)

    def test_relation_repair_restores_sign_flipped_pair_segment(self) -> None:
        base = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
        clean = np.column_stack([base, base])
        anomalous = np.column_stack([base, -base])

        repaired = repair_segment(
            clean,
            anomalous,
            constraint_tag="dependence.correlation",
            repair_operator="match_local_correlation",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - clean))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - clean))))
        self.assertGreater(raw_rmse, 0.5)
        self.assertLess(repaired_rmse, raw_rmse * 0.05)

    def test_covariance_repair_operator_dispatch_ignores_variance_substring(self) -> None:
        base = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
        clean = np.column_stack([base, base])
        anomalous = np.column_stack([base, -base])

        repaired = repair_segment(
            clean,
            anomalous,
            constraint_tag="dependence.covariance",
            repair_operator="match_local_covariance",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - clean))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - clean))))
        self.assertGreater(raw_rmse, 0.5)
        self.assertLess(repaired_rmse, raw_rmse * 0.05)

    def test_variance_repair_uses_affine_fallback_when_rescale_is_insufficient(self) -> None:
        base = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
        clean = np.column_stack([base, base])
        anomalous = -2.0 * clean + 0.35

        repaired = repair_segment(
            clean,
            anomalous,
            constraint_tag="scale.variance",
            repair_operator="match_local_variance",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - clean))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - clean))))
        self.assertGreater(raw_rmse, 1.0)
        self.assertLess(repaired_rmse, raw_rmse * 0.05)

    def test_variance_repair_never_worsens_segment_rmse(self) -> None:
        base = np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=float)
        anomalous = np.asarray([[0.0], [1.1], [0.0], [1.1], [0.0], [0.9]], dtype=float)

        repaired = repair_segment(
            base,
            anomalous,
            constraint_tag="scale.variance",
            repair_operator="match_local_variance",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - base))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - base))))
        self.assertLessEqual(repaired_rmse, raw_rmse + 1e-12)

    def test_pattern_repair_aligns_shifted_template_segment(self) -> None:
        base = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
        clean = np.column_stack([base, 0.5 * base + 0.2])
        anomalous = np.empty_like(clean)
        anomalous[5:] = clean[:-5]
        anomalous[:5] = clean[0]
        anomalous = 1.4 * anomalous - 0.25

        repaired = repair_segment(
            clean,
            anomalous,
            constraint_tag="shape.local_template",
            repair_operator="match_nearest_local_template",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - clean))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - clean))))
        self.assertGreater(raw_rmse, 0.5)
        self.assertLess(repaired_rmse, raw_rmse * 0.20)

    def test_pattern_repair_never_worsens_segment_rmse(self) -> None:
        clean = np.asarray(
            [[0.0], [1.0], [0.0], [-1.0], [0.0], [1.0], [0.0]],
            dtype=float,
        )
        anomalous = np.asarray(
            [[0.0], [0.9], [0.2], [-0.7], [0.1], [1.1], [-0.1]],
            dtype=float,
        )

        repaired = repair_segment(
            clean,
            anomalous,
            constraint_tag="shape.local_template",
            repair_operator="match_nearest_local_template",
        )

        raw_rmse = float(np.sqrt(np.mean(np.square(anomalous - clean))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired - clean))))
        self.assertLessEqual(repaired_rmse, raw_rmse + 1e-12)

    def test_relation_negative_controls_allow_clean_marginal_carrier(self) -> None:
        clean = np.zeros((80, 2), dtype=float)
        carrier = np.tile([-1.0, 1.0], 5)
        clean[35:45, 0] = carrier
        clean[35:45, 1] = carrier
        anomalous = clean.copy()
        anomalous[35:45, 1] *= -1.0
        group = _relation_event_group("0", 35, 45)
        instance = _instance_record_for_groups((group,), channels=2, length=80)
        frontier = pd.DataFrame(
            [
                {
                    "event_id": "random-mode-jump__mode-correlation__p00/train/instance_000/g0",
                    "alpha": 0.10,
                    "scan_threshold": 1.0,
                }
            ]
        )
        protocol = CapabilityProtocol(
            detection_witnesses=(
                "mean_z",
                "variance_log_ratio",
                "local_energy_z",
                "correlation_shift",
            )
        )

        controls = pd.DataFrame(
            _event_negative_controls(
                instance=instance,
                group=group,
                clean=clean,
                anomalous=anomalous,
                frontier=frontier,
                boundary_row={},
                protocol=protocol,
            )
        )

        wrong_witness = controls[controls["control_type"] == "wrong_witness_control"].iloc[0]
        self.assertGreater(float(wrong_witness["control_score"]), 1.0)
        self.assertEqual(wrong_witness["control_status"], "control_passed")
        self.assertAlmostEqual(
            float(wrong_witness["control_score"]),
            float(wrong_witness["baseline_control_score"]),
        )

    def test_realized_effects_preserve_signed_relation_flip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "rmj__mode-correlation__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            base = np.sin(np.linspace(0.0, 6.0 * np.pi, 120))
            clean = np.column_stack([base, base])
            anomalous = clean.copy()
            anomalous[40:80, 1] *= -1.0
            pd.DataFrame(clean, columns=["value-0", "value-1"]).to_csv(
                instance_dir / "clean.csv",
                index=False,
            )
            pd.DataFrame(anomalous, columns=["value-0", "value-1"]).to_csv(
                instance_dir / "anomalous.csv",
                index=False,
            )
            group = _relation_event_group("0", 40, 80)
            instance = InstanceRecord(
                dataset_root=root,
                variant_id="rmj__mode-correlation__p00",
                split="train",
                instance_id="instance_000",
                instance_dir=instance_dir,
                clean_path=instance_dir / "clean.csv",
                anomalous_path=instance_dir / "anomalous.csv",
                events_path=instance_dir / "events.json",
                summary_path=instance_dir / "instance_summary.json",
                base_oscillation="random-mode-jump",
                anomaly_type="mode-correlation",
                channels=2,
                length=120,
                event_groups=(group,),
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            effects = compute_realized_effects(dataset)

            row = effects.iloc[0]
            self.assertGreater(float(row["corr_event_clean"]), 0.99)
            self.assertLess(float(row["corr_event_anomalous"]), -0.99)
            self.assertGreater(float(row["abs_corr_event_anomalous"]), 0.99)
            self.assertGreater(float(row["realized_fisher_shift"]), 5.0)
            self.assertLess(float(row["anomalous_mode_agreement"]), -0.99)
            self.assertLess(float(row["target_vs_ensemble_corr_event"]), -0.99)

    def test_wrong_support_control_avoids_other_declared_events(self) -> None:
        clean = np.zeros((80, 1), dtype=float)
        anomalous = clean.copy()
        anomalous[20:30, 0] = 10.0
        current = _event_group("0", 10, 20)
        other = _event_group("1", 20, 30)
        instance = _instance_record_for_groups((current, other), channels=1, length=80)
        frontier = pd.DataFrame(
            [
                {
                    "event_id": "random-mode-jump__mode-correlation__p00/train/instance_000/g0",
                    "alpha": 0.10,
                    "scan_threshold": 1.0,
                }
            ]
        )

        controls = pd.DataFrame(
            _event_negative_controls(
                instance=instance,
                group=current,
                clean=clean,
                anomalous=anomalous,
                frontier=frontier,
                boundary_row={},
                protocol=CapabilityProtocol(),
            )
        )

        wrong_support = controls[controls["control_type"] == "wrong_support_control"].iloc[0]
        self.assertEqual(wrong_support["control_status"], "control_passed")
        self.assertLess(float(wrong_support["control_score"]), 1.0)

    def test_detector_attribution_downgrades_shortcut_when_canonical_candidate_also_detected(self) -> None:
        group = EventGroup(
            group_id="0",
            start=10,
            end=20,
            source_start=10,
            source_end=20,
            anomaly_type="variance",
            constraint_tag="scale.variance",
            repair_operator="match_local_variance",
            semantic_scope="channel",
            intervention_channels=(0,),
            context_channels=(0,),
            group_channels=(0,),
            primary_channels=(0,),
            event_scope="unit_test",
            purity_hint="pure",
            raw_events=(),
        )
        instance = InstanceRecord(
            dataset_root=Path("/tmp/synth-gen-capability-test"),
            variant_id="sine__variance__p00",
            split="train",
            instance_id="instance_000",
            instance_dir=Path("/tmp/synth-gen-capability-test"),
            clean_path=Path("/tmp/synth-gen-capability-test/clean.csv"),
            anomalous_path=Path("/tmp/synth-gen-capability-test/anomalous.csv"),
            events_path=Path("/tmp/synth-gen-capability-test/events.json"),
            summary_path=Path("/tmp/synth-gen-capability-test/instance_summary.json"),
            base_oscillation="sine",
            anomaly_type="variance",
            channels=1,
            length=80,
            event_groups=(group,),
        )
        dataset = DatasetIndex(root=instance.dataset_root, manifest={}, instances=(instance,))
        frontier = pd.DataFrame(
            [
                {
                    "event_id": "sine__variance__p00/train/instance_000/g0",
                    "variant_id": "sine__variance__p00",
                    "split": "train",
                    "instance_id": "instance_000",
                    "anomaly_type": "variance",
                    "constraint_tag": "scale.variance",
                    "semantic_scope": "channel",
                    "alpha": 0.01,
                    "family": "location.mean",
                    "witness_or_model": "mean_z",
                    "projection": "0",
                    "raw_score": 5.0,
                    "candidate_level_p_value": 0.001,
                    "scan_level_p_value": 0.001,
                    "scan_threshold": 3.0,
                    "detected": True,
                    "scan_null_count": 1000,
                    "best_canonical_witness": "variance_log_ratio@0",
                    "best_canonical_candidate_p_value": 0.01,
                    "best_canonical_detected": True,
                }
            ]
        )
        boundary = pd.DataFrame(
            [
                {
                    "event_id": "sine__variance__p00/train/instance_000/g0",
                    "boundary_primary_detection_cause": False,
                }
            ]
        )
        shortcut = pd.DataFrame(
            [
                {
                    "event_id": "sine__variance__p00/train/instance_000/g0",
                    "shortcut_status": "valid_no_shortcut",
                }
            ]
        )

        attribution = compute_detector_attribution(
            dataset,
            frontier,
            boundary,
            shortcut,
        )

        row = attribution.iloc[0]
        self.assertEqual(row["primary_detection_cause"], "valid_with_shortcut")
        self.assertTrue(bool(row["best_canonical_detected"]))

    def test_implementation_validity_uses_detector_attribution_for_shortcuts(self) -> None:
        support = pd.DataFrame(
            [
                {
                    "event_id": "variant/test/instance_000/g0",
                    "variant_id": "variant",
                    "split": "test",
                    "instance_id": "instance_000",
                    "anomaly_type": "correlation-flip",
                    "constraint_tag": "dependence.correlation",
                    "semantic_scope": "relation",
                    "support_status": "valid",
                },
                {
                    "event_id": "variant/test/instance_000/g1",
                    "variant_id": "variant",
                    "split": "test",
                    "instance_id": "instance_000",
                    "anomaly_type": "correlation-flip",
                    "constraint_tag": "dependence.correlation",
                    "semantic_scope": "relation",
                    "support_status": "valid",
                },
            ]
        )
        boundary = pd.DataFrame(
            [
                {
                    "event_id": "variant/test/instance_000/g0",
                    "boundary_status": "valid_interior_or_mixed",
                },
                {
                    "event_id": "variant/test/instance_000/g1",
                    "boundary_status": "valid_interior_or_mixed",
                },
            ]
        )
        shortcut = pd.DataFrame(
            [
                {
                    "event_id": "variant/test/instance_000/g0",
                    "shortcut_status": "shortcut_dominated",
                },
                {
                    "event_id": "variant/test/instance_000/g1",
                    "shortcut_status": "shortcut_dominated",
                },
            ]
        )
        attribution = pd.DataFrame(
            [
                {
                    "event_id": "variant/test/instance_000/g0",
                    "alpha": 0.01,
                    "rank_within_event": 1,
                    "primary_detection_cause": "valid_with_shortcut",
                    "best_canonical_detected": True,
                },
                {
                    "event_id": "variant/test/instance_000/g1",
                    "alpha": 0.01,
                    "rank_within_event": 1,
                    "primary_detection_cause": "detected_wrong_reason",
                    "best_canonical_detected": False,
                },
            ]
        )

        validity = _implementation_validity(
            support=support,
            boundary=boundary,
            shortcut=shortcut,
            realized=pd.DataFrame(),
            negative_controls=pd.DataFrame(),
            detector_attribution=attribution,
        ).set_index("event_id")

        self.assertEqual(
            validity.loc["variant/test/instance_000/g0", "implementation_validity_status"],
            "valid_candidate",
        )
        self.assertEqual(
            validity.loc["variant/test/instance_000/g0", "detector_primary_detection_cause"],
            "valid_with_shortcut",
        )
        self.assertTrue(
            bool(validity.loc["variant/test/instance_000/g0", "detector_best_canonical_detected"])
        )
        self.assertEqual(
            validity.loc["variant/test/instance_000/g1", "implementation_validity_status"],
            "detected_wrong_reason",
        )

    def test_variant_status_honors_needs_repair_share_gate(self) -> None:
        policy = AdmissionPolicy(max_needs_repair_share=0.05)
        at_gate = pd.DataFrame({"admission_status": ["release"] * 95 + ["needs_repair"] * 5})
        over_gate = pd.DataFrame({"admission_status": ["release"] * 94 + ["needs_repair"] * 6})

        self.assertEqual(
            _variant_status(
                at_gate,
                failures=[],
                warnings=["support_leakage_suspected"],
                gate_hits=[],
                policy=policy,
            ),
            "release_with_warning",
        )
        self.assertEqual(
            _variant_status(
                over_gate,
                failures=[],
                warnings=["support_leakage_suspected"],
                gate_hits=[],
                policy=policy,
            ),
            "needs_repair",
        )

    def test_variant_status_honors_debug_share_gate(self) -> None:
        policy = AdmissionPolicy(max_debug_event_share=0.05)
        at_gate = pd.DataFrame({"admission_status": ["release"] * 95 + ["debug"] * 5})
        over_gate = pd.DataFrame({"admission_status": ["release"] * 94 + ["debug"] * 6})

        self.assertEqual(
            _variant_status(
                at_gate,
                failures=[],
                warnings=["not_detected_at_min_alpha"],
                gate_hits=[],
                policy=policy,
            ),
            "release_with_warning",
        )
        self.assertEqual(
            _variant_status(
                over_gate,
                failures=[],
                warnings=["not_detected_at_min_alpha"],
                gate_hits=[],
                policy=policy,
            ),
            "debug",
        )

    def test_event_admission_accepts_model_zoo_only_detection_with_warning(self) -> None:
        event_id = "variant/test/instance_000/g0"
        protocol = CapabilityProtocol(alpha_grid=(0.01,), delta_grid=(0.20,))

        events = _event_admission(
            protocol=protocol,
            policy=AdmissionPolicy(),
            event_summary=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "variant_id": "variant",
                        "split": "test",
                        "instance_id": "instance_000",
                        "anomaly_type": "pattern",
                        "constraint_tag": "shape.local_template",
                        "semantic_scope": "channel",
                    }
                ]
            ),
            arity=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "delta": 0.20,
                        "canonical_is_observable_at_delta": True,
                        "canonical_observed_arity": 1,
                    }
                ]
            ),
            implementation_validity=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "implementation_validity_status": "valid_candidate",
                        "support_status": "valid",
                        "boundary_status": "valid_interior_or_mixed",
                        "shortcut_status": "valid_no_shortcut",
                        "realized_offset": 0.0,
                    }
                ]
            ),
            detector_attribution=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "alpha": 0.01,
                        "rank_within_event": 1,
                        "primary_detection_cause": "observable_not_detected",
                    }
                ]
            ),
            corrected_detectability=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "alpha": 0.01,
                        "detected": False,
                        "scan_level_p_value": 0.50,
                        "calibration_status": "calibration_ok",
                    }
                ]
            ),
            repair_profile=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "repair_status": "repair_effective",
                        "repair_gain": 0.80,
                    }
                ]
            ),
            model_zoo_frontier=pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "alpha": 0.01,
                        "detected": True,
                    }
                ]
            ),
            metadata={
                event_id: {
                    "genotype_id": "genotype:variant",
                    "contract_id": "synthgen.contract.pattern.v1",
                }
            },
        )

        row = events.iloc[0]
        self.assertEqual(row["admission_status"], "release_with_warning")
        self.assertFalse(bool(row["detected_at_min_alpha"]))
        self.assertTrue(bool(row["detected_for_admission"]))
        self.assertEqual(row["admission_detection_source"], "model_zoo")
        self.assertIn("not_detected_at_min_alpha", row["warning_reasons"])
        self.assertIn("observable_not_detected", row["warning_reasons"])
        self.assertIn("model_zoo_only_detection", row["warning_reasons"])
        self.assertTrue(bool(row["manual_review_recommended"]))

    def test_window_library_reuses_deterministic_specs(self) -> None:
        library = WindowLibrary()
        spec = WindowSpec(
            series_length=100,
            window_length=10,
            max_windows=8,
            stride_fraction=0.25,
        )

        first = library.get(spec)
        second = library.get(spec)

        self.assertIs(first, second)
        self.assertEqual(first.shape[1], 2)
        self.assertLessEqual(len(first), 8)
        self.assertTrue((first[:, 1] > first[:, 0]).all())

    def test_profile_resolution_adds_required_observability_dependency(self) -> None:
        self.assertEqual(resolve_profile_names(profiles=("identifiability",)), ("observability", "identifiability"))
        self.assertEqual(
            resolve_profile_names(profiles=("all",)),
            (
                "observability",
                "detectability",
                "corrected_detectability",
                "implementation_validity",
                "model_zoo",
                "law_observability",
                "identifiability",
                "describability",
                "annotation_alignment",
                "admission",
                "visual_audit",
            ),
        )
        self.assertEqual(
            resolve_profile_names(profile_preset="implementation_validity"),
            (
                "observability",
                "detectability",
                "corrected_detectability",
                "implementation_validity",
            ),
        )
        self.assertEqual(
            resolve_profile_names(profile_preset="calibration"),
            ("detectability", "corrected_detectability"),
        )
        self.assertEqual(resolve_profile_names(profile_preset="model_zoo"), ("model_zoo",))
        self.assertEqual(
            resolve_profile_names(profile_preset="law_observability"),
            ("law_observability",),
        )
        self.assertEqual(
            resolve_profile_names(profile_preset="diagnosis"),
            ("observability", "identifiability"),
        )
        self.assertEqual(
            resolve_profile_names(profile_preset="repair"),
            ("observability", "describability"),
        )
        self.assertEqual(
            resolve_profile_names(profile_preset="maturity"),
            ("observability", "identifiability", "describability"),
        )
        self.assertEqual(resolve_profile_names(profile_preset="annotation"), ("annotation_alignment",))
        self.assertEqual(resolve_profile_names(profile_preset="annotation_channels"), ("annotation_alignment",))
        self.assertEqual(
            resolve_profile_names(profile_preset="admission"),
            (
                "observability",
                "detectability",
                "corrected_detectability",
                "implementation_validity",
                "identifiability",
                "describability",
                "annotation_alignment",
                "admission",
            ),
        )
        self.assertEqual(
            resolve_profile_names(profile_preset="visual_audit"),
            (
                "observability",
                "detectability",
                "corrected_detectability",
                "implementation_validity",
                "identifiability",
                "describability",
                "annotation_alignment",
                "admission",
                "visual_audit",
            ),
        )

    def test_run_can_materialize_arrays_and_write_observability_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            cache_dir = tmp_path / "cache"
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
                profiles=("observability",),
                cache_dir=cache_dir,
                materialize_arrays=True,
            )

            self.assertEqual(certificate["profiles"], ["observability"])
            self.assertTrue((output_dir / "observability_profile.csv").exists())
            self.assertTrue((output_dir / "tables_csv" / "observability_profile.csv").exists())
            self.assertTrue((output_dir / "event_capability_summary.csv").exists())
            self.assertFalse((output_dir / "detectability_frontier.csv").exists())
            self.assertTrue(list((cache_dir / "arrays").glob("**/*.npy")))
            manifest = json.loads((output_dir / "manifests" / "output_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["output_manifest_version"], "synthgen.capability.output.v2")
            self.assertEqual(manifest["layout"]["release_csv_dir"], "tables_csv")
            table_names = {record["name"] for record in manifest["tables"]}
            self.assertIn("observability", table_names)
            self.assertIn("event_summary", table_names)
            observability_record = next(record for record in manifest["tables"] if record["name"] == "observability")
            self.assertEqual(observability_record["canonical"]["path"], "tables_csv/observability_profile.csv")
            self.assertEqual(observability_record["release_csv"]["path"], "tables_csv/observability_profile.csv")
            self.assertEqual(observability_record["legacy_csv"]["path"], "observability_profile.csv")
            hashes = json.loads((output_dir / "manifests" / "table_hashes.json").read_text(encoding="utf-8"))
            self.assertIn("tables_csv/observability_profile.csv", hashes["file_hashes"])
            self.assertIn("observability", hashes["table_content_hashes"])

    def test_run_parquet_output_fails_fast_without_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )
            if not _parquet_engine_available():
                with self.assertRaises(ParquetOutputError):
                    run_capability_analysis(
                        dataset_root=dataset_root,
                        output_dir=output_dir,
                        protocol=protocol,
                        profiles=("observability",),
                        output_format="parquet",
                    )
                return

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=protocol,
                profiles=("observability",),
                output_format="parquet",
            )

            self.assertEqual(certificate["profiles"], ["observability"])
            self.assertTrue((output_dir / "tables_csv" / "observability_profile.csv").exists())
            manifest = json.loads((output_dir / "manifests" / "output_manifest.json").read_text(encoding="utf-8"))
            observability_record = next(record for record in manifest["tables"] if record["name"] == "observability")
            canonical = observability_record["canonical"]
            self.assertEqual(canonical["format"], "parquet")
            self.assertTrue((output_dir / canonical["path"]).exists())
            self.assertTrue(canonical["path"].startswith("tables_parquet/"))

    def test_run_parquet_output_allows_explicit_fallback(self) -> None:
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
                profiles=("observability",),
                output_format="parquet",
                allow_parquet_fallback=True,
            )

            self.assertEqual(certificate["profiles"], ["observability"])
            self.assertTrue((output_dir / "observability_profile.csv").exists())
            self.assertTrue((output_dir / "tables_csv" / "observability_profile.csv").exists())
            manifest = json.loads((output_dir / "manifests" / "output_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["allow_parquet_fallback"])
            observability_record = next(record for record in manifest["tables"] if record["name"] == "observability")
            canonical = observability_record["canonical"]
            self.assertIn(canonical["format"], {"parquet", "csv.gz"})
            self.assertTrue((output_dir / canonical["path"]).exists())
            if canonical["format"] == "parquet":
                self.assertTrue(canonical["path"].startswith("tables_parquet/"))
            else:
                self.assertTrue(canonical["path"].startswith("tables_csv/"))
                self.assertIn("parquet_fallback", {warning["code"] for warning in manifest["warnings"]})

    def test_run_can_write_model_zoo_outputs(self) -> None:
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
                profile_preset="model_zoo",
            )

            self.assertEqual(certificate["profiles"], ["model_zoo"])
            for filename in (
                "model_zoo_frontier.csv",
                "model_zoo_event_scores.csv",
                "model_zoo_model_manifest.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            frontier = pd.read_csv(output_dir / "model_zoo_frontier.csv")
            self.assertIn("model_id", frontier.columns)
            self.assertIn("scan_statistic", frontier.columns)
            self.assertIn("scan_level_p_value", frontier.columns)
            self.assertIn("model_metadata_hash", frontier.columns)
            self.assertFalse(frontier.empty)
            self.assertTrue(np.isfinite(frontier["scan_statistic"]).all())

    def test_target_regression_scores_non_last_channel_break(self) -> None:
        base = np.linspace(-1.0, 1.0, 80)
        clean = np.column_stack([base, 2.0 * base + 0.1, -0.5 * base + 0.2])
        anomalous = clean.copy()
        anomalous[30:50, 1] *= -1.0

        model = TargetRegressionResidualModel()
        model.fit([clean])

        clean_score = float(model.score_windows(clean, np.asarray([[30, 50]], dtype=int))[0])
        anomalous_score = float(model.score_windows(anomalous, np.asarray([[30, 50]], dtype=int))[0])
        self.assertTrue(np.isfinite(clean_score))
        self.assertGreater(anomalous_score, clean_score * 1000.0)

    def test_lowrank_residual_keeps_residual_dimension_for_two_channel_projection(self) -> None:
        clean = np.column_stack(
            [
                np.linspace(-1.0, 1.0, 80),
                np.linspace(-1.0, 1.0, 80) * 0.5,
            ]
        )

        model = LowRankResidualModel(rank=2)
        model.fit([clean])

        self.assertEqual(model.metadata()["effective_rank"], 1)

    def test_model_zoo_uses_relation_group_channel_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "poly__relation__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            time = np.linspace(-1.0, 1.0, 120)
            clean = np.column_stack([time, 2.0 * time + 0.1, np.cos(time)])
            anomalous = clean.copy()
            anomalous[40:60, 1] *= -1.0
            pd.DataFrame(clean, columns=["ch_0", "ch_1", "ch_2"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["ch_0", "ch_1", "ch_2"]).to_csv(
                instance_dir / "anomalous.csv",
                index=False,
            )
            group = _relation_event_group("0", 40, 60)
            record = InstanceRecord(
                dataset_root=root,
                variant_id="poly__relation__p00",
                split="train",
                instance_id="instance_000",
                instance_dir=instance_dir,
                clean_path=instance_dir / "clean.csv",
                anomalous_path=instance_dir / "anomalous.csv",
                events_path=instance_dir / "events.json",
                summary_path=instance_dir / "instance_summary.json",
                base_oscillation="polynomial",
                anomaly_type="correlation-flip",
                channels=3,
                length=120,
                event_groups=(group,),
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(record,))

            result = compute_model_zoo_frontier(
                dataset,
                CapabilityProtocol(
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                models=(LowRankResidualModel(rank=1), TargetRegressionResidualModel()),
            )

            self.assertEqual(set(result.frontier["model_projection"]), {"0|1"})
            self.assertEqual(set(result.event_scores["model_projection"]), {"0|1"})
            self.assertEqual({row["model_projection"] for row in result.model_manifest["models"]}, {"0|1"})

    def test_model_zoo_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 2
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="model_zoo",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="model_zoo",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in ("model_zoo_frontier.csv", "model_zoo_event_scores.csv"):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            serial_manifest = json.loads(
                (serial_output / "model_zoo_model_manifest.json").read_text(encoding="utf-8")
            )
            parallel_manifest = json.loads(
                (parallel_output / "model_zoo_model_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sorted(
                    serial_manifest["models"],
                    key=lambda item: (item["fit_variant_id"], item["model_id"]),
                ),
                sorted(
                    parallel_manifest["models"],
                    key=lambda item: (item["fit_variant_id"], item["model_id"]),
                ),
            )
            for profile_name in (
                "model_zoo_frontier",
                "model_zoo_event_scores",
                "model_zoo_manifest_rows",
            ):
                metadata_paths = list((parallel_cache / "profile_partitions" / profile_name).glob("*.metadata.json"))
                self.assertGreaterEqual(len(metadata_paths), 2, profile_name)

    def test_run_can_write_law_observability_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 2
            TSDatasetGenerator.from_dict(config).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=8,
                ),
                profile_preset="law_observability",
            )

            self.assertEqual(certificate["profiles"], ["law_observability"])
            for filename in (
                "law_observability_profile.csv",
                "law_observability_summary.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            profile = pd.read_csv(output_dir / "law_observability_profile.csv")
            self.assertIn("energy_distance", profile.columns)
            self.assertIn("c2st_balanced_accuracy", profile.columns)
            self.assertIn("law_observability_status", profile.columns)
            self.assertFalse(profile.empty)

    def test_law_observability_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 4
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=8,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="law_observability",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="law_observability",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in ("law_observability_profile.csv", "law_observability_summary.csv"):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            metadata_paths = list(
                (parallel_cache / "profile_partitions" / "law_observability_profile").glob("*.metadata.json")
            )
            self.assertGreaterEqual(len(metadata_paths), 1)

    def test_run_can_write_corrected_detectability_outputs(self) -> None:
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
                profile_preset="calibration",
            )

            self.assertEqual(certificate["profiles"], ["detectability", "corrected_detectability"])
            for filename in (
                "corrected_detectability_frontier.csv",
                "oracle_window_diagnostic_frontier.csv",
                "blind_scan_events.csv",
                "calibration_resolution.csv",
                "candidate_nulls_manifest.json",
                "scan_nulls_manifest.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            corrected = pd.read_csv(output_dir / "corrected_detectability_frontier.csv")
            self.assertIn("candidate_level_p_value", corrected.columns)
            self.assertIn("scan_level_p_value", corrected.columns)
            self.assertIn("scan_statistic", corrected.columns)
            self.assertFalse(corrected.empty)

    def test_corrected_detectability_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 3
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="calibration",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="calibration",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "corrected_detectability_frontier.csv",
                "blind_scan_events.csv",
                "calibration_resolution.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            metadata_paths = list(
                (parallel_cache / "profile_partitions" / "blind_scan_events").glob("*.metadata.json")
            )
            self.assertGreaterEqual(len(metadata_paths), 1)
            corrected_metadata_paths = list(
                (parallel_cache / "profile_partitions" / "corrected_detectability").glob("*.metadata.json")
            )
            self.assertGreaterEqual(len(corrected_metadata_paths), 1)

    def test_corrected_detectability_uses_clean_only_calibration_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            config = _small_generation_config(dataset_root)
            config["dataset"]["length"] = 120
            config["dataset"]["splits"] = {
                "train": {"paired_instances_per_variant": 1},
                "calibration": {"clean_only_instances_per_variant": 3},
            }
            config["dataset"].pop("instances_per_split")
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            TSDatasetGenerator.from_dict(config).run()

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=4,
                    bootstrap_samples=0,
                ),
                profile_preset="calibration",
            )

            corrected = pd.read_csv(output_dir / "corrected_detectability_frontier.csv")
            self.assertEqual(set(corrected["split"]), {"train"})
            self.assertGreaterEqual(int(corrected["candidate_null_count"].max()), 12)

    def test_run_consumes_metadata_registry_without_legacy_events_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()
            write_v12_metadata_registries(dataset_root, provenance="generated")
            for events_path in dataset_root.glob("variants/*/*/instances/*/events.json"):
                events_path.unlink()

            dataset = discover_dataset(dataset_root)
            self.assertTrue(dataset.metadata_events)
            self.assertTrue(dataset.problem_genotypes)
            self.assertTrue(any(group.genotype_id for instance in dataset.instances for group in instance.event_groups))
            self.assertFalse(any(instance.events_path.exists() for instance in dataset.instances))

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=4,
                    bootstrap_samples=0,
                ),
                profile_preset="annotation",
                label_export="diagnostics",
            )

            alignment = pd.read_csv(output_dir / "annotation_alignment.csv")
            self.assertFalse(alignment.empty)

    def test_corrected_detectability_reuses_blind_scan_score_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = root / "variants" / "sine__mean__p00" / "train" / "instances" / "instance_000"
            instance_dir.mkdir(parents=True)
            clean = np.zeros((80, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 3.0
            anomalous[30:40, 0] = 3.0
            pd.DataFrame(clean, columns=["value-0"]).to_csv(instance_dir / "clean.csv", index=False)
            pd.DataFrame(anomalous, columns=["value-0"]).to_csv(instance_dir / "anomalous.csv", index=False)
            instance = InstanceRecord(
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
                length=80,
                event_groups=(
                    _event_group("0", 10, 20),
                    _event_group("1", 30, 40),
                ),
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))
            protocol = CapabilityProtocol(
                alpha_grid=(0.10,),
                detection_witnesses=("mean_z",),
                max_projection_size=1,
                max_scan_windows_per_length=6,
                bootstrap_samples=0,
            )

            with patch.object(
                corrected_detectability_module,
                "_blind_scan_score_block",
                wraps=corrected_detectability_module._blind_scan_score_block,
            ) as score_block:
                result = compute_corrected_detectability_frontier(dataset, protocol)

            self.assertFalse(result.frontier.empty)
            self.assertEqual(score_block.call_count, 1)

    def test_run_can_write_implementation_validity_audits(self) -> None:
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
                profile_preset="implementation_validity",
            )

            self.assertEqual(
                certificate["profiles"],
                [
                    "observability",
                    "detectability",
                    "corrected_detectability",
                    "implementation_validity",
                ],
            )
            for filename in (
                "support_integrity.csv",
                "boundary_audit.csv",
                "shortcut_audit.csv",
                "realized_effects.csv",
                "detector_attribution.csv",
                "negative_controls.csv",
                "implementation_validity.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            validity = pd.read_csv(output_dir / "implementation_validity.csv")
            self.assertIn("implementation_validity_status", validity.columns)
            self.assertFalse(validity.empty)
            attribution = pd.read_csv(output_dir / "detector_attribution.csv")
            self.assertIn("normalized_evidence", attribution.columns)
            self.assertIn("primary_detection_cause", attribution.columns)
            self.assertFalse(attribution.empty)
            controls = pd.read_csv(output_dir / "negative_controls.csv")
            self.assertIn("control_type", controls.columns)
            self.assertIn("control_status", controls.columns)
            self.assertFalse(controls.empty)

    def test_implementation_validity_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 9
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="implementation_validity",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="implementation_validity",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "detector_attribution.csv",
                "negative_controls.csv",
                "implementation_validity.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            for profile_name in ("detector_attribution", "negative_controls"):
                metadata_paths = list((parallel_cache / "profile_partitions" / profile_name).glob("*.metadata.json"))
                self.assertGreaterEqual(len(metadata_paths), 1, profile_name)

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

            self.assertEqual(certificate["profiles"], ["observability", "identifiability", "describability"])
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
            intervention = pd.read_csv(output_dir / "labels" / "labels_oracle_intervention.csv")
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
            manifest = json.loads((output_dir / "annotation_channel_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["label_export"], "diagnostics")
            self.assertFalse(manifest["label_tables_emitted"])
            self.assertEqual(manifest["table_paths"], {})
            output_manifest = json.loads((output_dir / "manifests" / "output_manifest.json").read_text(encoding="utf-8"))
            table_names = {record["name"] for record in output_manifest["tables"]}
            self.assertNotIn("labels_oracle_any", table_names)
            self.assertIn("annotation_alignment", table_names)
            self.assertIn("annotation_robustness", table_names)
            self.assertEqual(
                certificate["run_manifest"]["output"]["label_export"],
                "diagnostics",
            )

    def test_annotation_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 9
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="annotation",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="annotation",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "labels/labels_oracle_any.csv",
                "labels/labels_weak_point.csv",
                "labels/labels_censored.csv",
                "annotation_alignment.csv",
                "annotation_robustness.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            for profile_name in (
                "labels_oracle_any",
                "labels_weak_point",
                "labels_censored",
                "annotation_alignment",
            ):
                metadata_paths = list((parallel_cache / "profile_partitions" / profile_name).glob("*.metadata.json"))
                self.assertGreaterEqual(len(metadata_paths), 2, profile_name)

    def test_run_can_write_provisional_admission_outputs(self) -> None:
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
                profile_preset="admission",
            )

            self.assertEqual(
                certificate["profiles"],
                [
                    "observability",
                    "detectability",
                    "corrected_detectability",
                    "implementation_validity",
                    "identifiability",
                    "describability",
                    "annotation_alignment",
                    "admission",
                ],
            )
            for filename in (
                "admission_events.csv",
                "admission_variants.csv",
                "admission_policy_evaluation.json",
                "release_summary.json",
                "release_summary.md",
                "generator_feedback_report.json",
                "generator_feedback_report.md",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            events = pd.read_csv(output_dir / "admission_events.csv")
            variants = pd.read_csv(output_dir / "admission_variants.csv")
            evaluation = json.loads((output_dir / "admission_policy_evaluation.json").read_text(encoding="utf-8"))
            feedback = json.loads((output_dir / "generator_feedback_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifests" / "profile_run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("admission_status", events.columns)
            self.assertIn("failure_reasons", events.columns)
            self.assertIn("provisional_gate_hits", variants.columns)
            self.assertEqual(evaluation["mode"], "provisional")
            self.assertIn("recommendations", feedback)
            self.assertFalse(events.empty)
            self.assertFalse(variants.empty)
            runtime = manifest["runtime"]
            self.assertGreater(runtime["total_elapsed_seconds"], 0.0)
            self.assertGreaterEqual(runtime["output_dir_size_bytes"], 1)
            profile_records = {record["profile"]: record for record in runtime["profiles"]}
            for profile in certificate["profiles"]:
                self.assertIn(profile, profile_records)
                self.assertEqual(profile_records[profile]["status"], "complete")
                self.assertGreaterEqual(profile_records[profile]["elapsed_seconds"], 0.0)
            self.assertIn("generator_feedback", profile_records)
            self.assertIn("observability", profile_records["observability"]["written_tables"])
            self.assertGreater(profile_records["observability"]["table_rows"]["observability"], 0)
            self.assertEqual(profile_records["annotation_alignment"]["details"]["label_export"], "full")

    def test_run_uses_external_admission_policy_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            policy_path = tmp_path / "admission_policy.yaml"
            policy_path.write_text(
                "\n".join(
                    [
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
                    ]
                ),
                encoding="utf-8",
            )
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="admission",
                admission_policy_path=policy_path,
            )

            evaluation = json.loads((output_dir / "admission_policy_evaluation.json").read_text(encoding="utf-8"))
            events = pd.read_csv(output_dir / "admission_events.csv")
            variants = pd.read_csv(output_dir / "admission_variants.csv")
            manifest = json.loads((output_dir / "manifests" / "profile_run_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(evaluation["admission_policy_version"], "synthgen.admission.test")
            self.assertEqual(evaluation["mode"], "observation_mode")
            self.assertEqual(evaluation["numeric_gate_enforcement"], "observation_mode")
            self.assertEqual(evaluation["policy"]["numeric_gates"]["promote_after_full_runs"], 5)
            self.assertEqual(evaluation["policy"]["numeric_gates"]["max_debug_event_share"]["value"], 1.0)
            self.assertEqual(evaluation["policy"]["numeric_gates"]["max_debug_event_share"]["status"], "observation")
            self.assertEqual(evaluation["policy"]["numeric_gates"]["max_needs_repair_share"]["value"], 1.0)
            self.assertEqual(evaluation["policy"]["numeric_gates"]["max_needs_repair_share"]["status"], "observation")
            self.assertEqual(set(events["policy_mode"].astype(str)), {"observation_mode"})
            self.assertEqual(set(variants["policy_mode"].astype(str)), {"observation_mode"})
            self.assertIn("admission_policy_hash", manifest["fingerprints"])

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
            self.assertTrue((output_dir / "visual_audit" / "visual_audit_manifest.json").exists())
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
            self.assertEqual(certificate["release_certificate_version"], "synthgen.release_certificate.v12.1")
            self.assertEqual(certificate["certificate_status"], "complete")
            self.assertTrue((output_dir / "release_certificate.json").exists())
            self.assertTrue((output_dir / "release_certificate.md").exists())

    def test_visual_audit_writes_relation_panels_and_html_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "analysis"
            instance_dir = (
                root
                / "dataset"
                / "variants"
                / "rmj__mode-correlation__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
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
            dataset = DatasetIndex(root=root / "dataset", manifest={}, instances=(instance,))
            event_summary = pd.DataFrame(
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

            result = compute_visual_audit(
                dataset,
                CapabilityProtocol(alpha_grid=(0.10,)),
                output_dir=output_dir,
                event_summary=event_summary,
            )

            self.assertTrue((output_dir / "visual_audit" / "index.html").exists())
            written = result.selection[result.selection["plot_status"] == "written"]
            self.assertFalse(written.empty)
            first = written.iloc[0]
            for column in (
                "relation_scatter_path",
                "rolling_correlation_path",
                "pca_residual_path",
            ):
                panel_path = output_dir / "visual_audit" / str(first[column])
                self.assertTrue(panel_path.exists(), panel_path)
            self.assertGreaterEqual(result.manifest["relation_scatter_count"], 1)
            self.assertGreaterEqual(result.manifest["rolling_correlation_count"], 1)
            self.assertGreaterEqual(result.manifest["pca_residual_count"], 1)
            self.assertIn("Visual Audit", result.index_html)

    def test_full_certificate_parallel_matches_serial_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 2
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            serial_certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="full_certificate",
                n_jobs=1,
            )
            parallel_certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="full_certificate",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            self.assertEqual(serial_certificate["profiles"], parallel_certificate["profiles"])
            csv_paths = _release_csv_paths(serial_output)
            self.assertGreaterEqual(len(csv_paths), 30)
            self.assertEqual(csv_paths, _release_csv_paths(parallel_output))
            for relative_path in csv_paths:
                serial = pd.read_csv(serial_output / relative_path)
                parallel = pd.read_csv(parallel_output / relative_path)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            for profile_name in (
                "blind_scan_events",
                "detector_attribution",
                "law_observability_profile",
                "model_zoo_frontier",
                "negative_controls",
                "annotation_alignment",
                "labels_oracle_any",
            ):
                metadata_paths = list((parallel_cache / "profile_partitions" / profile_name).glob("*.metadata.json"))
                self.assertGreaterEqual(len(metadata_paths), 1, profile_name)


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


if __name__ == "__main__":
    unittest.main()
