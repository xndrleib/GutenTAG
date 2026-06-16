import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG.tsgen.capabilities import CapabilityProtocol
from gutenTAG.tsgen.capabilities.dataset import DatasetIndex, InstanceRecord
from gutenTAG.tsgen.capabilities.repair.operators import repair_segment
from gutenTAG.tsgen.capabilities.validation.boundary import compute_boundary_audit
from gutenTAG.tsgen.capabilities.validation.negative_controls import (
    _event_negative_controls,
)
from gutenTAG.tsgen.capabilities.validation.realized_effects import (
    compute_realized_effects,
)
from gutenTAG.tsgen.capabilities.validation.support import compute_support_integrity

from tests.capability_execution_fixtures import (
    _event_group,
    _instance_record_for_groups,
    _relation_event_group,
    _support_instance_record,
)


class TestCapabilityValidationUnits(unittest.TestCase):
    def test_support_integrity_excludes_other_known_event_supports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            clean = np.zeros((100, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 1.0
            anomalous[70:80, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
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

    def test_support_integrity_treats_near_field_residual_as_boundary_uncertain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            clean = np.zeros((100, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 1.0
            anomalous[20:30, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
            instance = _support_instance_record(
                root, instance_dir, (_event_group("0", 10, 20),)
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            support = compute_support_integrity(dataset)

            self.assertEqual(support.iloc[0]["support_status"], "boundary_uncertain")
            self.assertAlmostEqual(float(support.iloc[0]["inside_mass_share"]), 0.5)
            self.assertAlmostEqual(
                float(support.iloc[0]["support_or_near_field_mass_share"]), 1.0
            )
            self.assertAlmostEqual(float(support.iloc[0]["far_field_mass_share"]), 0.0)

    def test_support_integrity_keeps_far_field_residual_leaky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            clean = np.zeros((100, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 1.0
            anomalous[80:90, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
            instance = _support_instance_record(
                root, instance_dir, (_event_group("0", 10, 20),)
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))

            support = compute_support_integrity(dataset)

            self.assertEqual(support.iloc[0]["support_status"], "leaky")
            self.assertAlmostEqual(float(support.iloc[0]["inside_mass_share"]), 0.5)
            self.assertAlmostEqual(
                float(support.iloc[0]["support_or_near_field_mass_share"]), 0.5
            )
            self.assertAlmostEqual(float(support.iloc[0]["far_field_mass_share"]), 0.5)

    def test_boundary_audit_uses_source_edges_not_trimmed_effective_support_edges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "sine__variance__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            clean = np.zeros((40, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[12:14, 0] = 1.0
            anomalous[16:18, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
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
            instance_dir = (
                root
                / "variants"
                / "sine__variance__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            clean = np.zeros((40, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:12, 0] = 1.0
            anomalous[18:20, 0] = 1.0
            pd.DataFrame(clean, columns=["ch_0"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["ch_0"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
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

    def test_covariance_repair_operator_dispatch_ignores_variance_substring(
        self,
    ) -> None:
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

    def test_variance_repair_uses_affine_fallback_when_rescale_is_insufficient(
        self,
    ) -> None:
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

        wrong_witness = controls[
            controls["control_type"] == "wrong_witness_control"
        ].iloc[0]
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

        wrong_support = controls[
            controls["control_type"] == "wrong_support_control"
        ].iloc[0]
        self.assertEqual(wrong_support["control_status"], "control_passed")
        self.assertLess(float(wrong_support["control_score"]), 1.0)


if __name__ == "__main__":
    unittest.main()
