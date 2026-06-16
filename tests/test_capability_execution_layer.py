import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gutenTAG.tsgen.capabilities import CapabilityProtocol
from gutenTAG.tsgen.capabilities.admission import (
    AdmissionPolicy,
    _event_admission,
    _variant_status,
)
from gutenTAG.tsgen.capabilities.array_store import ArrayStore
from gutenTAG.tsgen.capabilities.dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
)
from gutenTAG.tsgen.capabilities.diagnosis import (
    _feature_matrix as _diagnosis_feature_matrix,
    _nearest_centroid_predictions,
    compute_diagnosis_profiles,
)
from gutenTAG.tsgen.capabilities.execution import resolve_profile_names
from gutenTAG.tsgen.capabilities.identifiability import _leave_one_out_nearest_labels
from gutenTAG.tsgen.capabilities.validation.attribution import (
    compute_detector_attribution,
)
from gutenTAG.tsgen.capabilities.validation.implementation_validity import (
    _implementation_validity,
)
from gutenTAG.tsgen.capabilities.windows import WindowLibrary, WindowSpec

VALIDITY_EVENT_IDS = (
    "variant/test/instance_000/g0",
    "variant/test/instance_000/g1",
)
DETECTOR_ATTRIBUTION_EVENT_ID = "sine__variance__p00/train/instance_000/g0"
DETECTOR_ATTRIBUTION_ROOT = Path("/tmp/synth-gen-capability-test")
PROFILE_RESOLUTION_CASES = (
    (
        {"profiles": ("identifiability",)},
        ("observability", "identifiability"),
    ),
    (
        {"profiles": ("all",)},
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
    ),
    (
        {"profile_preset": "implementation_validity"},
        (
            "observability",
            "detectability",
            "corrected_detectability",
            "implementation_validity",
        ),
    ),
    ({"profile_preset": "calibration"}, ("detectability", "corrected_detectability")),
    ({"profile_preset": "model_zoo"}, ("model_zoo",)),
    ({"profile_preset": "law_observability"}, ("law_observability",)),
    ({"profile_preset": "diagnosis"}, ("observability", "identifiability")),
    ({"profile_preset": "repair"}, ("observability", "describability")),
    (
        {"profile_preset": "maturity"},
        ("observability", "identifiability", "describability"),
    ),
    ({"profile_preset": "annotation"}, ("annotation_alignment",)),
    ({"profile_preset": "annotation_channels"}, ("annotation_alignment",)),
    (
        {"profile_preset": "admission"},
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
    ),
    (
        {"profile_preset": "visual_audit"},
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
    ),
)


def _single_event_frame(event_id: str, **columns: Any) -> pd.DataFrame:
    return pd.DataFrame([{"event_id": event_id, **columns}])


def _variance_event_group() -> EventGroup:
    return EventGroup(
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


def _variance_dataset_index() -> DatasetIndex:
    instance = InstanceRecord(
        dataset_root=DETECTOR_ATTRIBUTION_ROOT,
        variant_id="sine__variance__p00",
        split="train",
        instance_id="instance_000",
        instance_dir=DETECTOR_ATTRIBUTION_ROOT,
        clean_path=DETECTOR_ATTRIBUTION_ROOT / "clean.csv",
        anomalous_path=DETECTOR_ATTRIBUTION_ROOT / "anomalous.csv",
        events_path=DETECTOR_ATTRIBUTION_ROOT / "events.json",
        summary_path=DETECTOR_ATTRIBUTION_ROOT / "instance_summary.json",
        base_oscillation="sine",
        anomaly_type="variance",
        channels=1,
        length=80,
        event_groups=(_variance_event_group(),),
    )
    return DatasetIndex(root=instance.dataset_root, manifest={}, instances=(instance,))


def _variance_frontier_detecting_shortcut_and_canonical() -> pd.DataFrame:
    return _single_event_frame(
        DETECTOR_ATTRIBUTION_EVENT_ID,
        variant_id="sine__variance__p00",
        split="train",
        instance_id="instance_000",
        anomaly_type="variance",
        constraint_tag="scale.variance",
        semantic_scope="channel",
        alpha=0.01,
        family="location.mean",
        witness_or_model="mean_z",
        projection="0",
        raw_score=5.0,
        candidate_level_p_value=0.001,
        scan_level_p_value=0.001,
        scan_threshold=3.0,
        detected=True,
        scan_null_count=1000,
        best_canonical_witness="variance_log_ratio@0",
        best_canonical_candidate_p_value=0.01,
        best_canonical_detected=True,
    )


def _detector_attribution_with_detected_canonical() -> pd.DataFrame:
    return compute_detector_attribution(
        _variance_dataset_index(),
        _variance_frontier_detecting_shortcut_and_canonical(),
        _single_event_frame(
            DETECTOR_ATTRIBUTION_EVENT_ID,
            boundary_primary_detection_cause=False,
        ),
        _single_event_frame(
            DETECTOR_ATTRIBUTION_EVENT_ID,
            shortcut_status="valid_no_shortcut",
        ),
    )


def _relation_support_table(event_ids: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                "variant_id": "variant",
                "split": "test",
                "instance_id": "instance_000",
                "anomaly_type": "correlation-flip",
                "constraint_tag": "dependence.correlation",
                "semantic_scope": "relation",
                "support_status": "valid",
            }
            for event_id in event_ids
        ]
    )


def _constant_event_status_table(
    event_ids: tuple[str, ...],
    status_column: str,
    status_value: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [{"event_id": event_id, status_column: status_value} for event_id in event_ids]
    )


def _shortcut_detector_attribution_table() -> pd.DataFrame:
    valid_id, wrong_reason_id = VALIDITY_EVENT_IDS
    return pd.DataFrame(
        [
            {
                "event_id": valid_id,
                "alpha": 0.01,
                "rank_within_event": 1,
                "primary_detection_cause": "valid_with_shortcut",
                "best_canonical_detected": True,
            },
            {
                "event_id": wrong_reason_id,
                "alpha": 0.01,
                "rank_within_event": 1,
                "primary_detection_cause": "detected_wrong_reason",
                "best_canonical_detected": False,
            },
        ]
    )


def _implementation_validity_from_detector_attribution() -> pd.DataFrame:
    return _implementation_validity(
        support=_relation_support_table(VALIDITY_EVENT_IDS),
        boundary=_constant_event_status_table(
            VALIDITY_EVENT_IDS,
            "boundary_status",
            "valid_interior_or_mixed",
        ),
        shortcut=_constant_event_status_table(
            VALIDITY_EVENT_IDS,
            "shortcut_status",
            "shortcut_dominated",
        ),
        realized=pd.DataFrame(),
        negative_controls=pd.DataFrame(),
        detector_attribution=_shortcut_detector_attribution_table(),
    ).set_index("event_id")


def _assert_detector_attribution_drives_validity(
    testcase: unittest.TestCase,
    validity: pd.DataFrame,
) -> None:
    valid_id, wrong_reason_id = VALIDITY_EVENT_IDS
    testcase.assertEqual(
        validity.loc[valid_id, "implementation_validity_status"],
        "valid_candidate",
    )
    testcase.assertEqual(
        validity.loc[valid_id, "detector_primary_detection_cause"],
        "valid_with_shortcut",
    )
    testcase.assertTrue(
        bool(validity.loc[valid_id, "detector_best_canonical_detected"])
    )
    testcase.assertEqual(
        validity.loc[wrong_reason_id, "implementation_validity_status"],
        "detected_wrong_reason",
    )


def _model_zoo_only_admission_events(event_id: str) -> pd.DataFrame:
    return _event_admission(
        protocol=CapabilityProtocol(alpha_grid=(0.01,), delta_grid=(0.20,)),
        policy=AdmissionPolicy(),
        event_summary=_single_event_frame(
            event_id,
            variant_id="variant",
            split="test",
            instance_id="instance_000",
            anomaly_type="pattern",
            constraint_tag="shape.local_template",
            semantic_scope="channel",
        ),
        arity=_single_event_frame(
            event_id,
            delta=0.20,
            canonical_is_observable_at_delta=True,
            canonical_observed_arity=1,
        ),
        implementation_validity=_single_event_frame(
            event_id,
            implementation_validity_status="valid_candidate",
            support_status="valid",
            boundary_status="valid_interior_or_mixed",
            shortcut_status="valid_no_shortcut",
            realized_offset=0.0,
        ),
        detector_attribution=_single_event_frame(
            event_id,
            alpha=0.01,
            rank_within_event=1,
            primary_detection_cause="observable_not_detected",
        ),
        corrected_detectability=_single_event_frame(
            event_id,
            alpha=0.01,
            detected=False,
            scan_level_p_value=0.50,
            calibration_status="calibration_ok",
        ),
        repair_profile=_single_event_frame(
            event_id,
            repair_status="repair_effective",
            repair_gain=0.80,
        ),
        model_zoo_frontier=_single_event_frame(
            event_id,
            alpha=0.01,
            detected=True,
        ),
        metadata={
            event_id: {
                "genotype_id": "genotype:variant",
                "contract_id": "synthgen.contract.pattern.v1",
            }
        },
    )


class TestCapabilityExecutionLayer(unittest.TestCase):
    def test_identifiability_leave_one_out_nearest_labels_excludes_self(self) -> None:
        matrix = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
        labels = np.asarray(["a", "a", "b", "b"], dtype=object)

        self.assertEqual(
            _leave_one_out_nearest_labels(matrix, labels), ["a", "a", "b", "b"]
        )

        empty_features = np.zeros((3, 0), dtype=float)
        empty_labels = np.asarray(["x", "y", "z"], dtype=object)
        self.assertEqual(
            _leave_one_out_nearest_labels(empty_features, empty_labels), ["y", "x", "x"]
        )

    def test_diagnosis_leave_group_out_centroid_predictions_are_vectorized(
        self,
    ) -> None:
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

    def test_diagnosis_profiles_run_without_dense_pairwise_distance_matrix(
        self,
    ) -> None:
        embeddings = pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(8)],
                "variant_id": ["v0", "v1"] * 4,
                "base_oscillation": ["sine", "sine", "cosine", "cosine"] * 2,
                "anomaly_type": ["mean", "mean", "variance", "variance"] * 2,
                "constraint_tag": [
                    "location.mean",
                    "location.mean",
                    "scale.variance",
                    "scale.variance",
                ]
                * 2,
                "operator_family": ["location", "location", "scale", "scale"] * 2,
                "support_type": ["segment"] * 8,
                "semantic_scope": ["channel"] * 8,
                "repair_operator": [
                    "additive_offset",
                    "additive_offset",
                    "match_local_variance",
                    "match_local_variance",
                ]
                * 2,
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
            instance_dir = (
                root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
            pd.DataFrame(values, columns=["ch_0", "ch_1"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(values + 1.0, columns=["ch_0", "ch_1"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
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

    def test_detector_attribution_downgrades_shortcut_when_canonical_candidate_also_detected(
        self,
    ) -> None:
        attribution = _detector_attribution_with_detected_canonical()

        row = attribution.iloc[0]
        self.assertEqual(row["primary_detection_cause"], "valid_with_shortcut")
        self.assertTrue(bool(row["best_canonical_detected"]))

    def test_implementation_validity_uses_detector_attribution_for_shortcuts(
        self,
    ) -> None:
        validity = _implementation_validity_from_detector_attribution()

        _assert_detector_attribution_drives_validity(self, validity)

    def test_variant_status_honors_needs_repair_share_gate(self) -> None:
        policy = AdmissionPolicy(max_needs_repair_share=0.05)
        at_gate = pd.DataFrame(
            {"admission_status": ["release"] * 95 + ["needs_repair"] * 5}
        )
        over_gate = pd.DataFrame(
            {"admission_status": ["release"] * 94 + ["needs_repair"] * 6}
        )

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

    def test_event_admission_accepts_model_zoo_only_detection_with_warning(
        self,
    ) -> None:
        event_id = "variant/test/instance_000/g0"
        events = _model_zoo_only_admission_events(event_id)

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
        for kwargs, expected in PROFILE_RESOLUTION_CASES:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(resolve_profile_names(**kwargs), expected)


if __name__ == "__main__":
    unittest.main()
