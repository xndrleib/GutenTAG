"""Regressions for the audit, including integration and law-level invariants."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gutenTAG.tsgen.processes.kernels import KERNELS, KernelSpec, feature_sample
from gutenTAG.tsgen.processes.laws import ProcessLaw, sample_law
from gutenTAG.tsgen.processes.interventions import (
    Intervention,
    MECHANISMS,
    apply_interventions,
)
from gutenTAG.tsgen.benchmark.config import BenchmarkConfig, SplitSpec
from gutenTAG.tsgen.benchmark.generation import generate_benchmark
from gutenTAG.tsgen.capabilities.v13_evaluation import empirical_p, evaluate_benchmark
from gutenTAG.tsgen.capabilities.grouped_statistics import (
    paired_c2st,
    cluster_energy_interval,
)
from gutenTAG.tsgen.capabilities.corrected_detectability_partitions import (
    calibration_null_instances,
)
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol
from gutenTAG.tsgen.capabilities.law_observability import _window_features
from gutenTAG.tsgen.parameters import realize_parameters
from gutenTAG.tsgen.sensor_artifacts import SENSOR_KINDS, apply_sensor_artifact
from gutenTAG.generator.latent_laws import (
    apply_lmc_noise_correlation,
    recouple_lmc_target,
)
from gutenTAG.tsgen.fingerprint_audit import nuisance_matrix


@pytest.mark.parametrize("kind", KERNELS)
def test_spectrum_matches_named_kernel(kind):
    spec = KernelSpec(kind, length_scale=1.3, period=4, alpha=0.7)
    omega = spec.spectrum(np.random.default_rng(200), 120000)
    lags = np.array([0.0, 0.3, 0.8, 2.1])
    empirical = np.cos(lags[:, None] * omega).mean(axis=1)
    np.testing.assert_allclose(empirical, spec.covariance(lags), atol=0.011)


def test_feature_blocks_do_not_resample_coefficients():
    w = np.arange(20) / 10
    a = feature_sample(w, np.arange(257), np.random.default_rng(1), block_size=13)
    b = feature_sample(w, np.arange(257), np.random.default_rng(1), block_size=1000)
    np.testing.assert_allclose(a, b, atol=1e-14)


@pytest.mark.parametrize("family", ["diagonal-ar", "gp-lmc", "independent-gp"])
def test_immutable_law_roundtrip_and_independent_realizations(family):
    law = sample_law(family, 4, 71)
    assert ProcessLaw.from_dict(law.to_dict()).law_id == law.law_id
    assert copy.deepcopy(law) is law
    with pytest.raises(ValueError):
        law.mean[0] = 8
    np.testing.assert_array_equal(law.sample(64, 1).values, law.sample(64, 1).values)
    assert not np.array_equal(law.sample(64, 1).values, law.sample(64, 2).values)


@pytest.mark.parametrize("kind", MECHANISMS)
def test_every_mechanism_has_a_true_null(kind):
    law = sample_law("gp-lmc" if kind == "factor-loading" else "diagonal-ar", 3, 6)
    x = law.sample(96, 9)
    y = apply_interventions(x, [Intervention(kind, 24, 64, 0, (0,), 3)], seed=4)
    np.testing.assert_array_equal(y.values, x.values)
    assert not y.effect_mask.any() and not y.intervention_mask.any()


@pytest.mark.parametrize("kind", ["covariance-change", "precision-change"])
def test_relation_law_preserves_every_innovation_variance_and_clean_branch(kind):
    law = sample_law("diagonal-ar", 4, 5)
    clean = law.sample(128, 10)
    result = apply_interventions(
        clean, [Intervention(kind, 32, 96, 0.8, (1,), 4)], seed=20
    )
    a = np.array(result.events[0]["sigma_before"])
    b = np.array(result.events[0]["sigma_endpoint"])
    np.testing.assert_allclose(a.diagonal(), b.diagonal(), atol=1e-12)
    for weight in np.linspace(0, 1, 25):
        assert np.linalg.eigvalsh((1 - weight) * a + weight * b).min() > 0
    np.testing.assert_array_equal(clean.values, law.sample(128, 10).values)
    assert result.events[0]["trajectory_marginals_preserved"]


def test_noise_ratio_identity_and_negative_strength_are_real_attenuation():
    law = sample_law("diagonal-ar", 3, 44, memory=0)
    clean = law.sample(20000, 3)
    altered = apply_interventions(
        clean, [Intervention("noise-scale", 1000, 19000, -1, (0,))], seed=4
    )
    ratio = altered.values[1000:19000, 0].std() / clean.values[1000:19000, 0].std()
    assert ratio == pytest.approx(np.exp(-1), rel=1e-12)


def test_polynomial_coefficients_and_literal_arrays_are_not_ranges():
    result = realize_parameters(
        {"polynomial": [0.05, 0.4], "x": {"literal": [2, 3]}}, np.random.default_rng(1)
    )
    assert result == {"polynomial": [0.05, 0.4], "x": [2, 3]}


def test_impossible_rejection_cannot_fabricate_out_of_prior_value():
    with pytest.raises(ValueError, match="exhausted"):
        realize_parameters(
            {"x": {"distribution": "reject_if_abs_lt", "threshold": 1.0, "base": 0.0}},
            np.random.default_rng(1),
        )


@pytest.mark.parametrize("kind", SENSOR_KINDS)
def test_sensor_null_and_mask_preserve_missing_data(kind):
    x = np.array([1.0, np.nan, 3.0, 4.0])
    r = apply_sensor_artifact(x, kind=kind, severity=0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(r.values, x)
    np.testing.assert_array_equal(r.observed_mask, np.isfinite(x))


def test_lmc_state_is_shared_and_operator_uses_it():
    channels = [
        SimpleNamespace(noise=np.ones(100000), variance=0.3, amplitude=1.0)
        for _ in range(4)
    ]
    assert apply_lmc_noise_correlation(
        channel_bos=channels, seed=91, config={"mode": "lmc"}
    )
    assert channels[0]._process_state is channels[3]._process_state
    after = recouple_lmc_target(
        bo=channels[1],
        anchor=0,
        start=0,
        end=100000,
        target_correlation=-0.6,
        transition=0,
    )
    assert np.corrcoef(after, channels[0].noise)[0, 1] == pytest.approx(-0.6, abs=0.015)
    assert after.std() == pytest.approx(0.3, abs=0.008)


def test_required_calibration_never_falls_back_to_test():
    instance = SimpleNamespace(split="test")
    with pytest.raises(ValueError, match="absent"):
        calibration_null_instances(
            [instance], CapabilityProtocol(calibration_split="calibration")
        )


def test_signed_law_witness_distinguishes_flip():
    z = np.arange(-2, 3, dtype=float)
    a = _window_features(np.c_[z, z], 0, 5, (0, 1))
    b = _window_features(np.c_[z, -z], 0, 5, (0, 1))
    assert not np.array_equal(a, b)


def test_missing_nuisance_features_fail_closed():
    with pytest.raises(ValueError, match="Missing"):
        nuisance_matrix([{"source_length": 32}])


def test_cluster_uncertainty_and_identical_counterfactual_null():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(16, 4))
    groups = np.repeat(np.arange(4), 4)
    assert paired_c2st(x, x, groups) == 0.5
    low, high = cluster_energy_interval(x, x, groups, samples=15, rng=rng)
    assert low == pytest.approx(0, abs=1e-12) and high == pytest.approx(0, abs=1e-12)


def tiny_config():
    common = {
        "systems": 1,
        "families": ["diagonal-ar"],
        "mechanisms": ["mean", "covariance-change", "null"],
        "channels": [3],
    }
    return BenchmarkConfig(
        length=64,
        reference_length=64,
        calibration_realizations=49,
        replicas_per_genotype=2,
        features=8,
        window_lengths=(8, 16),
        stride=4,
        alpha_grid=(0.1,),
        splits={
            name: {**common, "role": role}
            for name, role in [
                ("train", "train"),
                ("validation", "validation"),
                ("calibration", "calibration"),
                ("test", "iid"),
            ]
        },
    )


def test_split_contracts_reject_fake_ood_and_relation_only():
    with pytest.raises(ValueError, match="relation_only"):
        SplitSpec(role="relation_only", mechanisms=("mean",))
    c = tiny_config().model_dump()
    c["splits"]["test"]["role"] = "parameter_ood"
    c["splits"]["test"]["ood_parameters"] = ["memory"]
    with pytest.raises(ValueError, match="overlap"):
        BenchmarkConfig.model_validate(c)


def test_complete_generation_evaluation_and_oracle_isolation(tmp_path):
    c = tiny_config()
    m = generate_benchmark(c, tmp_path / "data")
    assert len(m["systems"]) == 4
    assert len({x["law_id"] for x in m["systems"]}) == 4
    result = evaluate_benchmark(tmp_path / "data", tmp_path / "eval.json")
    assert result["calibration_resolution_ok"]
    assert len(result["rows"]) == 18
    # Hash verification is disabled ONLY to test independence of model scores.
    for p in (tmp_path / "data").rglob("oracle/*.npz"):
        with np.load(p) as a:
            data = dict(a)
        data["clean"][:] = 1e9
        data["intervention_mask"][:] = False
        np.savez_compressed(p, **data)
    second = evaluate_benchmark(
        tmp_path / "data", tmp_path / "eval2.json", verify_hashes=False
    )
    assert [r["maximum"] for r in result["rows"]] == [
        r["maximum"] for r in second["rows"]
    ]
    with pytest.raises(ValueError, match="integrity"):
        evaluate_benchmark(tmp_path / "data", tmp_path / "eval3.json")


def test_rank_calibration_is_conservative_with_ties():
    assert empirical_p(np.ones(19), np.array([1.0]))[0] == 1
    assert empirical_p(np.ones(19), np.array([2.0]))[0] == 0.05


def test_gp_offset_is_added_exactly_once_and_scale_not_forced():
    from gutenTAG.base_oscillations.gp_mixture import GaussianProcessMixture
    from gutenTAG.generator.base_channels import apply_variations

    def generate(offset, seed):
        bo = GaussianProcessMixture(
            length=100, offset=offset, gp_features=12, gp_components=2
        )
        x = bo.generate_only_base(SimpleNamespace(rng=np.random.default_rng(seed)))
        return apply_variations(x[:, None], [bo])[:, 0]

    np.testing.assert_allclose(generate(3.25, 4) - generate(0, 4), 3.25)
    assert np.std([generate(0, s).std() for s in range(8)]) > 0.05


def test_linear_trend_is_not_annihilated_by_default():
    from gutenTAG.anomalies.types.trend import AnomalyTrend, AnomalyTrendParameters

    t = AnomalyTrend(AnomalyTrendParameters(trend=None))
    x = np.linspace(2, 5, 33)
    np.testing.assert_allclose(t._bounded_local_trend(x, np.ones_like(x)), x - x[0])


def test_clone_is_exact_without_sharing_mutable_arrays():
    from gutenTAG.tsgen.paired_instance_generation import (
        _generate_anomalous_base_series,
    )

    state = SimpleNamespace(
        base_values=np.arange(12).reshape(6, 2).astype(float),
        channel_bos=[SimpleNamespace(noise=np.ones(6))],
    )
    clone = _generate_anomalous_base_series(
        config=None,
        variant=None,
        seeds={},
        base_instance=SimpleNamespace(series=state),
        effective_base_channel_correlation={},
    )
    np.testing.assert_array_equal(clone.base_values, state.base_values)
    clone.base_values[0, 0] = 999
    clone.channel_bos[0].noise[0] = 999
    assert state.base_values[0, 0] == 0 and state.channel_bos[0].noise[0] == 1


def test_empty_effect_support_and_unknown_resample_policy():
    from gutenTAG.tsgen.labels.effective_support import (
        resolve_label_bounds_from_effect,
        normalize_subsequence_length,
    )

    assert resolve_label_bounds_from_effect(
        protocol_start=8,
        protocol_end=16,
        delta=np.zeros(8),
        anomaly_type="mean",
        support_label_mode="effective_support",
        support_eps_mode="relative",
        support_eps_value=0.03,
        min_effective_label_length_non_extremum=2,
    ) == (8, 8)
    with pytest.raises(ValueError, match="Unknown"):
        normalize_subsequence_length(np.ones(8), 8, "typo")


def test_fully_blind_channel_permutation_invariance():
    from gutenTAG.tsgen.capabilities.v13_evaluation import blind_scan

    law = sample_law("diagonal-ar", 5, 8)
    ref, query = law.sample(256, 9).values, law.sample(256, 10).values
    perm = [2, 4, 0, 3, 1]
    a = blind_scan(ref, query, windows=(16, 32), stride=8)
    b = blind_scan(ref[:, perm], query[:, perm], windows=(16, 32), stride=8)
    np.testing.assert_allclose(a.scores, b.scores)


def test_unsupported_causal_mixture_is_rejected():
    x = sample_law("diagonal-ar", 3, 2).sample(128, 7)
    with pytest.raises(ValueError, match="causal composition"):
        apply_interventions(
            x,
            [
                Intervention("covariance-change", 16, 32, 0.5, (0,)),
                Intervention("clipping", 64, 96, 0.5, (0,)),
            ],
            seed=9,
        )


def test_v13_cannot_be_silently_discovered_by_legacy_loader(tmp_path):
    from gutenTAG.tsgen.capabilities.dataset import discover_dataset

    (tmp_path / "dataset_manifest.json").write_text(
        '{"schema_version": "synthgen.v13.1"}'
    )
    with pytest.raises(ValueError, match="evaluate_benchmark"):
        discover_dataset(tmp_path)


def test_streamed_annotation_manifest_hashes_final_bytes(tmp_path):
    from gutenTAG.tsgen.sidecars import write_dataset_annotation_channels, _file_hash
    from gutenTAG.tsgen.capabilities.dataset import DatasetIndex

    dataset = DatasetIndex(
        root=tmp_path,
        manifest={},
        instances=(),
        metadata_events={},
        problem_genotypes={},
    )
    result = write_dataset_annotation_channels(dataset, config={"emit": ["oracle"]})
    path = tmp_path / result["manifest_path"]
    assert result["manifest_hash"] == _file_hash(path)
    assert "manifest_hash" not in json.loads(path.read_text())
