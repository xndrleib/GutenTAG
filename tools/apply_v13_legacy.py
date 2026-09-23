"""Final assertion-guarded audit changes; removed after the source commit."""
from pathlib import Path
r = Path(__file__).resolve().parents[1]


def patch(path, old, new):
    f = r/path
    s = f.read_text()
    if old not in s:
        raise ValueError(f"Patch precondition failed: {path}: {old[:80]}")
    f.write_text(s.replace(old, new))


patch('gutenTAG/config/parser.py', '''            parameters = {
                PARAMETERS.TREND: decode_trend_obj(
                    deepcopy(d[PARAMETERS.OSCILLATION]), length
                )
            }''', '''            parameters = deepcopy(d)
            del parameters[PARAMETERS.KIND]
            parameters[PARAMETERS.TREND] = decode_trend_obj(
                parameters.pop(PARAMETERS.OSCILLATION), length
            )''')
p = 'gutenTAG/config/schema/anomaly-kind.guten-tag-generation-config.schema.yaml'
patch(p, 'enum: [inside_window_zero_endpoints, legacy_carry_over]', 'enum: [inside_window_preserve_shape, inside_window_zero_endpoints, legacy_carry_over]')
f = r/p; s = f.read_text(); a = s.index('  variance-params:')
s = s[:a]+s[a:].replace('      variance:\n', '      noise_std_ratio:\n        type: number\n        minimum: 0\n        description: Standard-deviation ratio; one is the identity with shared innovations.\n      variance:\n', 1)
f.write_text(s)
p = 'gutenTAG/tsgen/processes/laws.py'
patch(p, 'np.allclose(x, x.T, atol=1e-12)', 'np.allclose(x, x.T, atol=1e-12, rtol=0)')
patch(p, 'np.allclose(np.diag(x), 1.0, atol=1e-12)', 'np.allclose(np.diag(x), 1.0, atol=1e-12, rtol=0)')
p = 'gutenTAG/tsgen/processes/interventions.py'
patch(p, '    occupied = np.zeros(n, dtype=bool)', '''    # Mixed sensor/process events require causal composition. Never silently
    # add an AR recovery tail after a sensor transformation of the same samples.
    if len(events) > 1 and any(e.kind in RELATION_ONLY + ("noise-scale",) and e.strength != 0 for e in events):
        raise ValueError("Multi-event innovation interventions require an explicit causal composition contract")
    occupied = np.zeros(n, dtype=bool)''')
patch(p, '                shape[event.start:event.end] = local', '                shape[event.start:event.end] = local * g[event.start:event.end]')
patch(p, '''        if active:
            mask[event.start:event.end, selected] = True''', '''        if active and event.kind == "noise-scale" and law.family == "diagonal-ar":
            memory = float(np.max(np.abs(law.coefficients[selected])))
            recovery = 0 if memory == 0 else int(np.ceil(np.log(1e-8)/np.log(memory)))
            evaluation[event.end:min(n, event.end+recovery)] = False
            extra["recovery_exclusion"] = [event.end, min(n, event.end+recovery)]
        if active:
            mask[event.start:event.end, selected] = True''')
p = 'gutenTAG/tsgen/benchmark/generation.py'
patch(p, 'from ..seeding import derive_seed\n', '')
patch(p, 'LOGGER = logging.getLogger(__name__)', '''LOGGER = logging.getLogger(__name__)


def derive_seed(seed: int, *parts: str) -> int:
    """128-bit, unambiguously namespaced v13 seeds; legacy seeds are unchanged."""
    payload = json.dumps([int(seed), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")''')
p = 'gutenTAG/tsgen/capabilities/v13_evaluation.py'
patch(p, '    scores: np.ndarray\n', '    scores: np.ndarray\n    components: np.ndarray\n')
patch(p, '''    scores = np.maximum(np.max(np.abs(mean), axis=1), np.max(np.abs(np.log(var)), axis=1))
    scores = np.maximum(scores, 10*_window_mean(missing.astype(float), start, end).max(axis=1))''', '''    component = np.zeros((len(start), 5))
    component[:, 0] = np.max(np.abs(mean), axis=1)
    component[:, 1] = np.max(np.abs(np.log(var)), axis=1)
    component[:, 3] = _window_mean(missing.astype(float), start, end).max(axis=1)''')
patch(p, '        scores = np.maximum(scores, np.max(np.abs(qm-rm), axis=1))', '        component[:, 2] = np.maximum(component[:, 2], np.max(np.abs(qm-rm), axis=1))')
patch(p, '''        scores = np.maximum(scores, _window_mean(jitter, start, end).max(axis=1))
    return ScanResult(start, end, scores)''', '''        component[:, 4] = _window_mean(jitter, start, end).max(axis=1)
    # Raw maxima are diagnostics ONLY. Decisions use calibrated families.
    return ScanResult(start, end, component.max(axis=1), component)''')
patch(p, 'null.append(blind_scan(reference, f["values"], windows=config.window_lengths, stride=config.stride).maximum)', 'null.append(blind_scan(reference, f["values"], windows=config.window_lengths, stride=config.stride).components.max(axis=0))')
patch(p, '            predictions.append((query, scan, empirical_p(np.asarray(null), scan.scores)))', '''            family_p = np.column_stack([empirical_p(np.asarray(null)[:, k], scan.components[:, k])
                                        for k in range(scan.components.shape[1])])
            # Every family is scan-corrected; correct selection across families.
            pvalues = np.minimum(1.0, family_p.shape[1]*family_p.min(axis=1))
            predictions.append((query, scan, pvalues))''')
patch(p, '"scan_p": float(empirical_p(np.asarray(null), np.array([scan.maximum]))[0]),', '''"scan_p": float(pvalues.min()),
                             "aggregation": "calibrated_family_scan_maxima_bonferroni",
                             "witness_family_count": scan.components.shape[1],''')
patch(p, '"calibration_resolution_ok": 1/(len(null)+1) <= alpha,', '"calibration_resolution_ok": scan.components.shape[1]/(len(null)+1) <= alpha,')
patch(p, '                outside = ~endpoint_in_event if not event["is_null"] else np.ones(len(alerts), bool)', '                outside = ((scan.ends <= lo) | (scan.starts >= hi)) if not event["is_null"] else np.ones(len(alerts), bool)')
patch(p, '                false_alerts = alerts & outside & evaluable[scan.ends-1]', '''                # Entire windows, not just endpoints, must be normal and outside recovery.
                excluded = np.r_[0, np.cumsum(~evaluable)]
                false_alerts = alerts & outside & ((excluded[scan.ends]-excluded[scan.starts]) == 0)''')
patch(p, '''The maximum over ALL configured windows/channels/pairs is calibrated on whole,
independent normal trajectories from the same target system.''', '''Each witness family's maximum over ALL configured windows/channels/pairs is
calibrated on whole independent normal trajectories from the same system.
Bonferroni corrects selection across witness families; raw units are not mixed.''')
p = 'gutenTAG/generator/group_relation_policy.py'
patch(p, '''    if kind == "shared-noise-sine":
        return int(max(1, min(int(transition_length), 6)))
''', '')
patch(p, '''    kind = str(bo.get_base_oscillation_kind())
    return int(max(0, transition_length))''', '''    return int(max(0, transition_length))''')
patch(p, '    kind = str(bo.get_base_oscillation_kind())\n    if not np.isfinite', '    if not np.isfinite')
patch('gutenTAG/generator/group_covariance_change.py', 'max(0, min(state.transition_length, 8))', 'max(0, state.transition_length)')
patch('tests/test_group_relation_policy.py', 'effective_latent_transition_length(bo, 20), 6', 'effective_latent_transition_length(bo, 20), 20')
patch('tests/test_group_relation_policy.py', 'effective_latent_transition_length(bo, 0), 1', 'effective_latent_transition_length(bo, 0), 0')
patch('gutenTAG/tsgen/capabilities/dataset.py', '''        manifest = load_json(manifest_path)
    metadata_events = (''', '''        manifest = load_json(manifest_path)
        if manifest.get("schema_version") == "synthgen.v13.1":
            raise ValueError("v13.1 law datasets require evaluate_benchmark; legacy capability discovery would lose observation masks and oracle boundaries")
    metadata_events = (''')
patch('tests/test_v13_audit_regressions.py', 'calibration_realizations=9,', 'calibration_realizations=49,')
f = r/'tests/test_v13_audit_regressions.py'
f.write_text(f.read_text()+'''


def test_gp_offset_is_added_exactly_once_and_scale_not_forced():
    from gutenTAG.base_oscillations.gp_mixture import GaussianProcessMixture
    from gutenTAG.generator.base_channels import apply_variations
    def generate(offset, seed):
        bo = GaussianProcessMixture(length=100, offset=offset, gp_features=12, gp_components=2)
        x = bo.generate_only_base(SimpleNamespace(rng=np.random.default_rng(seed)))
        return apply_variations(x[:, None], [bo])[:, 0]
    np.testing.assert_allclose(generate(3.25, 4)-generate(0, 4), 3.25)
    assert np.std([generate(0, s).std() for s in range(8)]) > .05


def test_linear_trend_is_not_annihilated_by_default():
    from gutenTAG.anomalies.types.trend import AnomalyTrend, AnomalyTrendParameters
    t = AnomalyTrend(AnomalyTrendParameters(trend=None))
    x = np.linspace(2, 5, 33)
    np.testing.assert_allclose(t._bounded_local_trend(x, np.ones_like(x)), x-x[0])


def test_clone_is_exact_without_sharing_mutable_arrays():
    from gutenTAG.tsgen.paired_instance_generation import _generate_anomalous_base_series
    state = SimpleNamespace(base_values=np.arange(12).reshape(6, 2).astype(float),
                            channel_bos=[SimpleNamespace(noise=np.ones(6))])
    clone = _generate_anomalous_base_series(config=None, variant=None, seeds={},
                                           base_instance=SimpleNamespace(series=state),
                                           effective_base_channel_correlation={})
    np.testing.assert_array_equal(clone.base_values, state.base_values)
    clone.base_values[0, 0] = 999
    clone.channel_bos[0].noise[0] = 999
    assert state.base_values[0, 0] == 0 and state.channel_bos[0].noise[0] == 1


def test_empty_effect_support_and_unknown_resample_policy():
    from gutenTAG.tsgen.labels.effective_support import resolve_label_bounds_from_effect, normalize_subsequence_length
    assert resolve_label_bounds_from_effect(protocol_start=8, protocol_end=16, delta=np.zeros(8),
        anomaly_type="mean", support_label_mode="effective_support", support_eps_mode="relative",
        support_eps_value=.03, min_effective_label_length_non_extremum=2) == (8, 8)
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
        apply_interventions(x, [Intervention("covariance-change", 16, 32, .5, (0,)),
                                 Intervention("clipping", 64, 96, .5, (0,))], seed=9)


def test_v13_cannot_be_silently_discovered_by_legacy_loader(tmp_path):
    from gutenTAG.tsgen.capabilities.dataset import discover_dataset
    (tmp_path / "dataset_manifest.json").write_text('{"schema_version": "synthgen.v13.1"}')
    with pytest.raises(ValueError, match="evaluate_benchmark"):
        discover_dataset(tmp_path)


def test_streamed_annotation_manifest_hashes_final_bytes(tmp_path):
    from gutenTAG.tsgen.sidecars import write_dataset_annotation_channels, _file_hash
    from gutenTAG.tsgen.capabilities.dataset import DatasetIndex
    dataset = DatasetIndex(root=tmp_path, manifest={}, instances=(), metadata_events={}, problem_genotypes={})
    result = write_dataset_annotation_channels(dataset, config={"emit": ["oracle"]})
    path = tmp_path / result["manifest_path"]
    assert result["manifest_hash"] == _file_hash(path)
    assert "manifest_hash" not in json.loads(path.read_text())
''')
print('Applied final regression corrections')
