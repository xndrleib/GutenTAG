"""One-shot, assertion-guarded application of the reviewed legacy audit patch.

This transport helper is removed after the resulting source commit is saved.
It is run only by the fixed same-repository remediation branch workflow.
"""
from pathlib import Path
r = Path(__file__).resolve().parents[1]


def patch(path, old, new):
    f = r/path
    text = f.read_text()
    if old not in text:
        raise ValueError(f"Patch precondition failed: {path}: {old[:80]}")
    f.write_text(text.replace(old, new))


patch('gutenTAG/tsgen/parameters/sampling.py',
'''        str(key): _realize_parameter_value(value, rng)
        for key, value in dict(template).items()''',
'''        str(key): (copy.deepcopy(value) if str(key) in {"polynomial", "coefficients"}
                   and isinstance(value, (list, tuple)) else _realize_parameter_value(value, rng))
        for key, value in dict(template).items()''')
patch('gutenTAG/tsgen/parameters/sampling.py', '''        mapping = dict(value)
        if "distribution" in mapping:''', '''        mapping = dict(value)
        if set(mapping) == {"literal"}:
            return copy.deepcopy(mapping["literal"])
        if "distribution" in mapping:''')
patch('gutenTAG/tsgen/parameters/sampling.py', '''        return {
            str(key): _realize_parameter_value(nested, rng)
            for key, nested in mapping.items()
        }''', '''        return realize_parameters(mapping, rng)''')
patch('gutenTAG/tsgen/parameters/sampling.py', '''    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(sign * threshold)''', '''    raise ValueError("Rejection sampler exhausted: requested effect is outside the prior support")''')
patch('gutenTAG/tsgen/capabilities/corrected_detectability_partitions.py',
'''        if preferred:
            return preferred
    return tuple(instances)''',
'''        if not preferred:
            raise ValueError(f"Required calibration split {protocol.calibration_split!r} is absent")
        return preferred
    # None is the explicit legacy/oracle diagnostic mode, not a deployment protocol.
    return tuple(instances)''')
patch('gutenTAG/tsgen/paired_instance_generation.py', 'import logging\n', 'import logging\nimport copy\n')
patch('gutenTAG/tsgen/paired_instance_generation.py', '''    return generate_base_instance_series(
        base_kind=variant.base_oscillation,
        base_parameters_per_channel=base_instance.base_parameters_per_channel,
        seed=int(seeds["base_seed"]),
        shared_noise_seed=int(seeds["base_shared_noise_seed"]),
        base_channel_correlation=effective_base_channel_correlation,
        expected_length=config.length,
    )''', '''    # Clone mutable arrays once, instead of repeating GP generation. Immutable
    # process laws implement __deepcopy__ by returning themselves.
    return copy.deepcopy(base_instance.series)''')
patch('gutenTAG/tsgen/paired_instance_generation.py', '    generate_base_instance_series,\n', '')
patch('gutenTAG/tsgen/labels/effective_support.py', '''    if active.size == 0:
        return int(protocol_start), int(protocol_end)''', '''    if active.size == 0:
        return int(protocol_start), int(protocol_start)''')
patch('gutenTAG/tsgen/labels/effective_support.py', '''    if expected_length <= 0:
        return np.array([], dtype=np.float64)''', '''    if policy not in {"none", "crop", "pad", "resample"}:
        raise ValueError(f"Unknown length normalization policy: {policy}")
    if expected_length <= 0:
        return np.array([], dtype=np.float64)''')
patch('gutenTAG/generator/group_relation_policy.py', '''    if kind == "shared-noise-sine":
        return float(
            np.sign(target_correlation) * max(abs(float(target_correlation)), 0.92)
        )''', '''    if not np.isfinite(target_correlation) or abs(target_correlation) > 1:
        raise ValueError("target_correlation must be finite and in [-1, 1]")''')
patch('gutenTAG/generator/group_relation_policy.py', '''    if kind == "shared-noise-sine":
        return float(
            np.sign(coupling_strength) * max(abs(float(coupling_strength)), 0.95)
        )''', '''    if not np.isfinite(coupling_strength) or abs(coupling_strength) > 1:
        raise ValueError("coupling_strength must be finite and in [-1, 1]")''')
patch('gutenTAG/tsgen/capabilities/corrected_detectability_blind_scan.py',
'''"""Blind-scan scoring and row construction for corrected detectability."""''',
'''"""Time-blind, channel-oracle and duration-oracle diagnostic scans.

These legacy diagnostics receive event metadata. They are NOT fully blind
benchmark measurements; use tsgen.capabilities.v13_evaluation for that task.
"""''')
p = 'gutenTAG/anomalies/types/variance.py'
patch(p, '    variance: float = 0.0\n', '    variance: float = 0.0\n    noise_std_ratio: Optional[float] = None\n')
patch(p, '        self.variance = parameters.variance\n', '        self.variance = parameters.variance\n        self.noise_std_ratio = parameters.noise_std_ratio\n')
patch(p, '        target_std = max(0.0, float(self.variance) * reference_scale)', '''        if self.noise_std_ratio is not None:
            ratio = float(self.noise_std_ratio)
            if not np.isfinite(ratio) or ratio < 0:
                raise ValueError("noise_std_ratio must be finite and nonnegative")
            envelope = self.build_symmetric_envelope(length, self.transition_length)
            # Reuse innovations: ratio=1 is a true null, not resampled noise.
            base.noise[anomaly_protocol.start:anomaly_protocol.end] = (
                original_noise * (1.0 + envelope * (ratio - 1.0))
            )
            return
        target_std = max(0.0, float(self.variance) * reference_scale)''')
patch(p, '''        if base_kind in {"polynomial", "random-walk"}:
            return original_noise + envelope * subsequence_noise
        return original_noise + envelope * (subsequence_noise - original_noise)''', '''        return original_noise + envelope * (subsequence_noise - original_noise)''')
patch(p, '''        if base_kind in {"polynomial", "random-walk"}:
            effective_min_effect = max(effective_min_effect, 0.75 * reference_scale)
''', '')
p = 'gutenTAG/anomalies/types/trend.py'
patch(p, '    boundary_mode: str = "inside_window_zero_endpoints"', '    boundary_mode: str = "inside_window_preserve_shape"')
patch(p, '        local = self._anchor_zero_endpoints(local)', '''        if self.boundary_mode == "inside_window_zero_endpoints":
            local = self._anchor_zero_endpoints(local)
        elif self.boundary_mode == "inside_window_preserve_shape":
            local -= local[0] if local.size else 0.0
        else:
            raise ValueError(f"Unknown trend boundary_mode: {self.boundary_mode}")''')
patch(p, '        if self.envelope_kind in ("sine2", "sin2") and local.shape[0] > 1:', '        if self.boundary_mode == "inside_window_zero_endpoints" and self.envelope_kind in ("sine2", "sin2") and local.shape[0] > 1:')
p = 'gutenTAG/tsgen/sidecars.py'
f = r/p; s = f.read_text()
a = s.index('    frames_by_channel:'); b = s.index('    manifest = annotation_channel_manifest()', a)
s = s[:a]+'''    table_paths: dict[str, str] = {}
    table_hashes: dict[str, str] = {}
    row_counts = {name: 0 for name in selected_channels}
    temp_paths = {name: labels_dir / f".{name}.csv.tmp" for name in selected_channels}
    for path in temp_paths.values():
        path.write_text("", encoding="utf-8")
    for instance in dataset.instances:
        events = load_json(instance.events_path)
        channels = build_annotation_channels(
            length=instance.length, channels=instance.channels, events=events,
        )
        for name in selected_channels:
            channel = channels[name]
            frame = _flatten_label_table(instance, channel.values, channel.columns)
            frame.to_csv(temp_paths[name], mode="a", index=False,
                         header=row_counts[name] == 0)
            row_counts[name] += len(frame)
    for name, temp in temp_paths.items():
        path = labels_dir / f"{name}.csv"
        temp.replace(path)
        table_paths[name] = _relative_path(path, dataset.root)
        table_hashes[name] = _file_hash(path)

'''+s[b:]
f.write_text(s)
patch(p, '''    write_json(manifest_path, manifest, sort_keys=True, indent=2)
    manifest["manifest_path"] = _relative_path(manifest_path, dataset.root)
    manifest["manifest_hash"] = _file_hash(manifest_path)
    write_json(manifest_path, manifest, sort_keys=True, indent=2)''', '''    manifest["manifest_path"] = _relative_path(manifest_path, dataset.root)
    write_json(manifest_path, manifest, sort_keys=True, indent=2)
    # Only the parent stores the final-byte hash: no self-referential hash.
    manifest["manifest_hash"] = _file_hash(manifest_path)''')
patch(p, '"generated_paired_event_window"', '"event_window_registry_not_independent_realizations"')
for path, name, attr in [('group_covariance_change.py', '_apply_covariance_rewrite', 'effective_strength'), ('group_correlation_flip.py', '_apply_correlation_rewrite', 'effective_target_correlation')]:
    f = r/'gutenTAG/generator'/path
    s = f.read_text().replace('from .group_context import', 'from .latent_laws import recouple_lmc_target\nfrom .group_context import')
    i = s.index('    latent = latent_shared_noise_attrs(', s.index('def '+name))
    text = f'''    lmc = recouple_lmc_target(
        bo=channel_bos[state.channel], anchor=context.anchor_channel,
        start=context.source_start, end=context.source_end,
        target_correlation=state.{attr}, transition=state.transition_length,
    )
    if lmc is not None:
        runtime.replace_noise(bo=channel_bos[state.channel], start=context.source_start,
                              end=context.source_end, target_noise=lmc)
        return "noise"
'''
    f.write_text(s[:i]+text+s[i:])
patch('gutenTAG/generator/group_anomalies.py', '    handler = _GROUP_ANOMALY_HANDLERS.get(anomaly_type)', '''    if any(hasattr(bo, "_process_state") for bo in channel_bos) and anomaly_type not in {"covariance-change", "correlation-flip"}:
        raise ValueError("Unsupported white-noise LMC operator; use the explicit v13 process-law API, not an observed-window fallback")
    handler = _GROUP_ANOMALY_HANDLERS.get(anomaly_type)''')
p = 'gutenTAG/tsgen/capabilities/law_observability.py'
patch(p, 'from .array_store import ArrayStore', 'from .array_store import ArrayStore\nfrom .grouped_statistics import paired_c2st, cluster_energy_interval\nfrom scipy.spatial.distance import cdist, pdist')
patch(p, '        clean_features: list[np.ndarray] = []', '        group_ids: list[str] = []\n        clean_features: list[np.ndarray] = []')
patch(p, '            for group in instance.event_groups:\n', '            for group in instance.event_groups:\n                group_ids.append(f"{instance.variant_id}/{instance.split}/{instance.instance_id}")\n')
patch(p, '                bootstrap_samples=protocol.bootstrap_samples,', '                bootstrap_samples=protocol.bootstrap_samples,\n                groups=np.asarray(group_ids),')
patch(p, '    rng: np.random.Generator,\n) -> dict[str, object]:', '    rng: np.random.Generator,\n    groups: np.ndarray | None = None,\n) -> dict[str, object]:')
patch(p, '    energy = _energy_distance(clean, anomalous)', '    raw_clean, raw_anomalous = clean, anomalous\n    clean, anomalous = _standardize_pair(clean, anomalous)\n    energy = _energy_distance(clean, anomalous)')
patch(p, '    c2st = _nearest_centroid_balanced_accuracy(clean, anomalous)', '    c2st = paired_c2st(raw_clean, raw_anomalous, groups)')
patch(p, '        samples=int(bootstrap_samples),', '        samples=int(bootstrap_samples),\n        groups=groups,')
patch(p, '        "feature_space": "event_window_summary",', '        "feature_space": "signed_relations_event_summary_v2",\n        "independent_group_count": int(len(np.unique(groups))) if groups is not None else len(clean),\n        "uncertainty_unit": "paired_instance_cluster",')
patch(p, '    corr_proxy = 0.0', '    signed_pairs = np.zeros(matrix.shape[1]*(matrix.shape[1]-1)//2)\n    corr_proxy = 0.0')
patch(p, '        corr_proxy = float(np.nanmean(np.abs(corr[mask])))', '        signed_pairs = np.nan_to_num(corr[np.triu_indices(corr.shape[0], 1)])\n        corr_proxy = float(np.nanmean(corr[mask]))')
patch(p, '    features[~np.isfinite(features)] = 0.0', '    features = np.r_[features, signed_pairs]\n    features[~np.isfinite(features)] = 0.0')
patch(p, '    return _standardize_pair(clean, anomalous)', '    return clean, anomalous')
f = r/p; s = f.read_text()
a = s.index('    diff = left[:, None, :] - right[None, :, :]\n', s.index('def _mean_pairwise_distance')); b = s.index('\n\n\ndef ', a)
s = s[:a]+'''    total = 0.0
    for i in range(0, len(left), 256):
        for j in range(0, len(right), 256):
            total += float(cdist(left[i:i+256], right[j:j+256]).sum())
    return total / (len(left)*len(right))'''+s[b:]
a = s.index('    distances = []', s.index('def _median_gamma')); b = s.index('    return 1.0 / max(median, 1e-8)', a)
s = s[:a]+'''    subset = values[np.linspace(0, len(values)-1, min(len(values), 1024), dtype=int)]
    distances = pdist(subset, metric="sqeuclidean")
    median = float(np.median(distances)) if distances.size else 1.0
'''+s[b:]
s = s.replace('    diff = left[:, None, :] - right[None, :, :]\n    sqdist = np.sum(np.square(diff), axis=2)', '    sqdist = cdist(left, right, metric="sqeuclidean")')
a = s.index('    if clean.shape[0] < 2', s.index('def _nearest_centroid_balanced_accuracy')); b = s.index('\n\ndef _nearest_label', a)
s = s[:a]+'    return paired_c2st(clean, anomalous)\n'+s[b:]
a = s.index('def _bootstrap_ci('); b = s.index('\n\ndef _law_status', a)
s = s[:a]+'''def _bootstrap_ci(
    clean: np.ndarray, anomalous: np.ndarray, *, samples: int,
    rng: np.random.Generator, groups: np.ndarray | None = None,
) -> tuple[float, float]:
    group = np.arange(len(clean)) if groups is None else groups
    return cluster_energy_interval(clean, anomalous, group, samples=samples, rng=rng)
'''+s[b:]
a = s.index('    k_xx = _rbf_kernel(clean, clean, gamma)'); b = s.index('\n\ndef _median_gamma', a)
s = s[:a]+'''    def mean_kernel(left, right):
        total = 0.0
        for i in range(0, len(left), 256):
            for j in range(0, len(right), 256):
                total += float(_rbf_kernel(left[i:i+256], right[j:j+256], gamma).sum())
        return total / (len(left)*len(right))
    return float(max(0.0, mean_kernel(clean, clean) + mean_kernel(anomalous, anomalous)
                     - 2*mean_kernel(clean, anomalous)))
'''+s[b:]
s = s.replace('        "partition_size": int(partition_size),', '        "partition_size": int(partition_size),\n        "law_feature_version": "signed_relations_cluster_v2",')
f.write_text(s)
# Regression expectations for intentionally changed contracts. No test removed.
p = 'tests/test_parameter_sampling.py'; f = r/p; s = f.read_text()
a = s.index('    def test_reject_if_abs_lt_falls_back_to_threshold'); b = s.index('    def test_rejects_invalid_distribution_specs', a)
s = s[:a]+'''    def test_reject_if_abs_lt_rejects_impossible_prior(self) -> None:
        with self.assertRaisesRegex(ValueError, "exhausted"):
            realize_parameters(
                {"offset": {"distribution": "reject_if_abs_lt", "threshold": 0.25, "base": 0.0}},
                np.random.default_rng(3),
            )

'''+s[b:]; f.write_text(s)
p = 'tests/test_corrected_detectability_partitions.py'; f = r/p; s = f.read_text()
if 'import pytest' not in s:
    s = 'import pytest\n'+s
    # This file has no __future__ statement in the pinned base.
a = s.index('def test_calibration_null_instances_falls_back_to_variant_instances'); b = s.index('\n\ndef ', a+5)
s = s[:a]+'''def test_missing_explicit_calibration_split_fails_closed() -> None:
    from types import SimpleNamespace
    instances = (SimpleNamespace(split="train"), SimpleNamespace(split="test"))
    with pytest.raises(ValueError, match="absent"):
        calibration_null_instances(instances, CapabilityProtocol(calibration_split="missing"))
    assert calibration_null_instances(instances, CapabilityProtocol()) == instances
'''+s[b:]; f.write_text(s)
patch('tests/test_paired_instance_generation.py', '"generate_base_instance_series": fake_generate_base_instance_series', '"_generate_anomalous_base_series": fake_generate_base_instance_series')
patch('tests/test_paired_instance_generation.py', 'assert calls["base_series"]["base_kind"] == "sine"', 'assert calls["base_series"]["variant"].base_oscillation == "sine"')
p = 'tests/test_group_relation_policy.py'
patch(p, 'test_shared_noise_sine_strengthens_relation_targets', 'test_shared_noise_sine_preserves_requested_relation_targets')
patch(p, 'target_correlation=-0.2), -0.92', 'target_correlation=-0.2), -0.2')
patch(p, 'coupling_strength=0.3), 0.95', 'coupling_strength=0.3), 0.3')
patch('tests/test_ts_dataset_group_relations.py', 'self.assertLess(anomalous_corr, -0.5)', '''# No hidden target amplification: require the requested direction,
                    # not the obsolete hard-coded -0.92 target floor.
                    self.assertLess(anomalous_corr, clean_corr - 0.3)''')
patch('tests/test_v13_benchmark_contracts.py', 'rows.append({"source_length": length + delta * 0.01})', 'rows.append({"source_length": length + delta * 0.01, "effective_length": length, "boundary_jump": 0., "derivative_jump": 0., "realized_density": .1, "transition_length": 0.})')
patch('tests/configs/example-config-trend-anomaly.yaml', '          - kind: trend\n', '          - kind: trend\n            boundary_mode: inside_window_zero_endpoints\n')
p = 'gutenTAG/tsgen/processes/kernels.py'
patch(p, '    alpha: float = 1.0', '    alpha: float = 1.0\n    periodic_smoothness: float = 1.0')
patch(p, '("length_scale", "period", "alpha")', '("length_scale", "period", "alpha", "periodic_smoothness")')
patch(p, '/ self.length_scale**2)', '/ self.periodic_smoothness**2)')
patch(p, 'k = 0.5 / self.length_scale**2', 'k = 0.5 / self.periodic_smoothness**2')
print('Applied all assertion-guarded audit changes')
