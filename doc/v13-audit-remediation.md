# V13 audit remediation: implementation and evidence

This PR is stacked on the experimental v13 PR. It corrects the audit findings;
it does not retrospectively certify the old candidate or rewrite historical data.
The companion synth-gen PR pins this library and runs the complete parent-checkout
suite plus an end-to-end pilot and an illustrated post-scoring report.

## Before / after

| Audit finding | Implementation | Regression / evidence |
|---|---|---|
| Split names falsely imply OOD/purity | Strict BenchmarkConfig and SplitSpec enforce IID priors, held-out families/mechanisms, disjoint meaningful parameter domains and relation-only compatibility | test_split_contracts_reject_fake_ood_and_relation_only |
| Calibration silently includes test counterparts | Required legacy split fails closed; v13 uses independent target-normal calibration trajectories | test_required_calibration_never_falls_back_to_test; complete_generation_evaluation |
| Event windows called independent law replicates | Historical sidecar explicitly renamed; v13 freezes complete law and genotype, then independently samples each replica | immutable_law_roundtrip; serialized manifest IDs/seeds |
| LMC only correlates measurement noise | Explicit temporal GP-LMC family; white-noise compatibility adapter is separately named | law roundtrip for gp-lmc; genuine shared temporal factor construction |
| LMC operators fall back to observed rewrites | Stored immutable NoiseLawState is consumed by covariance/correlation operators; unsupported combinations fail rather than pretend purity | test_lmc_state_is_shared_and_operator_uses_it |
| GP offset applied twice | Carrier returns amplitude-scaled signal only; observation composition owns offset | test_gp_offset_is_added_exactly_once_and_scale_not_forced |
| Per-realization normalization alters the stated GP law | Removed; finite spectra and Gaussian coefficients have separate law/realization identities | spectrum tests; block invariance; variance not forced |
| RQ / angular-frequency conventions wrong | Explicit covariance and spectrum implementations for RBF, Matern32, RQ, periodic and locally-periodic | test_spectrum_matches_named_kernel for all kernels |
| Polynomial list sampled as scalar | Literal coefficients and explicit literal primitive, including nested configurations | test_polynomial_coefficients_and_literal_arrays_are_not_ranges |
| Rejection sampler fabricates impossible values | Exhaustion raises; no out-of-prior threshold injection | test_impossible_rejection_cannot_fabricate_out_of_prior_value |
| Variance decrease actually adds noise | Explicit std ratio reuses innovations; legacy target-std replacement consistent; no hidden carrier effect floor | noise_ratio_identity_and_negative_strength; full legacy regression suite |
| Linear trend is annihilated | Shape-preserving default; old zero-endpoint behavior explicit; parser now forwards options | test_linear_trend_is_not_annihilated_by_default; golden test retains explicit legacy contract |
| Hidden .92/.95 severity and transition floors | Requested parameters retained; no carrier-driven strengthening/caps | group_relation_policy tests |
| Sensor helpers disconnected / no true null / wrong masks | Observation-layer interventions are in the v13 generation path; zero is identity, masks follow finiteness, last-value anchors and aligned timestamps | all-mechanism null tests; sensor null/mask tests |
| Absolute correlation erases a sign flip | Signed pair features in legacy law profile | test_signed_law_witness_distinguishes_flip |
| Oracle scan presented as blind | Legacy scan explicitly time-blind/channel-oracle/duration-oracle; separate v13 fixed-grid all-channel scan | oracle-poison and channel-permutation tests |
| Mixed raw-score units | Each family trajectory maximum gets its own empirical p-value; Bonferroni across families | rank tie test; recorded aggregation and finite calibration resolution |
| Pseudoreplicated uncertainty / preprocessing leakage | Paired group-held-out C2ST, training-fold scaling, paired cluster bootstrap; report averages by system | test_cluster_uncertainty_and_identical_counterfactual_null |
| Missing fingerprint features silently zeroed | Complete finite measurements required; system-group holdout and sufficient-statistic centroids | test_missing_nuisance_features_fail_closed |
| Zero effect labels full event | Realized effect support may be empty; intervention and evaluation masks are separate | test_empty_effect_support_and_unknown_resample_policy |
| Dataset-wide sidecar DataFrames | Stream one instance at a time into temporary CSVs; replace final outputs | complete legacy suite; no corpus-wide frame list |
| Normal GP generated twice | Deep-copy mutable realized arrays, share immutable law once | test_clone_is_exact_without_sharing_mutable_arrays |
| C copies of CxC LMC matrices | One shared read-only state; direct low-rank sampling | shared state test |
| Dense repeated distances / centroid computation | Blocked distances, cached cluster sums, sufficient statistics | grouped statistic tests and preserved results |
| Manifest hashes intermediate bytes | Only external parent hashes final manifest bytes | test_streamed_annotation_manifest_hashes_final_bytes |
| v13 accidentally loaded as empty legacy dataset | Legacy loader and config reject law schema and direct users to new API | test_v13_cannot_be_silently_discovered_by_legacy_loader |
| Research-branch CI absent | All-PR standalone matrix, parent integration, exact snapshots, type delta and lint | Actions artifacts, not inferred success |

## The information contract

```text
fixed law_id ── reference realization ───────┐
             ├─ calibration realizations ──┼─ fixed blind detector/threshold
             └─ query realizations ─────────┘
                   ├─ queries/: values, masks, timestamps
                   └─ oracle/: clean counterparts, supports, descriptions
                                 (opened only after scoring)
```

Law parameters are generator/oracle information, not detector inputs. Target
normal reference/calibration access is explicit adaptation, not query-only
zero-shot. Splits separate laws; replicas separate trajectories; event IDs never
stand in for independent systems. Seeds use an unambiguous 128-bit namespace.

## Exact relation-only construction

For diagonal stable A, X_t = A X_(t-1) + epsilon_t. Normal innovations are iid
Gaussian with covariance Sigma0. The intervention uses Sigma_t =
(1-g_t) Sigma0 + g_t Sigma1, with both endpoint matrices SPD and the same diagonal.
Each marginal innovation sequence retains the same iid Gaussian law. With the
same normal stationary marginal initial law, every one-channel trajectory law
is unchanged, including event boundaries. Joint covariance can change.

The covariance endpoint uses sign conjugation; the precision endpoint uses an
SPD rank-one precision update followed by diagonal normalization. Tests verify
fixed diagonals and SPD over the interpolation, clean-branch immutability and
null identity. There is no per-event fitting, histogram matching or detector-
conditioned rejection. Actual sample trajectories need not match marginal
histograms exactly; equality is a law statement.

AR state is never forcibly reset at event end. A separately recorded recovery
exclusion makes labelled evaluation distinguish intervention support from
counterfactual tails. Exact purity is NOT asserted for changing temporal GP
loadings. Mixing innovation interventions with further observation events is
rejected until a causal composition contract is supplied, rather than silently
applying transformations in the wrong order. The shipped benchmark intentionally
uses one event per independent query replica.

## Calibration and interpretation

Every witness is maximized over a declared fixed window/channel/pair grid. Its
normal reference-conditional distribution is estimated with independent complete
normal trajectories from the same system. Empirical ranks include ties and the
finite-sample +1 correction. Bonferroni corrects across five witness families.
Thus alpha requires at least ceil(5/alpha)-1 calibration trajectories to be
resolvable. It does not supply a lifetime multiple-query guarantee or validity
under arbitrary drift. Long-event interior recall, boundaries and recovery are
not collapsed into point-adjusted scores.

A miss by this simple detector is not grounds to strengthen or discard an event.
The report is a diagnostic, not automatic scientific admission. Nuisance
classification needs mechanism-specific interpretation: impulses may legitimately
have sharp boundaries; covariance-only events must not depend on them.

## Compatibility and testing

The new schema is `synthgen.v13.1` and uses `generate_benchmark`. Existing v12
commands remain, but corrected polynomial, GP, trend and variance semantics can
change freshly generated data. Old stored datasets remain untouched. Explicit
zero-endpoint trend config preserves the historical golden test; silently ignored
trend options in the old parser are now passed through.

Local full parent-checkout run: **538 passed, 44 subtests passed**, one historical
All-NaN TimeEval warning. A pilot produced **16 independent systems and 222 queries**
with resolvable alpha=0.1. This is not execution of the larger full workload.

Mypy was compared base/head in the SAME local environment: 107 baseline diagnostics,
105 after remediation, no added diagnostics (line numbers excluded, messages not
suppressed). Full typing debt is not claimed resolved. CI repeats this comparison
and publishes both full logs. The standalone library CI excludes two modules
that import parent scripts; the companion parent CI executes them in the correct
checkout rather than counting exclusions as passes.

GP spectra are finite-feature approximations of their named target kernels;
conditional on stored spectra, realizations are Gaussian. This is not a claim
of universal realism, preservation of all nonlinear/higher-order alternatives,
or empirical superiority over real-world benchmarks. Future process families
must satisfy the same explicit contracts and tests rather than inherit admission
from the presence of an API or a filename.
