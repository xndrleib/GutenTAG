"""Anomaly parameter-policy helpers for TS dataset generation."""

from .effects import (
    apply_amplitude_parameter_policy,
    apply_mean_parameter_policy,
    apply_trend_parameter_policy,
)
from .frequency import apply_period_locked_frequency_policy
from .local_stats import annotate_segment_local_stats, window_residual_scale
from .resolution import (
    resolve_anomaly_parameters_for_segments,
    resolve_instance_anomaly_parameters,
)
from .sampling import realize_parameters, sample_between
from .sanitization import sanitize_anomaly_parameters
from .transitions import (
    apply_transition_policy_to_segments,
    sample_transition_span,
    transition_cap_for_anomaly,
)
from .variant_context import (
    VariantParameterContext,
    prepare_variant_parameter_context,
)
from .variant_templates import (
    classify_base_family,
    resolve_anomaly_parameters,
    resolve_base_channel_correlation,
    resolve_base_channel_parameters,
    resolve_base_parameters,
    resolve_variant_anomaly_policy,
)

__all__ = [
    "apply_amplitude_parameter_policy",
    "apply_mean_parameter_policy",
    "apply_period_locked_frequency_policy",
    "apply_trend_parameter_policy",
    "apply_transition_policy_to_segments",
    "classify_base_family",
    "annotate_segment_local_stats",
    "realize_parameters",
    "prepare_variant_parameter_context",
    "resolve_anomaly_parameters",
    "resolve_anomaly_parameters_for_segments",
    "resolve_instance_anomaly_parameters",
    "resolve_base_channel_correlation",
    "resolve_base_channel_parameters",
    "resolve_base_parameters",
    "resolve_variant_anomaly_policy",
    "sample_between",
    "sample_transition_span",
    "sanitize_anomaly_parameters",
    "transition_cap_for_anomaly",
    "VariantParameterContext",
    "window_residual_scale",
]
