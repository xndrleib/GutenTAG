from __future__ import annotations

import numpy as np


def correlation_flip_window(
    reference: np.ndarray,
    anchor: np.ndarray,
    target_correlation: float | None = -0.85,
) -> np.ndarray:
    """Rewrite a window to match a target correlation while preserving mean/scale.

    The transformation keeps the local mean and Euclidean energy of ``reference``
    and changes only its geometric decomposition relative to ``anchor``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    anc = np.asarray(anchor, dtype=np.float64)
    if ref.size == 0 or anc.size == 0:
        return np.array(ref, dtype=np.float64, copy=True)
    n = min(ref.shape[0], anc.shape[0])
    ref = ref[:n]
    anc = anc[:n]

    ref_mean = float(np.mean(ref))
    ref_centered = ref - ref_mean
    anc_centered = anc - float(np.mean(anc))
    ref_norm = float(np.linalg.norm(ref_centered))
    anc_norm = float(np.linalg.norm(anc_centered))
    if ref_norm <= 1e-12 or anc_norm <= 1e-12:
        return np.array(ref, dtype=np.float64, copy=True)

    anchor_unit = anc_centered / anc_norm
    projection_coeff = float(np.dot(ref_centered, anchor_unit))
    orth = ref_centered - projection_coeff * anchor_unit
    orth_norm = float(np.linalg.norm(orth))
    if orth_norm <= 1e-12:
        aux = np.roll(anchor_unit, 1)
        aux = aux - float(np.dot(aux, anchor_unit)) * anchor_unit
        aux_norm = float(np.linalg.norm(aux))
        if aux_norm <= 1e-12:
            return np.array(ref, dtype=np.float64, copy=True)
        orth_unit = aux / aux_norm
    else:
        orth_unit = orth / orth_norm

    current_corr = float(np.dot(ref_centered, anc_centered) / (ref_norm * anc_norm))
    desired_corr = (
        -current_corr if target_correlation is None else float(target_correlation)
    )
    desired_corr = float(np.clip(desired_corr, -0.995, 0.995))
    orth_scale = float(np.sqrt(max(0.0, 1.0 - desired_corr**2)))
    candidate_centered = ref_norm * (
        desired_corr * anchor_unit + orth_scale * orth_unit
    )
    return (ref_mean + candidate_centered).astype(np.float64)
