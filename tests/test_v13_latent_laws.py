import numpy as np

from gutenTAG.base_oscillations import BaseOscillation
from gutenTAG.generator.latent_laws import nearest_correlation, sample_lmc_correlation


def test_gp_mixture_is_registered():
    assert "gp-mixture" in BaseOscillation.key_mapping


def test_nearest_correlation_is_positive_definite():
    raw = np.array([[1.0, 1.4, -0.8], [1.4, 1.0, 0.9], [-0.8, 0.9, 1.0]])
    corr = nearest_correlation(raw)
    np.testing.assert_allclose(np.diag(corr), np.ones(3))
    assert np.min(np.linalg.eigvalsh(corr)) > 0.0


def test_lmc_sampler_is_deterministic_and_nontrivial():
    first_loadings, first_corr = sample_lmc_correlation(
        5, np.random.default_rng(123), rank=3, sparsity=0.2, idiosyncratic=0.2
    )
    second_loadings, second_corr = sample_lmc_correlation(
        5, np.random.default_rng(123), rank=3, sparsity=0.2, idiosyncratic=0.2
    )
    np.testing.assert_allclose(first_loadings, second_loadings)
    np.testing.assert_allclose(first_corr, second_corr)
    assert np.max(np.abs(first_corr - np.eye(5))) > 0.05
    assert np.min(np.linalg.eigvalsh(first_corr)) > 0.0
