"""Integer parameters of the public functions reject floats, strings and None with ValueError.

Each case calls a function with valid arguments, then replaces one integer parameter at a time. Parameters whose
default is None keep accepting None. The swept parameters (``k``, ``lag``, ``window_size``, ``k_values``) accept
integer-valued numbers such as ``[4.0, 8.0]``.
"""

import numpy
import pytest
import torch

from tests.conftest import pseudo_random_series
from torchnlts import (
    causality,
    chaos,
    dimension,
    embedding,
    entropy,
    fluctuations,
    lyapunov,
    matrixprofile,
    neighbors,
    recurrence,
    sciml,
    surrogate,
)

SERIES = pseudo_random_series(80, 1)
OTHER = pseudo_random_series(80, 2)
THIRD = pseudo_random_series(80, 3)
RECURRENCE = recurrence.hard_recurrence_matrix(SERIES, epsilon=0.5)
PROFILE, INDEX = matrixprofile.matrix_profile(SERIES, window=8)
ARC_CURVE = matrixprofile.corrected_arc_curve(INDEX, window=8)
DISTANCE = neighbors.pairwise_distance(SERIES)
BAD_INTEGERS = [2.0, 1.5, "2", None, numpy.float64(2.0), [2, 3]]


def _henon(state):
    return torch.stack([1.0 - 1.4 * state[0] ** 2 + state[1], 0.3 * state[0]])


def _recurrence_measure(name):
    return lambda **kwargs: getattr(recurrence, name)(RECURRENCE, **kwargs)


# name: (call, valid keyword arguments, parameters for which None is valid)
CASES = {
    "cross_map_estimate": (
        lambda **kwargs: causality.cross_map_estimate(SERIES, OTHER, **kwargs),
        dict(embedding_dim=2, delay=1, library_size=20, theiler_window=0),
        {"library_size"},
    ),
    "conditional_mutual_information": (
        lambda **kwargs: causality.conditional_mutual_information(SERIES, OTHER, THIRD, **kwargs),
        dict(num_neighbors=3, theiler_window=0),
        set(),
    ),
    "transfer_entropy": (
        lambda **kwargs: causality.transfer_entropy(SERIES, OTHER, **kwargs),
        dict(embedding_dim=1, delay=1, prediction_lag=1, num_neighbors=3, theiler_window=0),
        set(),
    ),
    "zero_one_growth_rate": (
        lambda **kwargs: chaos.zero_one_growth_rate(SERIES, [0.5, 1.0], **kwargs),
        dict(max_lag=5),
        {"max_lag"},
    ),
    "zero_one_test": (lambda **kwargs: chaos.zero_one_test(SERIES, [0.5, 1.0], **kwargs), dict(max_lag=5), {"max_lag"}),
    "correlation_sum": (
        lambda **kwargs: dimension.correlation_sum(SERIES, 0.5, **kwargs),
        dict(theiler_window=0),
        set(),
    ),
    "soft_correlation_sum": (
        lambda **kwargs: dimension.soft_correlation_sum(SERIES, 0.5, **kwargs),
        dict(theiler_window=0),
        set(),
    ),
    "cao_e1": (
        lambda **kwargs: embedding.cao_e1(SERIES, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "cao_e2": (
        lambda **kwargs: embedding.cao_e2(SERIES, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "delay_embedding": (
        lambda **kwargs: embedding.delay_embedding(SERIES, **kwargs),
        dict(embedding_dim=2, delay=1),
        set(),
    ),
    "autocorrelation": (lambda **kwargs: embedding.autocorrelation(SERIES, **kwargs), dict(max_delay=5), set()),
    "auto_mutual_information": (
        lambda **kwargs: embedding.auto_mutual_information(SERIES, **kwargs),
        dict(max_delay=5, num_bins=8),
        set(),
    ),
    "estimate_delay": (
        lambda **kwargs: embedding.estimate_delay(SERIES, **kwargs),
        dict(max_delay=5, num_bins=8),
        set(),
    ),
    "hard_fnn": (
        lambda **kwargs: embedding.hard_fnn(SERIES, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "approximate_entropy": (
        lambda **kwargs: entropy.approximate_entropy(SERIES, 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=-1),
        set(),
    ),
    "soft_approximate_entropy": (
        lambda **kwargs: entropy.soft_approximate_entropy(SERIES, 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=-1),
        set(),
    ),
    "dispersion_entropy": (
        lambda **kwargs: entropy.dispersion_entropy(SERIES, **kwargs),
        dict(num_classes=4, embedding_dim=2, delay=1),
        set(),
    ),
    "fuzzy_entropy": (
        lambda **kwargs: entropy.fuzzy_entropy(SERIES, 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "coarse_grain": (lambda **kwargs: entropy.coarse_grain(SERIES, **kwargs), dict(scale=2, offset=0), set()),
    "multiscale_entropy": (
        lambda **kwargs: entropy.multiscale_entropy(SERIES, [1, 2], 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "soft_multiscale_entropy": (
        lambda **kwargs: entropy.soft_multiscale_entropy(SERIES, [1, 2], 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "composite_multiscale_entropy": (
        lambda **kwargs: entropy.composite_multiscale_entropy(SERIES, [1, 2], 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "soft_composite_multiscale_entropy": (
        lambda **kwargs: entropy.soft_composite_multiscale_entropy(SERIES, [1, 2], 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "mutual_information": (
        lambda **kwargs: entropy.mutual_information(SERIES, OTHER, **kwargs),
        dict(num_neighbors=3, theiler_window=0),
        set(),
    ),
    "ordinal_pattern_probabilities": (
        lambda **kwargs: entropy.ordinal_pattern_probabilities(SERIES, **kwargs),
        dict(embedding_dim=3, delay=1),
        set(),
    ),
    "permutation_entropy": (
        lambda **kwargs: entropy.permutation_entropy(SERIES, **kwargs),
        dict(embedding_dim=3, delay=1),
        set(),
    ),
    "sample_entropy": (
        lambda **kwargs: entropy.sample_entropy(SERIES, 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "soft_sample_entropy": (
        lambda **kwargs: entropy.soft_sample_entropy(SERIES, 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "dfa_fluctuation": (lambda **kwargs: fluctuations.dfa_fluctuation(SERIES, [8, 16], **kwargs), dict(order=1), set()),
    "eckmann_spectrum": (
        lambda **kwargs: lyapunov.eckmann_spectrum(SERIES, **kwargs),
        dict(embedding_dim=3, matrix_dim=2, num_neighbors=None, delay=1, theiler_window=0),
        {"num_neighbors"},
    ),
    "kantz_divergence": (
        lambda **kwargs: lyapunov.kantz_divergence(SERIES, [0, 1, 2], 0.5, **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "rosenstein_divergence": (
        lambda **kwargs: lyapunov.rosenstein_divergence(SERIES, [0, 1, 2], **kwargs),
        dict(embedding_dim=2, delay=1, theiler_window=0),
        set(),
    ),
    "lyapunov_spectrum": (
        lambda **kwargs: lyapunov.lyapunov_spectrum(_henon, torch.tensor([0.1, 0.2], dtype=torch.float64), **kwargs),
        dict(num_steps=5, num_exponents=None, transient_steps=0),
        {"num_exponents"},
    ),
    "matrix_profile": (
        lambda **kwargs: matrixprofile.matrix_profile(SERIES, **kwargs),
        dict(window=8, exclusion_zone=None),
        {"exclusion_zone"},
    ),
    "motifs": (
        lambda **kwargs: matrixprofile.motifs(PROFILE, INDEX, **kwargs),
        dict(window=8, num_picks=1, exclusion_zone=None),
        {"exclusion_zone"},
    ),
    "motifs_given_exclusion_zone": (
        lambda **kwargs: matrixprofile.motifs(PROFILE, INDEX, **kwargs),
        dict(window=8, num_picks=1, exclusion_zone=0),
        {"exclusion_zone"},
    ),
    "discords": (
        lambda **kwargs: matrixprofile.discords(PROFILE, **kwargs),
        dict(window=8, num_picks=1, exclusion_zone=None),
        {"exclusion_zone"},
    ),
    "discords_given_exclusion_zone": (
        lambda **kwargs: matrixprofile.discords(PROFILE, **kwargs),
        dict(window=8, num_picks=1, exclusion_zone=0),
        {"exclusion_zone"},
    ),
    "corrected_arc_curve": (
        lambda **kwargs: matrixprofile.corrected_arc_curve(INDEX, **kwargs),
        dict(window=8, exclusion_factor=5),
        set(),
    ),
    "regime_boundaries": (
        lambda **kwargs: matrixprofile.regime_boundaries(ARC_CURVE, **kwargs),
        dict(window=8, num_regimes=2, exclusion_factor=5),
        set(),
    ),
    "knn_search": (
        lambda **kwargs: neighbors.knn_search(SERIES, **kwargs),
        dict(num_neighbors=2, theiler_window=0),
        set(),
    ),
    "radius_neighbors": (
        lambda **kwargs: neighbors.radius_neighbors(SERIES, 0.5, **kwargs),
        dict(theiler_window=0),
        set(),
    ),
    "theiler_window_mask": (
        lambda **kwargs: neighbors.theiler_window_mask(**kwargs),
        dict(num_points=5, window=1),
        set(),
    ),
    "apply_theiler_window": (
        lambda **kwargs: neighbors.apply_theiler_window(DISTANCE, **kwargs),
        dict(window=1),
        set(),
    ),
    "recurrence_rate": (_recurrence_measure("recurrence_rate"), dict(theiler_window=0), set()),
    "rt_entropy": (_recurrence_measure("rt_entropy"), dict(theiler_window=0), set()),
    "windowed_rqa": (
        lambda **kwargs: recurrence.windowed_rqa(RECURRENCE, recurrence.recurrence_rate, **kwargs),
        dict(window_size=20, step=1),
        set(),
    ),
    "iaaft_surrogate": (lambda **kwargs: surrogate.iaaft_surrogate(SERIES, **kwargs), dict(max_iterations=5), set()),
    "surrogate_test": (
        lambda **kwargs: surrogate.surrogate_test(
            SERIES, lambda batch: batch.mean(dim=(1, 2)), surrogate.phase_randomized_surrogate, **kwargs
        ),
        dict(num_surrogates=9),
        set(),
    ),
    "LyapunovLoss": (
        lambda **kwargs: sciml.LyapunovLoss(0.4, **kwargs)(_henon, torch.tensor([0.1, 0.2], dtype=torch.float64)),
        dict(num_steps=5, num_exponents=None, transient_steps=0),
        {"num_exponents"},
    ),
    "CorrelationDimensionLoss": (
        lambda **kwargs: sciml.CorrelationDimensionLoss(1.0, torch.linspace(0.3, 1.5, 5), **kwargs)(SERIES),
        dict(theiler_window=0),
        set(),
    ),
    "RecurrenceLoss": (
        lambda **kwargs: sciml.RecurrenceLoss(0.2, 0.5, **kwargs)(SERIES),
        dict(theiler_window=0),
        set(),
    ),
    "SinkhornLoss": (lambda **kwargs: sciml.SinkhornLoss(0.5, **kwargs)(SERIES, OTHER), dict(num_iters=5), set()),
}
for _name in ("determinism", "dl_average", "dl_max", "dl_entropy", "laminarity", "trapping_time"):
    CASES[_name] = (_recurrence_measure(_name), dict(lmin=2, theiler_window=0), set())
for _name in ("vl_max", "vl_entropy", "ratio", "divergence"):
    CASES[_name] = (_recurrence_measure(_name), dict(lmin=2, theiler_window=0), set())


@pytest.mark.parametrize("name", list(CASES))
def test_integer_parameters_reject_floats_strings_and_none(name):
    call, valid, none_is_valid = CASES[name]
    call(**valid)
    for parameter in valid:
        for bad in BAD_INTEGERS:
            if bad is None and parameter in none_is_valid:
                continue
            with pytest.raises(ValueError, match=parameter):
                call(**{**valid, parameter: bad})


# name: (call with the swept argument, valid sweep)
SWEEPS = {
    "higuchi_length": (lambda k: dimension.higuchi_length(SERIES, k), [2, 4]),
    "power_variogram": (lambda lag: dimension.power_variogram(SERIES, lag), [2, 4]),
    "dfa_fluctuation": (lambda window_size: fluctuations.dfa_fluctuation(SERIES, window_size), [8, 16]),
    "kantz_divergence": (lambda k_values: lyapunov.kantz_divergence(SERIES, k_values, 0.5), [0, 2]),
    "rosenstein_divergence": (lambda k_values: lyapunov.rosenstein_divergence(SERIES, k_values), [0, 2]),
}


@pytest.mark.parametrize("name", list(SWEEPS))
def test_swept_integer_parameters_accept_integer_valued_numbers_only(name):
    call, sweep = SWEEPS[name]
    expected = call(sweep)
    for integer_valued in ([float(value) for value in sweep], torch.tensor(sweep, dtype=torch.float64), tuple(sweep)):
        torch.testing.assert_close(call(integer_valued), expected, equal_nan=True)
    for bad in ([sweep[0] + 0.5], sweep[0] + 0.5, "4", None, [], float("nan"), [sweep]):
        with pytest.raises(ValueError):
            call(bad)
