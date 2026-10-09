"""Tests for torchnlts.matrixprofile.

Dual ground truth:
  1. Hand-derived values on tiny series (Yeh et al. 2016 distance, constant
     convention, FLUSS arc counts).
  2. stumpy: stump, motifs, fluss (with the parabola IAC passed as
     ``custom_iac``, since stumpy's default IAC is a fitted beta density).
"""

import math

import numpy as np
import pytest
import torch

from torchnlts.matrixprofile import corrected_arc_curve, discords, matrix_profile, motifs, regime_boundaries

stumpy = pytest.importorskip("stumpy")


def _series(seed, length=300, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.cumsum(torch.randn(length, generator=generator, dtype=dtype), dim=0) + torch.sin(
        torch.arange(length, dtype=dtype) / 5
    )


def _assert_index_matches(series, other, window, profile, index, expected_profile):
    """Index equals stumpy's, or (tie) points to a subsequence at the same distance."""
    base = series.numpy() if other is None else other.numpy()
    for query, neighbor in enumerate(index.tolist()):
        z_query = stumpy.core.z_norm(series.numpy()[query : query + window])
        z_neighbor = stumpy.core.z_norm(base[neighbor : neighbor + window])
        assert math.isclose(np.linalg.norm(z_query - z_neighbor), expected_profile[query], abs_tol=1e-6)


def test_hand_derived_anticorrelated_pair():
    # windows [0,1,2], [1,2,2], [2,2,1], [2,1,0]; zone 1 leaves j = 2, 3 for window 0.
    # rho(w0, w2) = -3 / sqrt(12) -> d = sqrt(6 + 3 sqrt(3)); rho(w0, w3) = -1 -> d = sqrt(12).
    profile, index = matrix_profile(torch.tensor([0.0, 1.0, 2.0, 2.0, 1.0, 0.0], dtype=torch.float64), 3)
    assert math.isclose(profile[0].item(), math.sqrt(6 + 3 * math.sqrt(3)), rel_tol=1e-12)
    assert index[0].item() == 2


def test_hand_derived_identical_shapes_have_zero_distance():
    # [0,1,2] and [0,2,4] are both linear ramps -> same z-normalization.
    profile, index = matrix_profile(torch.tensor([0.0, 1.0, 2.0, 9.0, 0.0, 2.0, 4.0], dtype=torch.float64), 3)
    assert profile[0].item() == pytest.approx(0.0, abs=1e-12) and index[0].item() == 4
    assert profile[4].item() == pytest.approx(0.0, abs=1e-12) and index[4].item() == 0


def test_constant_subsequence_convention():
    profile, index = matrix_profile(torch.tensor([2.0, 2.0, 2.0, 0.0, 1.0, 3.0], dtype=torch.float64), 3)
    assert profile[0].item() == pytest.approx(math.sqrt(3))  # constant vs non-constant
    profile, index = matrix_profile(torch.tensor([2.0, 2.0, 2.0, 7.0, 3.0, 3.0, 3.0], dtype=torch.float64), 3)
    assert profile[0].item() == 0.0 and index[0].item() == 4  # constant vs constant


def test_exclusion_zone_boundary():
    # x = [0,1,2,0,1,2]: w0 = w3 exactly. Zone 2 allows j = 3 (|0-3| > 2), zone 3 excludes it.
    series = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    profile, index = matrix_profile(series, 3, exclusion_zone=2)
    assert profile[0].item() == pytest.approx(0.0, abs=1e-12) and index[0].item() == 3
    profile, index = matrix_profile(series, 3, exclusion_zone=3)
    assert profile[0].item() == math.inf and index[0].item() == -1
    profile, index = matrix_profile(series, 3)  # default ceil(3/4) = 1
    assert index[0].item() == 3


@pytest.mark.parametrize("seed,window", [(0, 10), (1, 25), (2, 7)])
def test_self_join_matches_stumpy(seed, window):
    series = _series(seed)
    profile, index = matrix_profile(series, window)
    reference = stumpy.stump(series.numpy(), window)
    expected = reference[:, 0].astype(np.float64)
    assert np.allclose(profile.numpy(), expected, atol=1e-6)
    _assert_index_matches(series, None, window, profile, index, expected)


@pytest.mark.parametrize("seed,window", [(3, 12), (4, 30)])
def test_ab_join_matches_stumpy(seed, window):
    series, other = _series(seed, 250), _series(seed + 10, 180)
    profile, index = matrix_profile(series, window, other=other)
    reference = stumpy.stump(series.numpy(), window, other.numpy(), ignore_trivial=False)
    expected = reference[:, 0].astype(np.float64)
    assert profile.shape == (250 - window + 1,)
    assert np.allclose(profile.numpy(), expected, atol=1e-6)
    _assert_index_matches(series, other, window, profile, index, expected)


def test_constant_stretch_matches_stumpy():
    series = _series(5, 120)
    series[40:60] = 3.0
    profile, _ = matrix_profile(series, 8)
    expected = stumpy.stump(series.numpy(), 8)[:, 0].astype(np.float64)
    assert np.allclose(profile.numpy(), expected, atol=1e-6)


def test_input_shapes_and_dtype():
    series = _series(6, 80)
    profile, index = matrix_profile(series, 10)
    batched = torch.stack([series, _series(7, 80)])
    batch_profile, batch_index = matrix_profile(batched.unsqueeze(-1), 10)
    assert profile.shape == (71,) and batch_profile.shape == (2, 71) and batch_index.dtype == torch.int64
    assert torch.equal(batch_profile[0], profile) and torch.equal(batch_index[0], index)
    assert torch.equal(matrix_profile(series.unsqueeze(-1), 10)[0], profile)
    assert matrix_profile(series.float(), 10)[0].dtype == torch.float32
    assert matrix_profile(series, 10)[0].dtype == torch.float64
    assert torch.allclose(matrix_profile(series.float(), 10)[0].double(), profile, atol=1e-3)


def test_nonfinite_subsequences_are_unavailable():
    series = _series(8, 100)
    series[50] = float("nan")
    profile, index = matrix_profile(series, 10)
    bad = slice(41, 51)  # windows containing sample 50
    assert torch.isinf(profile[bad]).all() and (index[bad] == -1).all()
    good = torch.ones(91, dtype=torch.bool)
    good[bad] = False
    assert torch.isfinite(profile[good]).all()
    assert not ((index[good] >= 41) & (index[good] < 51)).any()
    expected = stumpy.stump(series.numpy(), 10)[:, 0].astype(np.float64)
    assert np.allclose(profile[good].numpy(), expected[good.numpy()], atol=1e-6)


def test_nonfinite_row_isolated_in_batch():
    series = _series(9, 60)
    broken = series.clone()
    broken[10] = float("inf")
    profile, index = matrix_profile(torch.stack([series, broken]).unsqueeze(-1), 8)
    clean_profile, clean_index = matrix_profile(series, 8)
    assert torch.equal(profile[0], clean_profile) and torch.equal(index[0], clean_index)
    assert torch.isinf(profile[1, 3:11]).all()


def test_gradcheck_profile():
    series = _series(10, 40).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda value: matrix_profile(value, 6)[0], (series,), atol=1e-6)


def test_gradcheck_ab_join():
    series, other = _series(11, 30).requires_grad_(True), _series(12, 25).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: matrix_profile(a, 5, other=b)[0], (series, other), atol=1e-6)


def test_gradient_isolated_across_batch_rows_and_nonfinite():
    series = _series(13, 50)
    broken = series.clone()
    broken[20] = float("nan")
    batch = torch.stack([series, broken, series * 2]).unsqueeze(-1).requires_grad_(True)
    profile, _ = matrix_profile(batch, 6)
    profile[0].sum().backward()
    assert torch.isfinite(batch.grad).all()
    assert batch.grad[1].abs().sum() == 0 and batch.grad[2].abs().sum() == 0
    flat = torch.zeros(30, dtype=torch.float64, requires_grad=True)
    matrix_profile(flat, 5)[0].sum().backward()
    assert torch.isfinite(flat.grad).all()


def test_validation():
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(20), 1)
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(20), 21)
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(20, 2), 5)
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(20), 5, exclusion_zone=-1)
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(2, 20, 1), 5, other=torch.randn(20, 1))


def _planted_series(window=20):
    generator = torch.Generator().manual_seed(0)
    series = torch.randn(400, generator=generator, dtype=torch.float64)
    pattern = torch.sin(torch.linspace(0, 2 * math.pi, window, dtype=torch.float64)) * 5
    series[60 : 60 + window] += pattern
    series[250 : 250 + window] += pattern * 0.9 + 1
    return series


def test_motifs_match_stumpy():
    series, window = _planted_series(), 20
    profile, index = matrix_profile(series, window)
    distances, pairs = motifs(profile, index, window, num_picks=3)
    reference_distances, reference_indices = stumpy.motifs(
        series.numpy(), profile.numpy(), max_distance=np.inf, cutoff=np.inf, max_matches=2, max_motifs=3
    )
    assert np.allclose(distances.numpy(), reference_distances[:, 1], atol=1e-6)
    assert sorted(pairs[0].tolist()) == sorted(reference_indices[0].tolist())
    assert [abs(start - planted) <= 5 for start, planted in zip(sorted(pairs[0].tolist()), (60, 250))] == [True, True]


def test_motifs_exclusion_removes_both_starts_hand_derived():
    profile = torch.tensor([5.0, 1.0, 1.1, 4.0, 2.0, 2.5, 1.2, 3.0])
    index = torch.tensor([7, 6, 6, 5, 6, 3, 1, 0])
    distances, pairs = motifs(profile, index, window=4, num_picks=3, exclusion_zone=1)
    # pick 1: i=1 (1.0) with 6 removes 0-2 and 5-7; pick 2: only i=3 or 4 left -> 4 (2.0, nbr 6), which removes 3.
    assert pairs.tolist() == [[1, 6], [4, 6], [-1, -1]]
    assert distances[:2].tolist() == [1.0, 2.0] and torch.isnan(distances[2])


def test_discords_hand_derived_and_batched():
    profile = torch.tensor([1.0, 9.0, 8.5, 2.0, 7.0, float("inf"), 3.0])
    values, indices = discords(profile, window=4, num_picks=3, exclusion_zone=1)
    assert indices.tolist() == [1, 4, 6] and values.tolist() == [9.0, 7.0, 3.0]
    values, indices = discords(torch.stack([profile, profile]), window=4, num_picks=1, exclusion_zone=1)
    assert indices.tolist() == [[1], [1]]


def test_discords_planted_anomaly():
    generator = torch.Generator().manual_seed(3)
    time = torch.arange(400, dtype=torch.float64)
    series = torch.sin(time * 2 * math.pi / 20) + 0.05 * torch.randn(400, generator=generator, dtype=torch.float64)
    series[200:220] = torch.sign(torch.sin(time[200:220] * 2 * math.pi / 7))
    profile, _ = matrix_profile(series, 20)
    values, indices = discords(profile, 20)
    assert 180 <= indices[0].item() <= 220
    assert indices[0].item() == profile.argmax().item() and values[0].item() == profile.max().item()


def test_arc_curve_hand_derived():
    # index [2, 3, 0, 1]: arcs (0,2) crosses 0,1; (1,3) crosses 1,2; (2,0) crosses 0,1; (3,1) crosses 1,2.
    # AC = [2, 4, 2, 0]; IAC = 2 j (4 - j) / 4 = [0, 1.5, 2, 1.5]; CAC (no edge) = [1, 1, 1, 0].
    curve = corrected_arc_curve(torch.tensor([2, 3, 0, 1]), window=1, exclusion_factor=0)
    assert curve.tolist() == [1.0, 1.0, 1.0, 0.0]
    # unmatched entry adds no arc: index [2, -1, 0, -1] -> arcs (0,2), (2,0): AC = [2, 2, 0, 0]
    curve = corrected_arc_curve(torch.tensor([2, -1, 0, -1]), window=1, exclusion_factor=0)
    assert curve.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_edges_set_to_one():
    curve = corrected_arc_curve(-torch.ones(50, dtype=torch.int64), window=4, exclusion_factor=2)
    assert (curve[:8] == 1).all() and (curve[-8:] == 1).all() and (curve[8:-8] == 0).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cac_and_boundaries_match_stumpy(seed):
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(900, dtype=torch.float64)
    series = torch.where(time < 450, torch.sin(time * 2 * math.pi / 40), torch.sin(time * 2 * math.pi / 25) * 1.3)
    series = series + 0.2 * torch.randn(900, generator=generator, dtype=torch.float64)
    window = 30
    _, index = matrix_profile(series, window)
    arc_curve = corrected_arc_curve(index, window)
    positions = np.arange(index.shape[0])
    parabola = 2 * positions * (index.shape[0] - positions) / index.shape[0]
    reference_cac, reference_locs = stumpy.fluss(index.numpy(), window, 2, custom_iac=parabola.astype(np.float64))
    assert np.allclose(arc_curve.numpy(), reference_cac, atol=1e-12)
    boundaries = regime_boundaries(arc_curve, window, num_regimes=3)
    _, locs3 = stumpy.fluss(index.numpy(), window, 3, custom_iac=parabola.astype(np.float64))
    assert boundaries.tolist() == locs3.tolist()
    assert abs(boundaries[0].item() - 450) < 40


def test_boundaries_match_stumpy_default_iac_on_clear_switch():
    time = torch.arange(800, dtype=torch.float64)
    series = torch.where(time < 400, torch.sin(time * 2 * math.pi / 40), torch.sign(torch.sin(time * 2 * math.pi / 23)))
    series = series + 0.05 * torch.randn(800, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    _, index = matrix_profile(series, 40)
    boundary = regime_boundaries(corrected_arc_curve(index, 40), 40)
    _, reference = stumpy.fluss(index.numpy(), 40, 2)
    assert abs(int(boundary[0]) - int(reference[0])) <= 20 and abs(int(boundary[0]) - 400) <= 40


def test_regime_boundaries_hand_derived_and_batched():
    cac = torch.ones(20)
    cac[6] = 0.2
    cac[15] = 0.5
    cac[9] = 0.4
    # exclusion_factor * window = 2: after picking 6, positions 4..7 are cleared; 9 (0.4) is next, clearing 7..10.
    assert regime_boundaries(cac, window=1, num_regimes=4, exclusion_factor=2).tolist() == [6, 9, 15]
    batched = regime_boundaries(torch.stack([cac, cac.flip(0)]), window=1, num_regimes=2, exclusion_factor=2)
    assert batched.tolist() == [[6], [13]]


def test_fluss_batched_rows_independent():
    series = _series(14, 200)
    other = _series(15, 200)
    _, index = matrix_profile(torch.stack([series, other]).unsqueeze(-1), 12)
    curve = corrected_arc_curve(index, 12)
    assert torch.equal(curve[1], corrected_arc_curve(index[1], 12))


def test_fluss_validation():
    with pytest.raises(ValueError):
        corrected_arc_curve(torch.randn(10), 3)
    with pytest.raises(ValueError):
        regime_boundaries(torch.ones(10), 3, num_regimes=1)


def test_regime_boundaries_no_evidence_is_minus_one():
    assert regime_boundaries(torch.ones(20), window=1, num_regimes=3).tolist() == [-1, -1]
    curve = torch.ones(20)
    curve[10] = 0.1
    assert regime_boundaries(curve, window=1, num_regimes=3, exclusion_factor=2).tolist() == [10, -1]


def test_exclusion_zone_with_other_raises():
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(30), 5, exclusion_zone=2, other=torch.randn(30))


def test_small_amplitude_series_is_not_constant():
    series = torch.as_tensor(np.random.default_rng(14).normal(size=30) * 1e-8)
    profile, _ = matrix_profile(series, 5)
    expected = stumpy.stump(series.numpy(), 5)[:, 0].astype(np.float64)
    assert (profile > 0.1).all() and np.allclose(profile.numpy(), expected, atol=1e-6)


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_profile_is_scale_invariant(scale):
    series = _series(20, 120)
    profile, index = matrix_profile(series, 9)
    scaled_profile, scaled_index = matrix_profile(series * scale, 9)
    assert torch.allclose(profile, scaled_profile, atol=1e-6) and torch.equal(index, scaled_index)


def test_near_constant_windows_match_stumpy():
    rng = np.random.default_rng(3)
    series = np.concatenate([1.0 + 1e-6 * rng.normal(size=40), rng.normal(size=60), np.full(20, 2.0)])
    profile, _ = matrix_profile(torch.as_tensor(series), 8)
    expected = stumpy.stump(series, 8)[:, 0].astype(np.float64)
    assert np.allclose(profile.numpy(), expected, atol=1e-6)


def test_every_batch_row_matches_unbatched_call():
    batch = torch.stack([_series(seed, 90) for seed in (30, 31, 32)])
    other = torch.stack([_series(seed, 70) for seed in (40, 41, 42)])
    other[1, 20] = float("nan")
    other[2, 33] = float("inf")
    for kwargs in ({}, {"other": other.unsqueeze(-1)}):
        profile, index = matrix_profile(batch.unsqueeze(-1), 8, **kwargs)
        for row in range(3):
            row_kwargs = {"other": other[row]} if kwargs else {}
            row_profile, row_index = matrix_profile(batch[row], 8, **row_kwargs)
            assert torch.equal(profile[row], row_profile) and torch.equal(index[row], row_index)
    assert not torch.equal(profile[0], profile[1])


def test_ab_join_nonfinite_candidates_match_stumpy():
    series, other = _series(50, 80), _series(51, 60)
    other[25] = float("nan")
    other[40] = float("inf")
    profile, index = matrix_profile(series, 8, other=other)
    expected = stumpy.stump(series.numpy(), 8, other.numpy(), ignore_trivial=False)[:, 0].astype(np.float64)
    assert np.allclose(profile.numpy(), expected, atol=1e-6)
    bad = torch.zeros(53, dtype=torch.bool)
    bad[18:26] = True
    bad[33:41] = True
    assert not bad[index].any()


def test_complex_input_rejected_in_both_positions():
    complex_series = torch.randn(30, dtype=torch.complex128)
    with pytest.raises(ValueError):
        matrix_profile(complex_series, 5)
    with pytest.raises(ValueError):
        matrix_profile(torch.randn(30), 5, other=complex_series)


def test_zero_denominator_convention_matches_stumpy():
    index = -torch.ones(4, dtype=torch.int64)
    curve = corrected_arc_curve(index, window=1, exclusion_factor=0)
    assert curve.tolist() == [0.0, 0.0, 0.0, 0.0]
    expected, _ = stumpy.fluss(index.numpy(), 1, 2, excl_factor=0, custom_iac=np.array([0.0, 1.5, 2.0, 1.5]))
    assert curve.tolist() == expected.tolist()


def test_index_validation():
    with pytest.raises(ValueError):
        corrected_arc_curve(torch.tensor([0.0, 1.0]), 1)
    with pytest.raises(ValueError):
        corrected_arc_curve(torch.tensor([0, 4, 1, 2]), 1)
    with pytest.raises(ValueError):
        corrected_arc_curve(torch.tensor([0, 5, 1, 2]), 1)
    with pytest.raises(ValueError):
        corrected_arc_curve(torch.tensor([0, -2, 1, 2]), 1)
    with pytest.raises(ValueError):
        motifs(torch.rand(4), torch.tensor([1.0, 0.0, 3.0, 2.0]), 2)
    with pytest.raises(ValueError):
        motifs(torch.rand(4), torch.tensor([1, 0, 3]), 2)
    pairs = motifs(torch.rand(4), torch.tensor([1, 0, 3, 2]), 2)[1]
    assert pairs.dtype == torch.int64


def test_empty_time_axis_rejected_and_empty_batch_works():
    with pytest.raises(ValueError):
        corrected_arc_curve(torch.zeros(0, dtype=torch.int64), 1)
    with pytest.raises(ValueError):
        regime_boundaries(torch.zeros(0), 1)
    with pytest.raises(ValueError):
        discords(torch.zeros(0), 1)
    with pytest.raises(ValueError):
        motifs(torch.zeros(0), torch.zeros(0, dtype=torch.int64), 1)
    assert corrected_arc_curve(torch.zeros(0, 5, dtype=torch.int64), 1).shape == (0, 5)
    assert regime_boundaries(torch.zeros(0, 5), 1).shape == (0, 1)
    assert discords(torch.zeros(0, 5), 1)[1].shape == (0, 1)
    values, pairs = motifs(torch.zeros(0, 5), torch.zeros(0, 5, dtype=torch.int64), 1)
    assert pairs.shape == (0, 1, 2) and values.shape == (0, 1)


def test_corrected_arc_curve_vmap_matches_batched():
    index = matrix_profile(torch.stack([_series(60, 100), _series(61, 100)]).unsqueeze(-1), 8)[1]
    mapped = torch.vmap(lambda row: corrected_arc_curve(row, 8, exclusion_factor=1))(index)
    assert torch.equal(mapped, corrected_arc_curve(index, 8, exclusion_factor=1))


def test_ab_join_nonfinite_candidate_cannot_win():
    # Query is a rising ramp; the only finite candidate is a falling ramp (distance 4), farther than
    # sqrt(4) = 2, the distance a zeroed non-finite window would have.
    profile, index = matrix_profile(
        torch.tensor([0.0, 1.0, 2.0, 3.0]), 4, other=torch.tensor([3.0, 2.0, 1.0, 0.0, float("nan")])
    )
    assert profile.item() == pytest.approx(4.0) and index.item() == 0
    profile, index = matrix_profile(
        torch.tensor([0.0, 1.0, 2.0, 3.0]), 4, other=torch.tensor([float("nan"), 3.0, 2.0, 1.0, 0.0])
    )
    assert profile.item() == pytest.approx(4.0) and index.item() == 1


def test_float32_offset_near_constant_matches_stumpy():
    values = torch.tensor(1 + 1e-7 * np.random.default_rng(14).normal(size=30), dtype=torch.float32)
    profile, _ = matrix_profile(values, 5)
    expected = stumpy.stump(values.double().numpy(), 5)[:, 0].astype(np.float64)
    assert np.allclose(profile.double().numpy(), expected, atol=1e-4)


def test_values_near_float32_max_do_not_overflow():
    values = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]) * 1e38
    profile, index = matrix_profile(values, 5, other=values.flip(0))
    assert profile.item() == pytest.approx(2 * math.sqrt(5), rel=1e-4) and index.item() == 0


def test_large_row_does_not_poison_gradient_of_shared_leaf():
    leaf = torch.randn(30, generator=torch.Generator().manual_seed(5), dtype=torch.float32).requires_grad_(True)
    batch = torch.stack([leaf, 1e38 + 1e36 * leaf]).unsqueeze(-1)
    matrix_profile(batch, 5)[0][0].sum().backward()
    assert torch.isfinite(leaf.grad).all()
    reference = leaf.detach().clone().requires_grad_(True)
    matrix_profile(reference, 5)[0].sum().backward()
    assert torch.allclose(leaf.grad, reference.grad, atol=1e-5)


def test_full_range_window_does_not_overflow_in_shift():
    values = torch.tensor([-3e38, 3e38, -3e38, 3e38, 0.0, 1e38, -2e38, 3e38])
    profile, _ = matrix_profile(values, 5, exclusion_zone=0)
    expected, _ = matrix_profile(values.double(), 5, exclusion_zone=0)
    assert torch.isfinite(profile).all() and torch.allclose(profile.double(), expected, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_values_at_dtype_maximum_do_not_collapse(dtype):
    values = torch.tensor([0, 0.25, 0.5, 0.75, 1.0], dtype=dtype) * torch.finfo(dtype).max
    profile, index = matrix_profile(values, 5, other=values.flip(0))
    assert profile.item() == pytest.approx(2 * math.sqrt(5), rel=1e-5) and index.item() == 0
