"""Tests for torchnlts.lyapunov.eckmann.

Dual ground truth:
  1. nolds.lyap_e, run live on the same data and parameters, for
     configurations whose matrix step is 1 (nolds composes the fitted maps
     at consecutive points, which follows the paper only then). Agreement
     is at float64 precision for well-conditioned fits; the one
     ill-conditioned case (an exactly-determined 1D map embedded in 3
     dimensions) uses a looser tolerance.
  2. The paper's algorithm (Sec. 2c: maps composed at i, i + m, i + 2m, ...)
     as an independent straightforward NumPy implementation, for matrix
     steps above 1 and delays above 1; and the known values: Henon map
     (a=1.4, b=0.3) exponents ~ 0.42 and -1.62, logistic map (r=4) ln 2,
     and ``lyapunov_spectrum`` on the map itself.
"""

import math

import numpy
import pytest
import torch

import torchnlts.lyapunov.eckmann as eckmann_module
from tests.conftest import logistic_map_trajectory
from torchnlts.lyapunov.eckmann import eckmann_spectrum
from torchnlts.lyapunov.spectrum import lyapunov_spectrum
from torchnlts.neighbors.knn import knn_search


def _henon_step(state, quadratic_coefficient=1.4, contraction=0.3):
    return torch.stack([1.0 - quadratic_coefficient * state[0] ** 2 + state[1], contraction * state[0]])


def _henon_series(num_points: int, initial_x: float = 0.1) -> torch.Tensor:
    state = torch.tensor([initial_x, 0.0], dtype=torch.float64)
    values = []
    for step in range(num_points + 200):
        state = _henon_step(state)
        if step >= 200:
            values.append(state[0])
    return torch.stack(values)


def _paper_oracle(series, embedding_dim, matrix_dim, num_neighbors, delay=1, theiler_window=0, norm_order=numpy.inf):
    """Straightforward NumPy version: per-point fit, chains at stride m * delay."""
    matrix_step = (embedding_dim - 1) // (matrix_dim - 1)
    shift = matrix_step * delay
    num_points = len(series) - (embedding_dim - 1) * delay - shift
    orbit = numpy.array([series[i : i + embedding_dim * delay : delay] for i in range(num_points)])
    # Neighbor indices come from knn_search (tested separately): under the
    # max-norm, overlapping windows give exact distance ties at the
    # neighborhood boundary, whose order is implementation-defined.
    neighbor_index = knn_search(
        torch.from_numpy(orbit), num_neighbors, theiler_window=theiler_window, norm_order=float(norm_order)
    )[1].numpy()
    tangents = []
    for i in range(num_points):
        neighbors = neighbor_index[i]
        design = numpy.array(
            [[series[j + k * shift] - series[i + k * shift] for k in range(matrix_dim)] for j in neighbors]
        )
        response = numpy.array([series[j + matrix_dim * shift] - series[i + matrix_dim * shift] for j in neighbors])
        coefficients = numpy.linalg.lstsq(design, response, rcond=None)[0]
        tangent = numpy.zeros((matrix_dim, matrix_dim))
        tangent[:-1, 1:] = numpy.eye(matrix_dim - 1)
        tangent[-1] = coefficients
        tangents.append(tangent)
    total = numpy.zeros(matrix_dim)
    for phase in range(min(shift, num_points)):
        orthonormal = numpy.eye(matrix_dim)
        for point in range(phase, num_points, shift):
            orthonormal, upper = numpy.linalg.qr(tangents[point] @ orthonormal)
            total += numpy.log(numpy.abs(numpy.diag(upper)))
    return total / num_points / shift


@pytest.mark.parametrize(
    "embedding_dim, matrix_dim, num_neighbors, theiler_window, tolerance",
    [(4, 4, None, 0, 1e-8), (2, 2, 6, 3, 1e-8), (3, 3, 9, 5, 1e-8), (5, 5, None, 0, 1e-8)],
)
def test_matches_nolds_henon(embedding_dim, matrix_dim, num_neighbors, theiler_window, tolerance):
    import nolds

    series = _henon_series(1200)
    result = eckmann_spectrum(series, embedding_dim, matrix_dim, num_neighbors, theiler_window=theiler_window)
    reference = nolds.lyap_e(
        series.numpy(), emb_dim=embedding_dim, matrix_dim=matrix_dim, min_nb=num_neighbors, min_tsep=theiler_window
    )
    assert result.shape == (matrix_dim,)
    numpy.testing.assert_allclose(result.numpy(), reference, rtol=0, atol=tolerance)


@pytest.mark.parametrize("embedding_dim, matrix_dim, tolerance", [(2, 2, 1e-8), (3, 3, 1e-4)])
def test_matches_nolds_logistic(embedding_dim, matrix_dim, tolerance):
    import nolds

    series = logistic_map_trajectory(1200)
    result = eckmann_spectrum(series, embedding_dim, matrix_dim)
    reference = nolds.lyap_e(series.numpy(), emb_dim=embedding_dim, matrix_dim=matrix_dim)
    numpy.testing.assert_allclose(result.numpy(), reference, rtol=0, atol=tolerance)


@pytest.mark.parametrize(
    "embedding_dim, matrix_dim, delay, theiler_window",
    [(5, 3, 1, 0), (7, 4, 1, 2), (4, 2, 1, 0), (2, 2, 2, 0), (3, 3, 3, 1), (5, 3, 2, 0), (7, 4, 3, 1)],
)
def test_matches_paper_oracle(embedding_dim, matrix_dim, delay, theiler_window):
    series = _henon_series(500)
    num_neighbors = min(2 * matrix_dim, matrix_dim + 4)
    result = eckmann_spectrum(series, embedding_dim, matrix_dim, delay=delay, theiler_window=theiler_window)
    reference = _paper_oracle(series.numpy(), embedding_dim, matrix_dim, num_neighbors, delay, theiler_window)
    numpy.testing.assert_allclose(result.numpy(), reference, rtol=0, atol=1e-8)


@pytest.mark.parametrize("norm_order", [1.0, 2.0, 3.0])
def test_norm_order_changes_neighbors_and_matches_oracle(norm_order):
    series = _henon_series(500)
    result = eckmann_spectrum(series, 4, 4, norm_order=norm_order)
    reference = _paper_oracle(series.numpy(), 4, 4, 8, norm_order=norm_order)
    numpy.testing.assert_allclose(result.numpy(), reference, rtol=0, atol=1e-8)
    assert not torch.allclose(result, eckmann_spectrum(series, 4, 4), atol=1e-6)


def test_henon_known_exponents():
    series = _henon_series(5000)
    spectrum = lyapunov_spectrum(
        _henon_step, torch.tensor([0.1, 0.0], dtype=torch.float64), num_steps=20000, transient_steps=200
    )
    result = eckmann_spectrum(series, 2, 2)
    assert abs(result[0] - 0.42) < 0.03
    assert abs(result[0] - spectrum[0]) < 0.03
    # The local linear fit biases the contracting exponent upward in magnitude.
    assert abs(result[1] - spectrum[1]) < 0.15
    assert abs(result.sum() - math.log(0.3)) < 0.15
    # Compressed embedding (matrix step 2): the true exponents are among the three.
    compressed = eckmann_spectrum(series, 5, 3)
    for true_value in spectrum.tolist():
        assert (compressed - true_value).abs().min() < 0.05


def test_logistic_contains_ln2():
    # A 1D map embedded with matrix_dim=2 also yields a spurious exponent.
    result = eckmann_spectrum(logistic_map_trajectory(3000), 2, 2)
    assert (result - math.log(2)).abs().min() < 0.03


def test_delay_exponents_are_per_sample_step():
    series = _henon_series(5000)
    assert torch.equal(eckmann_spectrum(series[:800], 4, 4, delay=1), eckmann_spectrum(series[:800], 4, 4))
    # A delay-2 embedding follows the tangent map over 2-sample steps; the
    # division by the step recovers the per-sample leading exponent.
    leading = eckmann_spectrum(series, 2, 2, delay=2)[0]
    assert 0.35 < leading < 0.5


def test_batch_matches_individual_rows():
    rows = torch.stack([_henon_series(600, 0.1), _henon_series(600, 0.3)])[..., None]
    for embedding_dim, matrix_dim in [(4, 4), (5, 3)]:
        batched = eckmann_spectrum(rows, embedding_dim, matrix_dim)
        assert batched.shape == (2, matrix_dim)
        for index in range(2):
            torch.testing.assert_close(batched[index], eckmann_spectrum(rows[index, :, 0], embedding_dim, matrix_dim))


def test_input_shapes():
    series = _henon_series(400)
    reference = eckmann_spectrum(series, 4, 4)
    torch.testing.assert_close(eckmann_spectrum(series[:, None], 4, 4), reference)
    torch.testing.assert_close(eckmann_spectrum(series.numpy(), 4, 4), reference)
    with pytest.raises(ValueError, match="scalar"):
        eckmann_spectrum(torch.randn(100, 2), 4, 4)
    with pytest.raises(ValueError):
        eckmann_spectrum(torch.randn(2, 3, 100, 1), 4, 4)


def test_dtypes():
    series = _henon_series(500)
    reference = eckmann_spectrum(series, 4, 4)
    assert eckmann_spectrum(series.float(), 4, 4).dtype == torch.float32
    torch.testing.assert_close(eckmann_spectrum(series.float(), 4, 4).double(), reference, atol=2e-2, rtol=0)
    # Half and bfloat16 are computed in float32 on the rounded values.
    for dtype, atol in [(torch.float16, 5e-3), (torch.bfloat16, 3e-2)]:
        rounded = series.to(dtype)
        result = eckmann_spectrum(rounded, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.double(), eckmann_spectrum(rounded.double(), 4, 4), atol=atol, rtol=0)
    integers = (series * 1000).round().to(torch.int64)
    result = eckmann_spectrum(integers, 4, 4)
    assert result.dtype == torch.get_default_dtype()
    torch.testing.assert_close(result.double(), eckmann_spectrum(integers.double(), 4, 4), atol=1e-3, rtol=0)
    with pytest.raises(ValueError, match="complex"):
        eckmann_spectrum(torch.randn(100, dtype=torch.complex128), 4, 4)


def test_scale_and_shift_invariant():
    series = _henon_series(600)
    torch.testing.assert_close(
        eckmann_spectrum(series * 1e30 + 5e29, 4, 4), eckmann_spectrum(series, 4, 4), atol=1e-6, rtol=1e-6
    )
    assert torch.isfinite(eckmann_spectrum((series * 1e30).float(), 4, 4)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(embedding_dim=4, matrix_dim=1),
        dict(embedding_dim=3, matrix_dim=4),
        dict(embedding_dim=4, matrix_dim=3),
        dict(embedding_dim=4, matrix_dim=4, delay=0),
        dict(embedding_dim=4, matrix_dim=4, delay=1.5),
        dict(embedding_dim=4, matrix_dim=4, num_neighbors=0),
        dict(embedding_dim=4, matrix_dim=4, num_neighbors=3),
        dict(embedding_dim=4, matrix_dim=4, num_neighbors=6.5),
        dict(embedding_dim=4, matrix_dim=4, theiler_window=-1),
        dict(embedding_dim=4, matrix_dim=4, theiler_window=1.5),
        dict(embedding_dim=4, matrix_dim=4, norm_order=-1.0),
        dict(embedding_dim=4.5, matrix_dim=4),
        dict(embedding_dim=4, matrix_dim=2.5),
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        eckmann_spectrum(torch.randn(200), **kwargs)


def test_degenerate_inputs_give_nan():
    assert eckmann_spectrum(torch.randn(5), 4, 4).isnan().all()
    # Window larger than the orbit: no admissible neighbors.
    assert eckmann_spectrum(torch.randn(60), 4, 4, theiler_window=1000).isnan().all()
    assert eckmann_spectrum(torch.randn(12), 4, 4, num_neighbors=20).isnan().all()
    # Too few points for a full-rank neighborhood (placeholder systems).
    assert eckmann_spectrum(_henon_series(20), 2, 2, theiler_window=8).isnan().all()


@pytest.mark.parametrize(
    "series",
    [
        torch.ones(200, dtype=torch.float64),
        torch.arange(200, dtype=torch.float64) % 2,
        torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64).repeat(70),
        torch.arange(200, dtype=torch.float64),
    ],
    ids=["constant", "period2", "period3", "ramp"],
)
def test_rank_deficient_series_give_nan(series):
    assert eckmann_spectrum(series, 3, 3).isnan().all()


def test_nonfinite_rows_give_nan_without_affecting_others():
    rows = torch.stack([_henon_series(500, 0.1), _henon_series(500, 0.3), _henon_series(500, 0.2)])[..., None]
    clean = eckmann_spectrum(rows, 4, 4)
    rows_bad = rows.clone()
    rows_bad[0, 10, 0] = float("nan")
    rows_bad[2, 20, 0] = float("inf")
    result = eckmann_spectrum(rows_bad, 4, 4)
    assert result[0].isnan().all() and result[2].isnan().all()
    torch.testing.assert_close(result[1], clean[1])


def test_gradients_finite_and_isolated_between_rows():
    rows = torch.stack([_henon_series(300, 0.1), _henon_series(300, 0.3)])[..., None].clone()
    rows.requires_grad_(True)
    result = eckmann_spectrum(rows, 2, 2)
    result[0].sum().backward()
    assert torch.isfinite(rows.grad).all()
    assert rows.grad[0].abs().sum() > 0
    assert rows.grad[1].abs().sum() == 0


@pytest.mark.parametrize("bad_row", ["constant", "period2", "ramp"])
def test_degenerate_rows_have_zero_gradient_and_do_not_leak(bad_row):
    clean = _henon_series(80)
    steps = torch.arange(80, dtype=torch.float64)
    bad = {"constant": torch.ones_like(clean), "period2": steps % 2, "ramp": steps}[bad_row]
    # A shared scale parameter multiplies both rows, as a trainable model would.
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    rows = torch.stack([clean, bad])[..., None].clone().requires_grad_(True)
    result = eckmann_spectrum(scale * rows, 2, 2)
    assert result[0].isfinite().all() and result[1].isnan().all()
    result.nan_to_num().sum().backward()
    assert torch.isfinite(scale.grad) and torch.isfinite(rows.grad).all()
    assert rows.grad[1].abs().sum() == 0
    # The shared parameter's gradient equals that of the clean row alone.
    scale_alone = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    eckmann_spectrum(scale_alone * clean[None, :, None], 2, 2).sum().backward()
    torch.testing.assert_close(scale.grad, scale_alone.grad)


def test_singular_neighborhood_gradient_is_zero_and_finite():
    # Window larger than the orbit gives no admissible neighbors; the
    # placeholder tangent maps must not make backward NaN.
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    result = eckmann_spectrum(scale * _henon_series(20), 2, 2, theiler_window=8)
    assert result.isnan().all()
    result.nan_to_num().sum().backward()
    assert scale.grad == 0


def test_short_series_gradient_is_finite_zero():
    short = torch.randn(5, requires_grad=True)
    eckmann_spectrum(short, 4, 4).nan_to_num().sum().backward()
    assert short.grad.abs().sum() == 0


def test_zero_leading_coefficient_row_is_nan_without_leaking(monkeypatch):
    # The companion matrix has determinant +-coefficients[..., 0]; an exactly
    # zero entry makes it singular. Force that in row 1 only.
    original = eckmann_module._least_squares

    def zero_leading(design, response):
        solution = original(design, response).clone()
        solution[1, ..., 0] = 0.0
        return solution

    monkeypatch.setattr(eckmann_module, "_least_squares", zero_leading)
    rows = torch.stack([_henon_series(300, 0.1), _henon_series(300, 0.3)])[..., None]
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    result = eckmann_spectrum(scale * rows, 2, 2)
    assert result[0].isfinite().all() and result[1].isnan().all()
    result.nan_to_num().sum().backward()
    assert torch.isfinite(scale.grad)


def test_zero_growth_factor_row_is_nan(monkeypatch):
    original = torch.linalg.qr

    def zero_growth(matrix):
        orthonormal, upper = original(matrix)
        if matrix.shape[-2] != matrix.shape[-1]:  # the tangent fit, not the chain
            return orthonormal, upper
        mask = torch.ones_like(upper)
        mask[1, ..., 0, 0] = 0.0
        return orthonormal, upper * mask

    monkeypatch.setattr(torch.linalg, "qr", zero_growth)
    rows = torch.stack([_henon_series(300, 0.1), _henon_series(300, 0.3)])[..., None]
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    result = eckmann_spectrum(scale * rows, 2, 2)
    assert result[0].isfinite().all() and result[1].isnan().all()
    assert torch.isfinite(torch.autograd.grad(result[0].sum(), scale)[0])


def test_mixed_valid_and_rejected_points_keep_shared_gradient():
    # Regression: a row with some rejected fits used a singular fallback tangent
    # ([[0, 1], [0, 1]]) whose QR backward put NaN into a shared parameter.
    clean = _henon_series(30)
    damaged = clean.clone()
    damaged[:10] = 1.0
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    result = eckmann_spectrum(scale * torch.stack([clean, damaged])[..., None], 3, 2, delay=2)
    assert result[0].isfinite().all() and result[1].isnan().all()
    gradient = torch.autograd.grad(result.nan_to_num().sum(), scale)[0]
    scale_alone = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.grad(eckmann_spectrum(scale_alone * clean, 3, 2, delay=2).sum(), scale_alone)[0]
    assert torch.isfinite(gradient) and torch.allclose(gradient, expected)


def test_ill_conditioned_fit_gradient_matches_finite_difference():
    # Regression: lstsq's backward was ~1000x off for a design with condition
    # number ~4e9 (ramp plus 1e-9 noise); the QR solve matches a central difference.
    noise = torch.randn(40, generator=torch.Generator().manual_seed(731), dtype=torch.float64)
    series = (torch.arange(40, dtype=torch.float64) + 1e-9 * noise).requires_grad_()
    gradient = torch.autograd.grad(eckmann_spectrum(series, 2, 2).sum(), series)[0]
    step = 1e-11
    plus, minus = series.detach().clone(), series.detach().clone()
    plus[0] += step
    minus[0] -= step
    difference = (eckmann_spectrum(plus, 2, 2).sum() - eckmann_spectrum(minus, 2, 2).sum()) / (2 * step)
    assert torch.isclose(gradient[0], difference, rtol=1e-3)


def test_gradcheck_well_conditioned():
    series = _henon_series(60).requires_grad_()
    assert torch.autograd.gradcheck(lambda values: eckmann_spectrum(values, 3, 3), (series,))


def test_tangent_singular_to_rounding_keeps_shared_gradient():
    # Regression: this row fits a leading coefficient of ~1.7e-16, not exactly
    # 0; the chain QR then had an exactly zero diagonal and its backward put NaN
    # into the shared parameter even though only the clean row was used.
    damaged = torch.tensor(
        [1, -1, 0, 0, -1, 1, 0, 0, 0, 2, 0, 0, 0, -1, -1, 2, 1, -1, -2, 2, -1, 1, 1, 2, 1, 0, 1, -1, 1, 0],
        dtype=torch.float64,
    )
    clean = torch.randn(30, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    parameter = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    result = eckmann_spectrum(parameter * torch.stack([clean, damaged])[..., None], 3, 2, delay=2)
    assert result[0].isfinite().all() and result[1].isnan().all()
    gradient = torch.autograd.grad(result[0].sum(), parameter)[0]
    parameter_alone = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.grad(eckmann_spectrum(parameter_alone * clean, 3, 2, delay=2).sum(), parameter_alone)[0]
    assert torch.isfinite(gradient) and torch.allclose(gradient, expected, atol=1e-12)


def test_half_precision_is_computed_in_float32():
    series = _henon_series(300).to(torch.float16)
    assert torch.equal(eckmann_spectrum(series, 2, 2), eckmann_spectrum(series.float(), 2, 2).half())


def test_chain_product_singular_to_rounding_keeps_shared_gradient():
    # Regression: in float32 every tangent map of row 1 passes its own checks
    # (fit condition ~1.6e4), but a chain product is singular to rounding (R
    # diagonal [19720, 0]); QR's backward then put NaN into the shared
    # parameter although only row 0 was used.
    generator = torch.Generator().manual_seed(981)
    for trial in range(4):
        length = (2 + trial % 3) * (1 + trial // 3 % 2) * (1 + trial // 6 % 2) + 20
        rows = torch.randn(129, length, generator=generator)
        rows[1:] *= 10 ** (torch.rand(128, length, generator=generator) * (2 + trial % 11))
    parameter = torch.tensor(0.8, requires_grad=True)
    result = eckmann_spectrum(parameter * rows[[0, 25], :, None], 2, 2, num_neighbors=2, delay=2)
    assert result[0].isfinite().all() and result[1].isnan().all()
    gradient = torch.autograd.grad(result[0].sum(), parameter)[0]
    parameter_alone = torch.tensor(0.8, requires_grad=True)
    expected = torch.autograd.grad(
        eckmann_spectrum(parameter_alone * rows[0], 2, 2, num_neighbors=2, delay=2).sum(), parameter_alone
    )[0]
    assert torch.isfinite(gradient) and torch.allclose(gradient, expected, atol=1e-6)


def test_qr_chain_rejects_zero_and_subnormal_growth():
    smallest_normal = torch.finfo(torch.float32).tiny
    for growth, rejected in ((0.0, True), (smallest_normal / 2, True), (smallest_normal, False)):
        tangent = torch.eye(2).reshape(1, 1, 2, 2).clone()
        tangent[0, 0, 0, 0] = growth
        assert eckmann_module._qr_chain(tangent, 1)[1].item() == rejected
