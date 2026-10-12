"""Tests for torchnlts.lyapunov.spectrum and torchnlts.lyapunov.jacobian.

Ground truth:

- Benettin et al. (1980): for a fixed linear map ``x -> A x`` the spectrum is ``log(abs(eigenvalues(A)))``. A diagonal
  ``A`` keeps the deviation matrix diagonal (up to sign) at every QR step, so the result is exact from the first step.
- ChaosTools.jl's ``lyapunovspectrum``, run through ``juliaenv`` on ``PredefinedDynamicalSystems.henon()`` (a = 1.4,
  b = 0.3, u0 = (0, 0)) for the full spectrum, a partial spectrum and a ``Ttr`` transient, and on a
  ``DeterministicIteratedMap`` with an unsorted diagonal, where both implementations return a non-descending spectrum.
"""

import math

import numpy
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.lyapunov.jacobian import step_jacobian
from torchnlts.lyapunov.spectrum import lyapunov_spectrum


def _julia_floats(julia_code: str) -> list[float]:
    return [float(value) for value in run_julia(julia_code).split()]


def henon_step(x, a=1.4, b=0.3):
    # Same formula as PredefinedDynamicalSystems.jl's henon_rule (0-indexed here).
    return torch.stack([1.0 - a * x[0] ** 2 + x[1], b * x[0]])


def diagonal_step(x, values):
    return values * x


def test_step_jacobian_unbatched_matches_hand_derivative():
    x = torch.tensor([2.0, 3.0], dtype=torch.float64)
    jacobian = step_jacobian(henon_step, x)
    # Jacobian rows: (-2.8*x0, 1) and (0.3, 0).
    expected = torch.tensor([[-2.8 * 2.0, 1.0], [0.3, 0.0]], dtype=torch.float64)
    assert torch.allclose(jacobian, expected)


def test_step_jacobian_batched_matches_independent_loop():
    torch.manual_seed(0)
    x = torch.randn(4, 2, dtype=torch.float64)
    batched = step_jacobian(henon_step, x)
    for i in range(4):
        assert torch.allclose(batched[i], step_jacobian(henon_step, x[i]))


def test_step_jacobian_rejects_bad_ndim():
    with pytest.raises(ValueError):
        step_jacobian(henon_step, torch.zeros(2, 2, 2))


def test_lyapunov_spectrum_diagonal_map_is_exact_from_first_step():
    # Fixed diagonal map: spectrum is exactly log(abs(diagonal)), exact already at num_steps=1 (see module docstring).
    values = torch.tensor([2.0, 0.5], dtype=torch.float64)

    def step(x):
        return diagonal_step(x, values)

    state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = lyapunov_spectrum(step, state, num_steps=1)
    expected = torch.log(values)
    assert torch.allclose(result, expected)

    result_many = lyapunov_spectrum(step, state, num_steps=50)
    assert torch.allclose(result_many, expected)


def test_lyapunov_spectrum_diagonal_map_negative_entry_and_nonunit_dt():
    # Negative diagonal entry exercises |diag(R)|; dt rescales time, so log(abs(values)) / dt is exact.
    values = torch.tensor([-2.0, 0.5], dtype=torch.float64)

    def step(x):
        return diagonal_step(x, values)

    state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = lyapunov_spectrum(step, state, num_steps=10, dt=0.25)
    expected = torch.log(values.abs()) / 0.25
    assert torch.allclose(result, expected)


_HENON_SETUP = "using ChaosTools, DynamicalSystems\nds = PredefinedDynamicalSystems.henon()\n"


def test_lyapunov_spectrum_matches_chaostools_jl_henon_full_spectrum():
    julia_lambdas = _julia_floats(_HENON_SETUP + 'λ = lyapunovspectrum(ds, 10000)\nprintln(λ[1], " ", λ[2])')
    state0 = torch.zeros(2, dtype=torch.float64)
    result = lyapunov_spectrum(henon_step, state0, num_steps=10000)
    expected = torch.tensor(julia_lambdas, dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-10, rtol=0)
    assert result[0] > result[1]


def test_lyapunov_spectrum_matches_chaostools_jl_henon_partial_spectrum():
    (julia_lambda,) = _julia_floats(_HENON_SETUP + "λ = lyapunovspectrum(ds, 10000, 1)\nprintln(λ[1])")
    state0 = torch.zeros(2, dtype=torch.float64)
    result = lyapunov_spectrum(henon_step, state0, num_steps=10000, num_exponents=1)
    assert result.shape == (1,)
    assert torch.isclose(result[0], torch.tensor(julia_lambda, dtype=torch.float64), atol=1e-10, rtol=0)


def test_lyapunov_spectrum_matches_chaostools_jl_henon_with_transient():
    julia_lambdas = _julia_floats(_HENON_SETUP + 'λ = lyapunovspectrum(ds, 5000; Ttr=100)\nprintln(λ[1], " ", λ[2])')
    state0 = torch.zeros(2, dtype=torch.float64)
    result = lyapunov_spectrum(henon_step, state0, num_steps=5000, transient_steps=100)
    expected = torch.tensor(julia_lambdas, dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-10, rtol=0)


def test_lyapunov_spectrum_diagonal_map_can_be_non_descending():
    # Non-descending diagonal: the coordinate axes are invariant directions, so QR never reorders;
    # ChaosTools.jl returns the same spectrum (see module docstring).
    julia_lambdas = _julia_floats(
        "using ChaosTools, DynamicalSystems\n"
        "diag_rule(x, p, n) = SVector{2}(p[1]*x[1], p[2]*x[2])\n"
        "ds = DeterministicIteratedMap(diag_rule, [1.0, 1.0], [0.5, 2.0])\n"
        'λ = lyapunovspectrum(ds, 50)\nprintln(λ[1], " ", λ[2])'
    )

    def step(x):
        return diagonal_step(x, torch.tensor([0.5, 2.0], dtype=torch.float64))

    state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = lyapunov_spectrum(step, state, num_steps=50)
    expected = torch.tensor(julia_lambdas, dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-8, rtol=0)
    assert result[0] < result[1]


def test_lyapunov_spectrum_matches_analytic_logistic_map_exponent():
    # r=4 logistic map: exponent exactly ln(2). Hand-rolled because the estimator takes a differentiable step callable.
    def logistic_step(x):
        return 4.0 * x * (1.0 - x)

    state = torch.tensor([0.4], dtype=torch.float64)
    result = lyapunov_spectrum(logistic_step, state, num_steps=5000, transient_steps=100)
    assert abs(result.item() - math.log(2)) < 1e-2


def test_lyapunov_spectrum_batched_matches_independent_loop():
    torch.manual_seed(1)
    state0 = torch.stack([torch.zeros(2, dtype=torch.float64), torch.tensor([0.05, -0.05], dtype=torch.float64)])
    batched = lyapunov_spectrum(henon_step, state0, num_steps=200)
    assert batched.shape == (2, 2)
    for i in range(2):
        expected = lyapunov_spectrum(henon_step, state0[i], num_steps=200)
        assert torch.allclose(batched[i], expected)


def test_lyapunov_spectrum_dtype_preserved_float32():
    state0 = torch.zeros(2, dtype=torch.float32)
    result = lyapunov_spectrum(henon_step, state0, num_steps=50)
    assert result.dtype == torch.float32


def test_lyapunov_spectrum_rejects_bad_state_ndim():
    with pytest.raises(ValueError):
        lyapunov_spectrum(henon_step, torch.zeros(2, 2, 2), num_steps=1)


def test_lyapunov_spectrum_rejects_non_positive_num_steps():
    with pytest.raises(ValueError):
        lyapunov_spectrum(henon_step, torch.zeros(2, dtype=torch.float64), num_steps=0)


def test_lyapunov_spectrum_rejects_num_exponents_out_of_range():
    state = torch.zeros(2, dtype=torch.float64)
    for bad in (0, 3):
        with pytest.raises(ValueError):
            lyapunov_spectrum(henon_step, state, num_steps=1, num_exponents=bad)


def test_lyapunov_spectrum_rejects_non_positive_dt():
    state = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        lyapunov_spectrum(henon_step, state, num_steps=1, dt=0.0)


def test_lyapunov_spectrum_rejects_negative_transient_steps():
    state = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        lyapunov_spectrum(henon_step, state, num_steps=1, transient_steps=-1)


def test_lyapunov_spectrum_gradcheck_w_r_t_initial_state():
    # 3 steps: exercises the Jacobian/QR loop under finite-difference perturbations without chaotic amplification.
    state = torch.tensor([0.1, 0.1], dtype=torch.float64, requires_grad=True)

    def f(x):
        return lyapunov_spectrum(henon_step, x, num_steps=3)

    assert torch.autograd.gradcheck(f, (state,))


def test_lyapunov_spectrum_gradcheck_w_r_t_model_parameter():
    # step_fn's own parameters require grad (a trained model); a fixed diagonal map keeps the check safe.
    values = torch.tensor([1.5, 0.5], dtype=torch.float64, requires_grad=True)
    state = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def f(values):
        return lyapunov_spectrum(lambda x: diagonal_step(x, values), state, num_steps=4)

    assert torch.autograd.gradcheck(f, (values,))


@pytest.mark.parametrize("name", ["num_steps", "num_exponents", "transient_steps"])
def test_lyapunov_spectrum_integer_arguments_raise_value_error(name):
    state = torch.zeros(2, dtype=torch.float64)
    arguments = {"num_steps": 2, "num_exponents": 1, "transient_steps": 1}
    for bad in (2.0, 1.5, "2", None):
        if name == "num_exponents" and bad is None:
            continue
        with pytest.raises(ValueError, match=name):
            lyapunov_spectrum(henon_step, state, **{**arguments, name: bad})


def test_lyapunov_spectrum_accepts_numpy_integers_and_floats():
    state = torch.tensor([0.1, 0.1], dtype=torch.float64)
    result = lyapunov_spectrum(
        henon_step,
        state,
        num_steps=numpy.int64(20),
        num_exponents=numpy.int32(1),
        dt=numpy.float32(0.5),
        transient_steps=numpy.int16(3),
    )
    expected = lyapunov_spectrum(henon_step, state, num_steps=20, num_exponents=1, dt=0.5, transient_steps=3)
    assert torch.equal(result, expected)


def test_lyapunov_spectrum_dt_must_be_a_positive_real_number():
    state = torch.zeros(2, dtype=torch.float64)
    for bad in ("a", None, 1j, 0.0, -1.0, float("nan"), [1.0, 2.0]):
        with pytest.raises(ValueError, match="dt"):
            lyapunov_spectrum(henon_step, state, num_steps=1, dt=bad)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float16, torch.bfloat16, torch.complex64])
def test_lyapunov_spectrum_rejects_states_that_are_not_float32_or_float64(dtype):
    for state in (torch.ones(2).to(dtype), torch.ones(3, 2).to(dtype)):
        with pytest.raises(ValueError, match="float32 or float64"):
            lyapunov_spectrum(henon_step, state, num_steps=2)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_step_jacobian_rejects_non_floating_states(dtype):
    for state in (torch.ones(2).to(dtype), torch.ones(3, 2).to(dtype)):
        with pytest.raises(ValueError, match="real floating-point"):
            step_jacobian(henon_step, state)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_step_jacobian_supports_half_precision_states(dtype):
    state = torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=dtype)
    jacobian = step_jacobian(henon_step, state)
    assert jacobian.dtype == dtype
    torch.testing.assert_close(jacobian.float(), step_jacobian(henon_step, state.float()), rtol=1e-2, atol=1e-2)


def test_step_jacobian_nonfinite_rows_are_nan_and_keep_gradients_finite():
    nan, inf = float("nan"), float("inf")
    states = torch.tensor([[0.1, 0.2], [nan, 0.2], [0.3, 0.1], [inf, 0.0]], dtype=torch.float64)
    parameter = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)

    def step(state):
        return torch.stack([1.0 - parameter * state[0] ** 2 + state[1], 0.3 * state[0]])

    jacobian = step_jacobian(step, states)
    assert torch.isnan(jacobian).flatten(1).all(dim=-1).tolist() == [False, True, False, True]
    valid = [0, 2]
    torch.testing.assert_close(jacobian[valid], step_jacobian(step, states[valid]))
    (gradient,) = torch.autograd.grad(jacobian[valid].sum(), parameter)
    (expected,) = torch.autograd.grad(step_jacobian(step, states[valid]).sum(), parameter)
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


def test_lyapunov_spectrum_nonfinite_rows_are_nan_and_keep_gradients_finite():
    nan, inf = float("nan"), float("inf")
    states = torch.tensor([[0.1, 0.2], [nan, 0.2], [0.3, 0.1], [inf, 0.0]], dtype=torch.float64)
    parameter = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)

    def step(state):
        return torch.stack([1.0 - parameter * state[0] ** 2 + state[1], 0.3 * state[0]])

    spectrum = lyapunov_spectrum(step, states, num_steps=10)
    assert torch.isnan(spectrum).all(dim=-1).tolist() == [False, True, False, True]
    valid = [0, 2]
    expected_spectrum = lyapunov_spectrum(step, states[valid], num_steps=10)
    torch.testing.assert_close(spectrum[valid], expected_spectrum)
    (gradient,) = torch.autograd.grad(spectrum[valid].sum(), parameter)
    (expected,) = torch.autograd.grad(expected_spectrum.sum(), parameter)
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


def _quadratic_step(parameter):
    # The Jacobian 2 * parameter * x is singular at the zero state that stands in for a non-finite one.
    return lambda state: parameter * state.square()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_unbatched_nonfinite_state_gives_nan_like_a_batch_row(bad):
    state = torch.tensor([0.5, bad], dtype=torch.float64)

    def step(state):
        return 2 * state

    assert step_jacobian(step, state).shape == (2, 2) and torch.isnan(step_jacobian(step, state)).all()
    assert torch.isnan(step_jacobian(step, state[None])).all()
    spectrum = lyapunov_spectrum(step, state, num_steps=3)
    assert spectrum.shape == (2,) and torch.isnan(spectrum).all()
    assert torch.isnan(lyapunov_spectrum(step, state[None], num_steps=3)).all()
    finite = torch.tensor([0.5, 0.25], dtype=torch.float64)
    torch.testing.assert_close(step_jacobian(step, finite), 2 * torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(
        lyapunov_spectrum(step, finite, num_steps=3), torch.full((2,), math.log(2), dtype=finite.dtype)
    )


@pytest.mark.parametrize("batched", [False, True], ids=["unbatched_inside", "batched_inside"])
def test_step_jacobian_under_vmap_nonfinite_state_keeps_shared_gradients_finite(batched):
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    nan = float("nan")
    states = torch.tensor([[[1.0], [2.0]], [[nan], [nan]]], dtype=torch.float64)  # [N, B, D=1]
    if not batched:
        states = states[:, 0]
    jacobian = torch.vmap(lambda state: step_jacobian(_quadratic_step(parameter), state))(states)
    assert torch.isnan(jacobian[1]).all() and torch.isfinite(jacobian[0]).all()
    (gradient,) = torch.autograd.grad(jacobian[0].sum(), parameter)
    assert gradient == (6.0 if batched else 2.0)


@pytest.mark.parametrize("function", ["step_jacobian", "lyapunov_spectrum"])
@pytest.mark.parametrize("states", [[[float("nan")]], [[float("nan")], [float("inf")]]], ids=["one_row", "two_rows"])
def test_all_nonfinite_batch_keeps_shared_gradients_finite(function, states):
    parameter = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    states = torch.tensor(states, dtype=torch.float64)
    step = _quadratic_step(parameter)
    result = step_jacobian(step, states) if function == "step_jacobian" else lyapunov_spectrum(step, states, 3)
    assert torch.isnan(result).all()
    (gradient,) = torch.autograd.grad(parameter + torch.nan_to_num(result).sum(), parameter)
    assert gradient == 1


def test_lyapunov_spectrum_masked_rows_do_not_follow_the_orbit_from_zero():
    # From zero, s -> p s^2 + 2 overflows float32 within 9 steps; a masked row has to be reset instead of evolving.
    parameter = torch.tensor(1.0, requires_grad=True)
    initial_state = torch.tensor([[float("nan")]])
    spectrum = lyapunov_spectrum(lambda state: parameter * state.square() + 2, initial_state, num_steps=12)
    assert torch.isnan(spectrum).all()
    (gradient,) = torch.autograd.grad(parameter + torch.nan_to_num(spectrum).sum(), parameter)
    assert gradient == 1


def test_lyapunov_spectrum_under_vmap_matches_the_batch_and_keeps_gradients_finite():
    nan = float("nan")
    states = torch.tensor([[0.1, 0.2], [nan, 0.2], [0.3, 0.1]], dtype=torch.float64)
    parameter = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)

    def step(state):
        return torch.stack([1.0 - parameter * state[0] ** 2 + state[1], 0.3 * state[0]])

    mapped = torch.vmap(lambda state: lyapunov_spectrum(step, state, num_steps=10))(states)
    batched = lyapunov_spectrum(step, states, num_steps=10)
    torch.testing.assert_close(mapped, batched, equal_nan=True)
    assert torch.isnan(mapped[1]).all()
    (gradient,) = torch.autograd.grad(mapped[[0, 2], 0].sum(), parameter)
    (expected,) = torch.autograd.grad(batched[[0, 2], 0].sum(), parameter)
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_batches_give_empty_results(dtype):
    states = torch.empty(0, 2, dtype=dtype)

    def step(state):
        return 2 * state

    jacobian = step_jacobian(step, states)
    assert jacobian.shape == (0, 2, 2) and jacobian.dtype == dtype
    spectrum = lyapunov_spectrum(step, states, num_steps=3)
    assert spectrum.shape == (0, 2) and spectrum.dtype == dtype
    assert lyapunov_spectrum(step, states, num_steps=3, num_exponents=1).shape == (0, 1)
