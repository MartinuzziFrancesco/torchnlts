"""Tests for torchnlts.lyapunov.spectrum and torchnlts.lyapunov.jacobian.

Dual ground truth:
  1. Paper: Benettin, Galgani, Giorgilli & Strelcyn (1980) -- for a fixed
     (state-independent) linear map ``x_{n+1} = A x_n``, the Lyapunov
     spectrum is exactly ``log(abs(eigenvalues(A)))`` (Oseledets'
     theorem for an autonomous linear cocycle). A diagonal ``A`` makes
     this exact from the very first step: its eigenvalues sit on the
     diagonal already, so the deviation matrix stays exactly diagonal
     (up to sign) at every QR step, with no asymptotic convergence
     needed.
  2. Julia: ChaosTools.jl's ``lyapunovspectrum``, run live via
     subprocess (same pattern as ``tests/test_kantz.py``'s disagreement
     test) on ``PredefinedDynamicalSystems.henon()`` (the standard
     a=1.4, b=0.3 Hénon map, u0=(0,0)) for the full spectrum, a partial
     spectrum, and with a ``Ttr`` transient, and on a plain
     ``DeterministicIteratedMap`` for the non-descending-diagonal edge
     case below -- not hardcoded literals, so a future ChaosTools.jl
     change that altered any of these values would fail this suite
     rather than go unnoticed.
"""

import math
import subprocess

import torch

from torchnlts.lyapunov.jacobian import step_jacobian
from torchnlts.lyapunov.spectrum import lyapunov_spectrum

JULIA_PROJECT = "juliaenv"


def _run_julia(julia_code: str) -> list[float]:
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return [float(value) for value in result.stdout.split()]


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
    try:
        step_jacobian(henon_step, torch.zeros(2, 2, 2))
        assert False, "expected ValueError"
    except ValueError:
        pass


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
    julia_lambdas = _run_julia(_HENON_SETUP + 'λ = lyapunovspectrum(ds, 10000)\nprintln(λ[1], " ", λ[2])')
    state0 = torch.zeros(2, dtype=torch.float64)
    result = lyapunov_spectrum(henon_step, state0, num_steps=10000)
    expected = torch.tensor(julia_lambdas, dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-10, rtol=0)
    assert result[0] > result[1]


def test_lyapunov_spectrum_matches_chaostools_jl_henon_partial_spectrum():
    (julia_lambda,) = _run_julia(_HENON_SETUP + "λ = lyapunovspectrum(ds, 10000, 1)\nprintln(λ[1])")
    state0 = torch.zeros(2, dtype=torch.float64)
    result = lyapunov_spectrum(henon_step, state0, num_steps=10000, num_exponents=1)
    assert result.shape == (1,)
    assert torch.isclose(result[0], torch.tensor(julia_lambda, dtype=torch.float64), atol=1e-10, rtol=0)


def test_lyapunov_spectrum_matches_chaostools_jl_henon_with_transient():
    julia_lambdas = _run_julia(_HENON_SETUP + 'λ = lyapunovspectrum(ds, 5000; Ttr=100)\nprintln(λ[1], " ", λ[2])')
    state0 = torch.zeros(2, dtype=torch.float64)
    result = lyapunov_spectrum(henon_step, state0, num_steps=5000, transient_steps=100)
    expected = torch.tensor(julia_lambdas, dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-10, rtol=0)


def test_lyapunov_spectrum_diagonal_map_can_be_non_descending():
    # Non-descending diagonal: the coordinate axes are invariant directions, so QR never reorders;
    # ChaosTools.jl returns the same spectrum (see module docstring).
    julia_lambdas = _run_julia(
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
    try:
        lyapunov_spectrum(henon_step, torch.zeros(2, 2, 2), num_steps=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lyapunov_spectrum_rejects_non_positive_num_steps():
    try:
        lyapunov_spectrum(henon_step, torch.zeros(2, dtype=torch.float64), num_steps=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lyapunov_spectrum_rejects_num_exponents_out_of_range():
    state = torch.zeros(2, dtype=torch.float64)
    for bad in (0, 3):
        try:
            lyapunov_spectrum(henon_step, state, num_steps=1, num_exponents=bad)
            assert False, f"expected ValueError for num_exponents={bad}"
        except ValueError:
            pass


def test_lyapunov_spectrum_rejects_non_positive_dt():
    state = torch.zeros(2, dtype=torch.float64)
    try:
        lyapunov_spectrum(henon_step, state, num_steps=1, dt=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lyapunov_spectrum_rejects_negative_transient_steps():
    state = torch.zeros(2, dtype=torch.float64)
    try:
        lyapunov_spectrum(henon_step, state, num_steps=1, transient_steps=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


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
