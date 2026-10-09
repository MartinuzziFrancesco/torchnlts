"""Tests for torchnlts.lyapunov.kaplan_yorke.

Ground truth: hand-derived values from the Kaplan-Yorke formula, live
FractalDimensions.jl ``kaplanyorke_dim`` calls (run via subprocess), and the
literature dimensions of the Henon (1.258) and Lorenz (2.06) attractors.
"""

import subprocess

import pytest
import torch

from torchnlts.lyapunov.kaplan_yorke import kaplan_yorke_dimension
from torchnlts.lyapunov.spectrum import lyapunov_spectrum

JULIA_PROJECT = "juliaenv"

SPECTRA = [
    [0.9, 0.0, -14.5],  # Lorenz-like, k = 2
    [0.419, -1.623],  # Henon, k = 1
    [0.5, 0.2, 0.1],  # no negative partial sum -> D
    [-0.1, -0.5],  # first negative -> 0
    [0.5, -0.5, -1.0],  # partial sum exactly 0 at k = 2 -> counts as non-negative
    [-1.0, 0.3, 0.9],  # unsorted
    [0.0, 0.0],  # all zero -> D
    [1.0, -1.0],  # last partial sum exactly 0 -> D
    [0.0, -1.0],  # leading zero is non-negative -> 1
    [2.0],
    [-2.0],
]


def test_hand_values():
    # k = 1: 1 + 0.419 / 1.623
    assert kaplan_yorke_dimension(torch.tensor([0.419, -1.623], dtype=torch.float64)).item() == pytest.approx(
        1 + 0.419 / 1.623
    )
    # k = 2: 2 + (0.9 + 0.0) / 14.5
    assert kaplan_yorke_dimension(torch.tensor([0.9, 0.0, -14.5], dtype=torch.float64)).item() == pytest.approx(
        2 + 0.9 / 14.5
    )


def test_edge_cases():
    assert kaplan_yorke_dimension(torch.tensor([0.5, 0.2, 0.1])).item() == 3.0
    assert kaplan_yorke_dimension(torch.tensor([-0.1, -0.5])).item() == 0.0
    assert kaplan_yorke_dimension(torch.tensor([1.0, -1.0])).item() == 2.0
    # The formula is continuous at zero partial sums except at k = 0: a leading zero counts.
    assert kaplan_yorke_dimension(torch.tensor([0.0, -1.0])).item() == 1.0
    assert kaplan_yorke_dimension(torch.tensor([0.0, 0.0])).item() == 2.0
    # Exact zero partial sum at k = 2 followed by a negative exponent: D = 2.
    assert kaplan_yorke_dimension(torch.tensor([0.5, -0.5, -1.0])).item() == 2.0


def test_unsorted_input_is_sorted():
    assert kaplan_yorke_dimension(torch.tensor([-14.5, 0.9, 0.0])).item() == pytest.approx(2 + 0.9 / 14.5)


def test_batched_matches_rows_and_dtype():
    batch = torch.tensor([[0.419, -1.623], [-0.1, -0.5], [0.5, 0.2]], dtype=torch.float64)
    out = kaplan_yorke_dimension(batch)
    assert out.shape == (3,) and out.dtype == torch.float64
    for row in range(3):
        assert out[row].item() == kaplan_yorke_dimension(batch[row]).item()
    assert kaplan_yorke_dimension(torch.tensor([1, -2])).dtype == torch.get_default_dtype()
    assert kaplan_yorke_dimension(torch.tensor([0.5, -1.0], dtype=torch.float16)).dtype == torch.float16


def test_vmap():
    batch = torch.tensor([[0.419, -1.623], [-0.1, -0.5], [0.5, 0.2]], dtype=torch.float64)
    torch.testing.assert_close(torch.vmap(kaplan_yorke_dimension)(batch), kaplan_yorke_dimension(batch))


def test_nan_row_isolated_with_gradients():
    # Regression: a NaN/inf row must give NaN without NaN gradients in other rows.
    batch = torch.tensor(
        [[0.419, -1.623], [float("nan"), -1.0], [float("inf"), -1.0], [0.9, -0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    out = kaplan_yorke_dimension(batch)
    assert out[1].isnan() and out[2].isnan() and out[0].isfinite() and out[3].isfinite()
    out.nansum().backward()
    assert torch.isfinite(batch.grad[[0, 3]]).all()
    assert (batch.grad[[1, 2]] == 0).all()


def test_gradient_values_and_gradcheck():
    spectrum = torch.tensor([0.9, 0.1, -3.0], dtype=torch.float64, requires_grad=True)
    kaplan_yorke_dimension(spectrum).backward()
    # D = 2 + (l1 + l2) / (-l3)
    # dD/dl1 = dD/dl2 = 1/3, dD/dl3 = (l1 + l2) / l3^2 = 1/9
    torch.testing.assert_close(spectrum.grad, torch.tensor([1 / 3.0, 1 / 3.0, 1 / 9.0], dtype=torch.float64))
    # Away from k switches, including unsorted input and a batch.
    for data in ([0.9, 0.1, -3.0], [-3.0, 0.1, 0.9], [0.419, -1.623, -5.0]):
        point = torch.tensor([data, data], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(kaplan_yorke_dimension, (point,))


def test_validation():
    for bad in (torch.zeros(2, 2, 2), torch.zeros(0), torch.tensor(1.0), torch.zeros(2, dtype=torch.complex64)):
        with pytest.raises(ValueError):
            kaplan_yorke_dimension(bad)


def test_julia_cross_check():
    literal = "[" + ", ".join("[" + ", ".join(repr(value) for value in spectrum) + "]" for spectrum in SPECTRA) + "]"
    code = f"using FractalDimensions; for s in {literal}; println(repr(kaplanyorke_dim(s))); end"
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    expected = [float(value) for value in result.stdout.split()]
    assert len(expected) == len(SPECTRA)
    for spectrum, reference in zip(SPECTRA, expected):
        got = kaplan_yorke_dimension(torch.tensor(spectrum, dtype=torch.float64)).item()
        assert got == pytest.approx(reference, abs=1e-12), spectrum


def test_henon_literature_value():
    def henon_step(state):
        return torch.stack([1.0 - 1.4 * state[0] ** 2 + state[1], 0.3 * state[0]])

    spectrum = lyapunov_spectrum(henon_step, torch.zeros(2, dtype=torch.float64), num_steps=20000, transient_steps=1000)
    assert kaplan_yorke_dimension(spectrum).item() == pytest.approx(1.258, abs=0.005)


def test_lorenz_literature_value():
    sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01

    def flow(state):
        return torch.stack(
            [
                sigma * (state[1] - state[0]),
                state[0] * (rho - state[2]) - state[1],
                state[0] * state[1] - beta * state[2],
            ]
        )

    def rk4(state):
        k1 = flow(state)
        k2 = flow(state + 0.5 * dt * k1)
        k3 = flow(state + 0.5 * dt * k2)
        k4 = flow(state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    spectrum = lyapunov_spectrum(
        rk4, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64), num_steps=20000, dt=dt, transient_steps=2000
    )
    assert kaplan_yorke_dimension(spectrum).item() == pytest.approx(2.06, abs=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_computed_in_float32(dtype):
    # Regression: half input is computed in float32 and cast back.
    torch.manual_seed(4)
    batch = torch.randn(1000, 8).to(dtype)
    torch.testing.assert_close(
        kaplan_yorke_dimension(batch), kaplan_yorke_dimension(batch.float()).to(dtype), rtol=0, atol=0
    )


def test_all_zero_spectrum_gradient_is_finite():
    # Regression: the unused interpolation divisor must not be zero.
    spectrum = torch.zeros(2, requires_grad=True)
    kaplan_yorke_dimension(spectrum).backward()
    assert torch.isfinite(spectrum.grad).all()
