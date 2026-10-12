"""Shared test helpers.

Not a torchnlts module -- pytest loads this file automatically, but tests
explicitly import the ordinary helper functions defined here.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

requires_nolds = pytest.mark.skipif(sys.version_info < (3, 12), reason="nolds 0.6.3 does not import on Python < 3.12")


def logistic_map_trajectory(num_points: int, x0: float = 0.4) -> torch.Tensor:
    """The r=4 logistic map, whose largest Lyapunov exponent is exactly
    ln(2) -- the standard analytic benchmark for this kind of estimator. Via
    ``dysts.maps.Logistic`` (not hand-rolled: reference dynamical systems
    come from dysts).
    """
    import dysts.maps as maps

    system = maps.Logistic(parameters={"r": 4.0})
    trajectory = system.make_trajectory(num_points, init_cond=np.array([x0]))
    return torch.as_tensor(trajectory, dtype=torch.float64)


def pseudo_random_series(num_points: int, seed: int, scale: float = 1.0, offset: float = 0.0) -> torch.Tensor:
    """Float64 values in ``offset +- scale`` from a linear congruential generator, so they do not depend on the torch
    version's random stream."""
    state, values = seed, []
    for _ in range(num_points):
        state = (1103515245 * state + 12345) % 2**31
        values.append(offset + scale * (2 * state / 2**31 - 1))
    return torch.tensor(values, dtype=torch.float64)


def assert_first_row_gradients_unaffected(function, rows, bandwidth, dtype=torch.float64):
    """Evaluate ``function(series, bandwidth)`` on the batch ``rows`` (lists of equal length) and on its first row
    alone, and assert the gradients of the first result with respect to the series and the shared ``bandwidth`` are
    finite and equal in the two cases, so a degenerate other row adds nothing. Returns the batch result."""

    def first_row_gradients(batch):
        series = torch.tensor(batch, dtype=dtype).unsqueeze(-1).requires_grad_()
        parameter = torch.tensor(bandwidth, dtype=dtype, requires_grad=True)
        result = function(series, parameter)
        gradients = torch.autograd.grad(result[..., 0].sum(), [series, parameter])
        return result.detach(), *gradients

    result, series_gradient, bandwidth_gradient = first_row_gradients(rows)
    _, single_series_gradient, single_bandwidth_gradient = first_row_gradients(rows[:1])
    assert torch.isfinite(series_gradient).all() and torch.isfinite(bandwidth_gradient)
    assert (series_gradient[1:] == 0).all()
    torch.testing.assert_close(series_gradient[:1], single_series_gradient)
    torch.testing.assert_close(bandwidth_gradient, single_bandwidth_gradient)
    return result


def run_julia(code: str, project: str = "juliaenv", timeout: float = 300) -> str:
    """Run ``code`` with ``julia --project=<project> -e`` and return its stdout.

    ``project`` is a directory relative to the repository root, or an absolute path. Without ``julia`` on PATH the
    calling test is skipped, or failed if the environment variable ``TORCHNLTS_REQUIRE_JULIA`` is ``"1"``. A non-zero
    exit status fails the test with Julia's stderr.
    """
    if shutil.which("julia") is None:
        if os.environ.get("TORCHNLTS_REQUIRE_JULIA") == "1":
            pytest.fail("julia not found on PATH, and TORCHNLTS_REQUIRE_JULIA=1 requires it", pytrace=False)
        pytest.skip("julia not available")
    command = ["julia", f"--project={REPOSITORY_ROOT / project}", "-e", code]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"julia did not finish within {timeout} s", pytrace=False)
    if result.returncode != 0:
        pytest.fail(f"julia exited with status {result.returncode}:\n{result.stderr}", pytrace=False)
    return result.stdout


class LargestOutput(TorchDispatchMode):
    """Records the most elements any single operation produces: ``numel`` overall, ``numel_by_dtype`` per dtype."""

    def __init__(self):
        super().__init__()
        self.numel = 0
        self.numel_by_dtype = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for output in result if isinstance(result, (tuple, list)) else (result,):
            if isinstance(output, torch.Tensor):
                self.numel = max(self.numel, output.numel())
                self.numel_by_dtype[output.dtype] = max(self.numel_by_dtype.get(output.dtype, 0), output.numel())
        return result
