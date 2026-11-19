import numpy as np
import pytest
import torch

import torchnlts.dimension as dimension
import torchnlts.entropy as entropy
import torchnlts.fluctuations as fluctuations
import torchnlts.lyapunov as lyapunov
import torchnlts.neighbors as neighbors
import torchnlts.recurrence as recurrence
from torchnlts.convert import to_numpy

GENERATOR = torch.Generator().manual_seed(0)
SWEEP = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
VALUES = torch.tensor([0.5, 1.1, 2.3, 4.2], dtype=torch.float64)
PROBABILITY = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
DISTANCE = torch.rand(6, 6, generator=GENERATOR, dtype=torch.float64)
BOOL = torch.rand(12, 12, generator=GENERATOR) > 0.6
STATE = torch.tensor([0.3, 0.2], dtype=torch.float64)


def _step(state):
    return torch.stack([state[1], 0.5 * state[0] + 0.9 * state[1] ** 2 * 0.1])


CASES = {
    "renyi_entropy": (entropy.renyi_entropy, (PROBABILITY, 2.0)),
    "tsallis_entropy": (entropy.tsallis_entropy, (PROBABILITY, 2.0)),
    "correlation_dimension": (dimension.correlation_dimension, (SWEEP, VALUES)),
    "higuchi_dimension": (dimension.higuchi_dimension, (SWEEP, VALUES)),
    "generalized_dimension": (dimension.generalized_dimension, (SWEEP, VALUES)),
    "variogram_dimension": (
        lambda lag, variogram: dimension.variogram_dimension(lag, variogram, power=1.0),
        (SWEEP, VALUES),
    ),
    "dfa_exponent": (fluctuations.dfa_exponent, (SWEEP, VALUES)),
    "rosenstein_lyapunov": (lyapunov.rosenstein_lyapunov, (SWEEP, VALUES)),
    "trajectory_kernel": (neighbors.trajectory_kernel, (DISTANCE, 0.5)),
    "trajectory_kernel_array_temperature": (
        neighbors.trajectory_kernel,
        (DISTANCE, torch.tensor(0.5, dtype=torch.float64)),
    ),
    "apply_theiler_window": (neighbors.apply_theiler_window, (DISTANCE, 1)),
    "lyapunov_spectrum": (lyapunov.lyapunov_spectrum, (_step, STATE, 20)),
    "step_jacobian": (lyapunov.step_jacobian, (_step, STATE)),
    "recurrence_rate": (recurrence.recurrence_rate, (BOOL,)),
    "determinism": (recurrence.determinism, (BOOL,)),
    "dl_average": (recurrence.dl_average, (BOOL,)),
    "dl_max": (recurrence.dl_max, (BOOL,)),
    "dl_entropy": (recurrence.dl_entropy, (BOOL,)),
    "laminarity": (recurrence.laminarity, (BOOL,)),
    "trapping_time": (recurrence.trapping_time, (BOOL,)),
    "vl_max": (recurrence.vl_max, (BOOL,)),
    "vl_entropy": (recurrence.vl_entropy, (BOOL,)),
    "rt_entropy": (recurrence.rt_entropy, (BOOL,)),
    "ratio": (recurrence.ratio, (BOOL,)),
    "divergence": (recurrence.divergence, (BOOL,)),
    "windowed_rqa": (recurrence.windowed_rqa, (BOOL, recurrence.determinism, 6)),
    "integer_probability_renyi": (entropy.renyi_entropy, (torch.tensor([1, 0, 0]), 2.0)),
    "integer_probability_tsallis": (entropy.tsallis_entropy, (torch.tensor([1, 0, 0]), 2.0)),
}


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return to_numpy(value)
    return value


@pytest.mark.parametrize("name", CASES)
def test_numpy_input_matches_torch_input(name):
    function, arguments = CASES[name]
    expected = function(*arguments)
    actual = function(*(_to_numpy(argument) for argument in arguments))
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("function", [entropy.renyi_entropy, entropy.tsallis_entropy])
def test_list_probability(function):
    torch.testing.assert_close(function([0.5, 0.25, 0.25], 2.0), function(PROBABILITY.float(), 2.0))


def test_numpy_bool_matrix_is_boolean():
    assert isinstance(recurrence.determinism(np.asarray(BOOL)), torch.Tensor)
