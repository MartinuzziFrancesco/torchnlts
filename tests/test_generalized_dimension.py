"""Tests for torchnlts.dimension.generalized.

Dual ground truth:
  1. Paper: Russell, Hanson & Ott (1980) -- checked via a hand-derived
     box-occupation example (independently re-derived by hand, not
     copied from Julia's output, then cross-checked against it).
  2. Julia: ComplexityMeasures.jl's ``probabilities(ValueHistogram(eps), X)``
     and ``entropy(Renyi(...), ValueHistogram(eps), X)``.
"""

import subprocess

import torch

from torchnlts.dimension.generalized import box_counting_entropy, box_occupation_probabilities, generalized_dimension

JULIA_PROJECT = "juliaenv"


def test_box_occupation_probabilities_hand_derived():
    # box_size=0.3, min per dim = (0.1, 0.2): p0, p1 -> box (0,0); p2, p3 -> (2,2); p4 -> (1,1),
    # so the probabilities are {0.4, 0.4, 0.2}.
    trajectory = torch.tensor([[0.1, 0.2], [0.15, 0.25], [0.9, 0.8], [0.95, 0.85], [0.5, 0.5]], dtype=torch.float64)
    probability = box_occupation_probabilities(trajectory, box_size=0.3)
    assert torch.allclose(torch.sort(probability).values, torch.tensor([0.2, 0.4, 0.4], dtype=torch.float64))


def test_box_occupation_probabilities_single_box():
    trajectory = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    probability = box_occupation_probabilities(trajectory, box_size=10.0)
    assert torch.allclose(probability, torch.tensor([1.0], dtype=torch.float64))


def test_box_occupation_probabilities_one_box_per_point():
    trajectory = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    probability = box_occupation_probabilities(trajectory, box_size=0.1)
    assert torch.allclose(torch.sort(probability).values, torch.full((4,), 0.25, dtype=torch.float64))


def test_box_occupation_probabilities_nan_trajectory_is_nan_not_silently_wrong():
    trajectory = torch.tensor([0.0, 1.0, float("nan")], dtype=torch.float64)
    probability = box_occupation_probabilities(trajectory, box_size=0.5)
    assert torch.isnan(probability).all()


def test_box_occupation_probabilities_batched_raises():
    trajectory = torch.randn(2, 5, 2, dtype=torch.float64)
    try:
        box_occupation_probabilities(trajectory, box_size=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_box_occupation_probabilities_int64_overflow_is_nan_not_silently_wrong():
    trajectory = torch.tensor([0.0, 1e10], dtype=torch.float64)
    probability = box_occupation_probabilities(trajectory, box_size=1e-10)
    assert torch.isnan(probability).all()


def test_box_occupation_probabilities_int64_overflow_exact_boundary():
    # Regression: 2**63 - 1 rounds up to 2**63 in float64, so a scaled value of 2**63 slipped past the `>` guard;
    # 2**63 - 1024 is a legitimate index and must NOT trigger it.
    exact_boundary = torch.tensor([0.0, float(2**63)], dtype=torch.float64)
    assert torch.isnan(box_occupation_probabilities(exact_boundary, box_size=1.0)).all()

    just_below = torch.tensor([0.0, float(2**63 - 1024)], dtype=torch.float64)
    probability = box_occupation_probabilities(just_below, box_size=1.0)
    assert not torch.isnan(probability).any()
    assert torch.allclose(torch.sort(probability).values, torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_box_occupation_probabilities_infinite_trajectory_is_nan_not_silently_wrong():
    # Regression: a lone -inf leaves points.min() = -inf (not NaN) and the overflow check never fires, silently
    # returning [1.0].
    with_negative_inf = torch.tensor([0.0, float("-inf"), 1.0], dtype=torch.float64)
    assert torch.isnan(box_occupation_probabilities(with_negative_inf, box_size=1.0)).all()

    all_infinite = torch.tensor([float("inf"), float("inf")], dtype=torch.float64)
    assert torch.isnan(box_occupation_probabilities(all_infinite, box_size=1.0)).all()


def test_box_counting_entropy_matches_complexitymeasures_jl():
    torch.manual_seed(0)
    trajectory = torch.randn(30, 2, dtype=torch.float64)
    box_size = 0.4

    # Row-per-line Julia matrix literal: reshape is column-major, torch's flatten is row-major, which would scramble the
    # points.
    rows = "; ".join(" ".join(f"{v:.17g}" for v in row) for row in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{rows}]
    X = StateSpaceSet(x)
    probs = probabilities(ValueHistogram({box_size}), X)
    println(join(sort(collect(probs)), ","))
    for q in (0.0, 1.0, 2.0, 3.5)
        h = entropy(Renyi(q=q, base=ℯ), ValueHistogram({box_size}), X)
        println(h)
    end
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().split("\n")
    julia_probs = torch.tensor([float(v) for v in lines[0].split(",")], dtype=torch.float64)
    julia_entropies = [float(v) for v in lines[1:]]

    torch_probs = torch.sort(box_occupation_probabilities(trajectory, box_size=box_size)).values
    assert torch.allclose(torch_probs, julia_probs, atol=1e-9)

    for order, julia_h in zip([0.0, 1.0, 2.0, 3.5], julia_entropies):
        torch_h = box_counting_entropy(trajectory, box_size=box_size, order=order)
        assert torch.isclose(torch_h, torch.tensor(julia_h, dtype=torch.float64), atol=1e-9), order


def test_box_counting_entropy_batched_and_sweep_shapes():
    torch.manual_seed(1)
    trajectory = torch.randn(3, 25, 2, dtype=torch.float64)
    sizes = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
    swept = box_counting_entropy(trajectory, box_size=sizes, order=1.0)
    assert swept.shape == (3, 3)
    for i, size in enumerate(sizes):
        expected = box_counting_entropy(trajectory, box_size=size, order=1.0)
        assert expected.shape == (3,)
        assert torch.allclose(swept[i], expected)


def test_box_counting_entropy_unbatched_scalar_is_scalar():
    trajectory = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    result = box_counting_entropy(trajectory, box_size=0.5, order=1.0)
    assert result.shape == ()


def test_box_counting_entropy_negative_order_raises():
    trajectory = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    try:
        box_counting_entropy(trajectory, box_size=0.5, order=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generalized_dimension_hand_derived_exact_power_law():
    # H(x) = 3x exactly for x = -log(box_size), so the slope is 3.
    box_size = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    neg_log_size = -torch.log(box_size)
    entropy = 3.0 * neg_log_size
    assert torch.isclose(generalized_dimension(box_size, entropy), torch.tensor(3.0, dtype=torch.float64))


def test_generalized_dimension_uniform_interval_is_near_one_for_every_order():
    # A uniform measure on an interval has D_q = 1 for every q (analytic, not Julia-derived).
    torch.manual_seed(2)
    trajectory = torch.rand(4000, dtype=torch.float64)
    sizes = torch.tensor([0.05, 0.02, 0.01, 0.005], dtype=torch.float64)
    for order in (0.0, 1.0, 2.0, 4.0):
        entropy = box_counting_entropy(trajectory, box_size=sizes, order=order)
        dimension = generalized_dimension(sizes, entropy)
        assert abs(dimension.item() - 1.0) < 0.05, (order, dimension.item())


def test_generalized_dimension_non_increasing_in_order():
    # D_q is non-increasing in q (Renyi). A p-weighted binary cascade is strongly multifractal, so its D_q spectrum
    # varies well above the finite-sample OLS noise floor (a near-uniform mixture did not).
    generator = torch.Generator().manual_seed(3)
    depth = 20
    probability_left = 0.8
    bits = torch.bernoulli(torch.full((20000, depth), 1 - probability_left, dtype=torch.float64), generator=generator)
    weights = 0.5 ** torch.arange(1, depth + 1, dtype=torch.float64)
    trajectory = (bits * weights).sum(dim=-1)
    sizes = torch.tensor([0.02, 0.01, 0.005, 0.002, 0.001], dtype=torch.float64)
    orders = [0.0, 0.5, 1.0, 2.0, 4.0]
    dimensions = [
        generalized_dimension(sizes, box_counting_entropy(trajectory, box_size=sizes, order=order)) for order in orders
    ]
    for earlier, later in zip(dimensions, dimensions[1:]):
        assert earlier >= later - 1e-6


def test_box_counting_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(4)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = box_counting_entropy(trajectory, box_size=0.5, order=1.0)
    unbatched = box_counting_entropy(trajectory[0], box_size=0.5, order=1.0)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)
