"""Tests for torchnlts.entropy.approximate.

Dual ground truth:
  1. Paper: Pincus (1991), eq. 1-4 -- hand-derived hard and soft examples
     (the hard one additionally cross-checked at runtime against
     ComplexityMeasures.jl on the same random-data setup elsewhere in
     this file, with matching parameters; the soft one against a
     from-scratch pure-Python oracle computed once, before writing the
     test, and hardcoded here as the expected value) plus a hand-derived
     self-excluded ("modified ApEn") degenerate case.
  2. Julia: ComplexityMeasures.jl's ``ApproximateEntropy`` (self-inclusive
     by construction, matching this module's default ``theiler_window=-1``)
     -- except at ``embedding_dim=1``, where that package deliberately
     special-cases its own definition (see the module docstring and the
     locked disagreement test below).
"""

import math
import subprocess

import torch

from torchnlts.entropy.approximate import approximate_entropy, soft_approximate_entropy

JULIA_PROJECT = "juliaenv"


def test_approximate_entropy_hand_derived():
    # embedding_dim=2, delay=1, r=1.5. m=2 windows (0,1),(1,2),(2,3) of
    # [1,2,1,3] are (1,2),(2,1),(1,3); Chebyshev distances 1, 1, 2 give
    # per-window conditional probabilities 1, 2/3, 2/3 (self included) ->
    # phi(2) = mean(log(1), log(2/3), log(2/3)).
    # m=3 windows (1,2,1),(2,1,3): Chebyshev distance 2 (> 1.5) -> each
    # window's only other candidate is excluded -> phi(3) = log(1/2).
    # ApEn = phi(2) - phi(3).
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    result = approximate_entropy(trajectory, radius=1.5, embedding_dim=2, delay=1)
    assert torch.isclose(result, torch.tensor(0.4228371084878357, dtype=torch.float64))


def test_approximate_entropy_delay_2():
    # embedding_dim=2, delay=2: exercises a nonunit lag. Trajectory
    # [1,2,1,3,5,0], m=2 windows at lag 2 are (1,1),(2,3),(1,5),(3,0);
    # Chebyshev distance matrix (off-diagonal):
    #   w0-w1=2, w0-w2=4, w0-w3=2, w1-w2=2, w1-w3=3, w2-w3=5
    # At r=2.5 (self-inclusive): C = [3/4, 3/4, 1/2, 1/2] ->
    # phi(2) = (2*log(0.75) + 2*log(0.5)) / 4.
    # m=3 windows at lag 2 are (1,1,5),(2,3,0) (only 2, since
    # N - (m-1)*delay = 6 - 4 = 2); their Chebyshev distance is
    # max(1,2,5)=5 > 2.5, so each only matches itself: phi(3) = log(0.5).
    # ApEn = phi(2) - phi(3), asserted exactly rather than just checking
    # finiteness, so a delay that's silently ignored (e.g. defaulting to
    # 1) would be caught -- confirmed it would be by deliberately
    # breaking delay-forwarding and re-running this assertion.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 5.0, 0.0], dtype=torch.float64)
    embedded = torch.tensor([[1.0, 1.0], [2.0, 3.0], [1.0, 5.0], [3.0, 0.0]], dtype=torch.float64)
    from torchnlts.embedding.delay import delay_embedding

    assert torch.equal(delay_embedding(trajectory, embedding_dim=2, delay=2), embedded)
    result = approximate_entropy(trajectory, radius=2.5, embedding_dim=2, delay=2)
    phi_2 = (2 * math.log(0.75) + 2 * math.log(0.5)) / 4
    phi_3 = math.log(0.5)
    assert torch.isclose(result, torch.tensor(phi_2 - phi_3, dtype=torch.float64))


def test_approximate_entropy_inclusive_radius_boundary():
    # Same trajectory/embedding as the hand-derived test: the m=2
    # windows' nonzero Chebyshev distances are exactly 1 and 2 (see that
    # test), so radius=1.0 (exactly the smaller distance) qualifies the
    # same pairs as radius=1.5 under an *inclusive* "<=" -- if this
    # module instead used a strict "<", dropping the radius to exactly
    # 1.0 would exclude those distance-1 pairs and change the result.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    at_boundary = approximate_entropy(trajectory, radius=1.0, embedding_dim=2, delay=1)
    above_boundary = approximate_entropy(trajectory, radius=1.5, embedding_dim=2, delay=1)
    assert torch.isclose(at_boundary, above_boundary)


def test_log_conditional_probability_positive_theiler_window_unequal_row_denominators():
    # 5 points [0, 10, 1, 11, 2], theiler_window=1, m=1 (windows == raw
    # points, tested directly via the private per-row helper so this
    # isolates the row-wise mask.sum(dim=-1) normalization rather than
    # depending on a second (m+1) embedding too).
    # Off-diagonal Chebyshev distances (|i-j|>1 only, else excluded):
    #   row0 (j=2,3,4): 1, 11, 2  |  row1 (j=3,4): 1, 8
    #   row2 (j=0,4):   1, 1      |  row3 (j=0,1): 11, 1
    #   row4 (j=0,1,2): 2, 8, 1
    # At r=1.5: matching counts / row denominators = 1/3, 1/2, 2/2, 1/2, 1/3
    # -- genuinely unequal denominators (2 vs 3) and unequal
    # probabilities, so a normalization bug (e.g. a shared denominator
    # instead of a per-row one) would change this value -- confirmed it
    # would by deliberately breaking the normalization and re-running
    # this assertion.
    from torchnlts.entropy.approximate import _log_conditional_probability

    trajectory = torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0], dtype=torch.float64)
    embedded = trajectory.unsqueeze(-1)
    result = _log_conditional_probability(embedded, radius=1.5, theiler_window=1, norm_order=float("inf"))
    expected = (2 * math.log(1 / 3) + 2 * math.log(1 / 2) + math.log(1.0)) / 5
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_approximate_entropy_self_excluded_degenerate_case():
    # Same trajectory/radius, theiler_window=0 (exclude only self, the
    # "modified ApEn" variant). The m=3 embedding has only 2 windows, and
    # excluding self leaves each with a single candidate at Chebyshev
    # distance 2 > 1.5 -- zero conditional probability, log(0) = -inf for
    # phi(3), so ApEn = phi(2) - (-inf) = +inf. Not a crash: an honest
    # consequence of a 2-window statistic with self excluded.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    result = approximate_entropy(trajectory, radius=1.5, embedding_dim=2, delay=1, theiler_window=0)
    assert torch.isinf(result) and result > 0


def test_approximate_entropy_radius_sweep():
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    result = approximate_entropy(trajectory, radius=torch.tensor([1.5, 3.0], dtype=torch.float64))
    expected_at_1_5 = approximate_entropy(trajectory, radius=1.5)
    expected_at_3_0 = approximate_entropy(trajectory, radius=3.0)
    assert torch.allclose(result, torch.stack([expected_at_1_5, expected_at_3_0]))


def test_approximate_entropy_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 30, 2, dtype=torch.float64)
    radii = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    swept = approximate_entropy(trajectory, radius=radii, embedding_dim=2)
    assert swept.shape == (3, 4)
    for i, radius in enumerate(radii):
        expected = approximate_entropy(trajectory, radius=radius, embedding_dim=2)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected, equal_nan=True)


def test_approximate_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = approximate_entropy(trajectory, radius=1.0, embedding_dim=2)
    unbatched = approximate_entropy(trajectory[0], radius=1.0, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_approximate_entropy_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0])
    result = approximate_entropy(trajectory, radius=0.5, embedding_dim=1, theiler_window=1)
    assert torch.isnan(result)


def test_approximate_entropy_vmap_compatible():
    torch.manual_seed(2)
    trajectory = torch.randn(2, 20, 1, dtype=torch.float64)
    direct = approximate_entropy(trajectory, radius=1.0, embedding_dim=2)
    vmapped = torch.vmap(lambda t: approximate_entropy(t, radius=1.0, embedding_dim=2))(trajectory)
    assert torch.allclose(direct, vmapped, equal_nan=True)


def test_approximate_entropy_nan_input_gives_nan_output():
    # A NaN anywhere in a window makes every distance from that window
    # (including to itself) NaN, so a naive `<= radius` comparison would
    # silently evaluate to "no match" instead of propagating NaN -- not
    # raised, since a raise would need a data-dependent Python `if` on
    # the tensor's contents, which rejects torch.vmap's BatchedTensor
    # inputs outright.
    trajectory = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    assert torch.isnan(approximate_entropy(trajectory, radius=1.5, embedding_dim=2))


def test_approximate_entropy_dtype_preserved_float32():
    torch.manual_seed(7)
    trajectory = torch.randn(20, dtype=torch.float32)
    result = approximate_entropy(trajectory, radius=1.0, embedding_dim=2)
    assert result.dtype == torch.float32


def test_approximate_entropy_embedding_dim_one_disagrees_with_causalitytools_jl():
    # ComplexityMeasures.jl's ApproximateEntropy special-cases m=1
    # (confirmed by reading its source: approximate_entropy.jl returns
    # phi(1) alone rather than phi(1) - phi(2), citing a different paper
    # than Pincus for that case), so this module's Pincus-faithful
    # embedding_dim=1 output does not match it. If a future
    # ComplexityMeasures.jl version changes this, this test will start
    # failing, which is the point -- it means the documented discrepancy
    # has gone stale and should be revisited.
    trajectory = [0.0, 1.0, 2.0]
    radius = 0.5
    julia_code = f"""
    using ComplexityMeasures
    x = {trajectory}
    est = ApproximateEntropy(m=1, τ=1, r={radius})
    println(complexity(est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_apen = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_apen = approximate_entropy(torch.tensor(trajectory, dtype=torch.float64), radius=radius, embedding_dim=1)
    # Exact expected values (not just "these differ"): phi(1) alone
    # (Julia's special case) is -ln(3); phi(1) - phi(2) (this module's
    # Pincus-faithful definition) is ln(2/3).
    assert torch.isclose(julia_apen, torch.tensor(-math.log(3), dtype=torch.float64))
    assert torch.isclose(torch_apen, torch.tensor(math.log(2 / 3), dtype=torch.float64))


def test_soft_approximate_entropy_low_temperature_limit_matches_radius_zero_hard():
    trajectory = torch.tensor([0.0, 1.0, 0.3, 1.0], dtype=torch.float64)
    soft = soft_approximate_entropy(trajectory, temperature=1e-6)
    hard = approximate_entropy(trajectory, radius=0.0)
    assert torch.allclose(soft, hard, atol=1e-4)


def test_soft_approximate_entropy_hand_derived_finite_temperature():
    # embedding_dim=1, delay=1, temperature=1.0, gaussian kernel, on
    # [0, 1, 3] -- a genuine finite-temperature forward-value check, not
    # just a low-temperature limit or a gradcheck (which only verifies
    # the gradient matches whatever the forward pass happens to
    # compute, not that the forward value itself is correct).
    # Independently derived in pure Python (no torchnlts code), from
    # gaussian weight w(d) = exp(-(d/T)^2/2):
    #   m=1 windows are the raw points themselves; C_i = mean_j w(|x_i-x_j|)
    #   -> phi(1) = mean_i log(C_i)
    #   m=2 windows are (0,1),(1,3), Chebyshev distance 2 between them
    #   -> phi(2) = mean_i log(C_i)
    #   ApEn = phi(1) - phi(2)
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    result = soft_approximate_entropy(trajectory, temperature=1.0, embedding_dim=1)
    assert torch.isclose(result, torch.tensor(-0.14152971049673713, dtype=torch.float64))


def test_soft_approximate_entropy_nan_input_gives_nan_output():
    # trajectory_kernel treats a NaN distance the same as an excluded
    # (infinite) one -- weight 0, not NaN -- so without an explicit
    # guard, a NaN window would silently contribute "definitely no
    # match" instead of propagating NaN.
    trajectory = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    assert torch.isnan(soft_approximate_entropy(trajectory, temperature=0.5, embedding_dim=2))


def test_soft_approximate_entropy_batched_and_sweep_shapes():
    torch.manual_seed(8)
    trajectory = torch.randn(4, 20, 2, dtype=torch.float64)
    temperatures = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
    swept = soft_approximate_entropy(trajectory, temperature=temperatures, embedding_dim=2)
    assert swept.shape == (3, 4)
    for i, temperature in enumerate(temperatures):
        expected = soft_approximate_entropy(trajectory, temperature=temperature, embedding_dim=2)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected)


def test_soft_approximate_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(9)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = soft_approximate_entropy(trajectory, temperature=0.5, embedding_dim=2)
    unbatched = soft_approximate_entropy(trajectory[0], temperature=0.5, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_soft_approximate_entropy_vmap_compatible():
    torch.manual_seed(10)
    trajectory = torch.randn(2, 20, 1, dtype=torch.float64)
    direct = soft_approximate_entropy(trajectory, temperature=0.5, embedding_dim=2)
    vmapped = torch.vmap(lambda t: soft_approximate_entropy(t, temperature=0.5, embedding_dim=2))(trajectory)
    assert torch.allclose(direct, vmapped)


def test_soft_approximate_entropy_gradcheck():
    torch.manual_seed(3)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return soft_approximate_entropy(x, temperature=0.5, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_soft_approximate_entropy_gradcheck_exponential():
    torch.manual_seed(4)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return soft_approximate_entropy(x, temperature=0.5, embedding_dim=2, kind="exponential")

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_soft_approximate_entropy_gradcheck_temperature():
    torch.manual_seed(5)
    trajectory = torch.randn(15, dtype=torch.float64)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_approximate_entropy(trajectory, temperature=t, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (temperature,))


def test_approximate_entropy_matches_complexitymeasures_jl():
    torch.manual_seed(6)
    trajectory = torch.randn(30, dtype=torch.float64)
    radius = 1.0
    embedding_dim = 2

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = ApproximateEntropy(m={embedding_dim}, τ=1, r={radius})
    println(complexity(est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_apen = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_apen = approximate_entropy(trajectory, radius=radius, embedding_dim=embedding_dim)
    # rtol=atol=1e-9: both sides do the same self-inclusive Chebyshev
    # counting and log/mean reduction in float64 on identical data, so
    # the residual is float64 parsed-string round-trip noise, not model
    # mismatch (unlike embedding_dim=1, see the dedicated disagreement
    # test above -- this test uses embedding_dim=2, which isn't special-cased).
    assert torch.isclose(torch_apen, julia_apen, rtol=1e-9, atol=1e-9)
