"""Tests for torchnlts.entropy.lempel_ziv.

Dual ground truth:
  1. Paper: hand-derived/well-known reference values -- the "1001 1110
     1100 0010" string commonly cited (e.g. in EEG Lempel-Ziv-complexity
     literature) as having complexity 6, plus minimal (constant) and
     maximally-regular (alternating) sequences.
  2. Julia: ComplexityMeasures.jl's `LempelZiv76`. Unlike this
     subpackage's other entropies, no known algorithmic deviation was
     found (this is a direct, symbol-for-symbol port of the same
     Kaspar & Schuster 1987 algorithm ComplexityMeasures.jl itself
     implements), so this dual validation is an exact-match test, not
     a neutralized-or-locked-disagreement one.
"""

import subprocess

import torch

from torchnlts.entropy.lempel_ziv import _lz76_count, lempel_ziv_complexity

JULIA_PROJECT = "juliaenv"


def _naive_phrase_count(sequence):
    """Textbook exhaustive-history phrase decomposition (Lempel & Ziv
    1976's own operational definition): each phrase is the shortest
    prefix of what remains that is not a substring of the sequence read
    so far (history + itself, up to but excluding its own last symbol).
    Brute-force substring search, structurally unrelated to the
    pointer-based Kaspar & Schuster algorithm ``_lz76_count`` implements
    -- an independent, from-scratch second ground truth, not just a
    comparison against Julia's shared algorithmic ancestry.
    """
    n = len(sequence)
    phrases = 0
    start = 0
    while start < n:
        end = start + 1
        while end <= n:
            candidate = sequence[start:end]
            history = sequence[: end - 1]
            found = any(history[j : j + len(candidate)] == candidate for j in range(len(history) - len(candidate) + 1))
            if not found:
                break
            end += 1
        phrases += 1
        start = end
    return phrases


def test_lz76_count_classic_example():
    # Widely cited example (e.g. antropy's own docstring/tests, various
    # EEG Lempel-Ziv-complexity papers): "10011110110000 10" -> 6.
    sequence = [int(c) for c in "1001111011000010"]
    assert _lz76_count(sequence) == 6


def test_lz76_count_constant_sequence_is_minimal():
    assert _lz76_count([0] * 10) == 2
    assert _lz76_count([1] * 20) == 2


def test_lz76_count_alternating_sequence():
    assert _lz76_count([0, 1] * 5) == 3


def test_lz76_count_matches_independent_phrase_decomposition_exhaustively():
    # Genuinely independent ground truth (not Julia, which implements
    # the same Kaspar & Schuster pointer algorithm): every binary
    # string of length 2-12 (8188 strings), checked against a
    # structurally unrelated brute-force phrase decomposition.
    import itertools

    for length in range(2, 13):
        for bits in itertools.product((0, 1), repeat=length):
            sequence = list(bits)
            assert _lz76_count(sequence) == _naive_phrase_count(sequence)


def test_lempel_ziv_complexity_mean_threshold_hand_derived():
    # mean = 37.5/16 = 2.34375. Below it -> 0, at/above -> 1: [1,0,0,1,1,1,1,0,1,1,0,0,0,0,1,0]
    values = [3.0, 1.0, 2.0, 4.0, 5.0, 3.5, 3.0, 0.0, 3.0, 3.0, 1.0, 2.0, 0.0, 1.0, 4.0, 2.0]
    trajectory = torch.tensor(values, dtype=torch.float64)
    assert lempel_ziv_complexity(trajectory).item() == 6


def test_lempel_ziv_complexity_explicit_threshold():
    trajectory = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    below = lempel_ziv_complexity(trajectory, threshold=0.0)  # -> [1,0,1,0,1]
    above = lempel_ziv_complexity(trajectory, threshold=2.0)  # -> [0,0,0,0,0]
    assert below.item() == _lz76_count([1, 0, 1, 0, 1])
    assert above.item() == _lz76_count([0, 0, 0, 0, 0])


def test_lempel_ziv_complexity_normalized_matches_formula():
    torch.manual_seed(0)
    trajectory = torch.randn(20, dtype=torch.float64)
    raw = lempel_ziv_complexity(trajectory)
    normalized = lempel_ziv_complexity(trajectory, normalize=True)
    expected = raw.to(torch.float64) / (20 / torch.log2(torch.tensor(20.0, dtype=torch.float64)))
    assert torch.isclose(normalized, expected)


def test_lempel_ziv_complexity_batched_matches_independent_loop():
    torch.manual_seed(1)
    trajectory = torch.randn(4, 25, 1, dtype=torch.float64)
    batched = lempel_ziv_complexity(trajectory)
    assert batched.shape == (4,)
    for i in range(4):
        expected = lempel_ziv_complexity(trajectory[i])
        assert batched[i] == expected


def test_lempel_ziv_complexity_rejects_multivariate():
    trajectory = torch.randn(20, 2, dtype=torch.float64)
    try:
        lempel_ziv_complexity(trajectory)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lempel_ziv_complexity_rejects_too_short():
    trajectory = torch.tensor([1.0], dtype=torch.float64)
    try:
        lempel_ziv_complexity(trajectory)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lz76_count_matches_complexitymeasures_jl():
    for sequence in ([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0], [0] * 10, [0, 1] * 5):
        values = ", ".join(str(v) for v in sequence)
        julia_code = f"""
        using ComplexityMeasures
        x = [{values}]
        println(complexity(LempelZiv76(), x))
        """
        result = subprocess.run(
            ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        julia_count = int(result.stdout.strip())
        assert _lz76_count(sequence) == julia_count


def test_lempel_ziv_complexity_full_pipeline_matches_complexitymeasures_jl():
    # End-to-end: mean-threshold binarization + counting on a raw,
    # continuous, random trajectory, cross-checked against Julia's own
    # binarize-then-LempelZiv76 pipeline (Julia's own docstring for
    # LempelZiv76 names mean-threshold binarization as the standard
    # convention, matching this module's default).
    torch.manual_seed(2)
    trajectory = torch.randn(50, dtype=torch.float64)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    binary = Int.(x .>= sum(x)/length(x))
    println(complexity(LempelZiv76(), binary))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    julia_count = int(result.stdout.strip())
    assert lempel_ziv_complexity(trajectory).item() == julia_count
