"""1-nearest-neighbor classification with Euclidean distance vs the complexity-invariant distance (CID) of
Batista, Keogh, Tataw & de Souza (2014), "CID: an efficient complexity-invariant distance for time series",
Data Mining and Knowledge Discovery 28(3), 634-669.

Two classes of z-normalized series of the same length: smooth sinusoids with random frequency and phase
(low complexity) and Lorenz x-component windows sampled at a random step, with additive noise (high
complexity). Euclidean distance between z-normalized series is dominated by phase alignment, so a complex
query is often matched with a smooth series whose shape happens to line up. CID multiplies the distance by
max(CE) / min(CE), which penalizes pairs with different complexity and removes those mismatches.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.neighbors import complexity_estimate, complexity_invariant_distance

LENGTH = 128
PER_CLASS = 40
torch.manual_seed(0)


def znormalize(series: torch.Tensor) -> torch.Tensor:
    return (series - series.mean()) / series.std(unbiased=False)


def sinusoid() -> torch.Tensor:
    frequency = 1.0 + 3.0 * torch.rand((), dtype=torch.float64)
    phase = 2 * torch.pi * torch.rand((), dtype=torch.float64)
    time = torch.linspace(0, 1, LENGTH, dtype=torch.float64)
    return znormalize(torch.sin(2 * torch.pi * frequency * time + phase))


lorenz_x = torch.as_tensor(Lorenz().make_trajectory(3000))[:, 0].to(torch.float64)


def lorenz_window() -> torch.Tensor:
    stride = int(torch.randint(1, 4, ()))  # 3000 points fit LENGTH samples at every stride up to 3
    start = int(torch.randint(0, lorenz_x.shape[0] - (LENGTH - 1) * stride, ()))
    window = lorenz_x[start : start + LENGTH * stride : stride]
    return znormalize(znormalize(window) + 0.5 * torch.randn(LENGTH, dtype=torch.float64))


def make_set(per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    series = torch.stack([sinusoid() for _ in range(per_class)] + [lorenz_window() for _ in range(per_class)])
    labels = torch.tensor([0] * per_class + [1] * per_class)
    return series.unsqueeze(-1), labels


train, train_labels = make_set(PER_CLASS)
test, test_labels = make_set(PER_CLASS)

complexity = complexity_estimate(train)
print(f"mean CE, smooth sinusoids: {complexity[train_labels == 0].mean():.3f}")
print(f"mean CE, noisy Lorenz:     {complexity[train_labels == 1].mean():.3f}")
assert complexity[train_labels == 1].min() > complexity[train_labels == 0].max()

euclidean_correct = cid_correct = 0
for query, label in zip(test, test_labels):
    euclidean = torch.linalg.vector_norm(train - query, dim=(1, 2))
    cid = complexity_invariant_distance(query, train)
    euclidean_correct += int(train_labels[euclidean.argmin()] == label)
    cid_correct += int(train_labels[cid.argmin()] == label)

total = test.shape[0]
print(f"1-NN accuracy, Euclidean: {euclidean_correct / total:.3f}")
print(f"1-NN accuracy, CID:       {cid_correct / total:.3f}")
assert cid_correct >= euclidean_correct
assert cid_correct / total > 0.95
