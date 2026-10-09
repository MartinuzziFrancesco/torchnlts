"""Reproduces the core comparison of Jiang, Lu, Orlova & Willett (2023,
NeurIPS), "Training neural operators to preserve invariant measures of
chaotic attractors": adding a distributional (whole point-cloud) loss
to ordinary pointwise MSE trains a differentiable dynamics model whose
own long rollout better matches the true attractor's invariant
measure -- and their paper reports their entropic-optimal-transport
(Sinkhorn) loss beating Maximum Mean Discrepancy (MMD) at that task.
`torchnlts.sciml.losses` already has an MMD-based loss
(`AttractorLoss`) for exactly this whole-attractor-matching purpose;
this tutorial adds `SinkhornLoss` (their entropic-OT loss) and runs
their comparison directly against it.

Setup: the same 3D Lorenz system used elsewhere in this repo's
tutorials, a small MLP standing in for the vector field, RK4
integration via torchdiffeq. Three training arms -- MSE-only,
MSE + AttractorLoss (MMD), MSE + SinkhornLoss (OT) --
each add their invariant-measure term on a short autonomous rollout
window from the model compared against an *independently sampled*
window of the true trajectory (unaligned in time -- this is
distribution matching, not trajectory tracking, unlike the pointwise
MSE term). Every model is then scored by two metrics *neither* loss
optimizes directly: top Lyapunov exponent error and correlation
(fractal) dimension error, both against the true Lorenz attractor,
averaged over 2 random seeds (not a rigorous multi-run study, but
enough to see whether a ranking is a fluke of one seed).

Honest result, not the paper's exact finding: their comparison runs a
Fourier Neural Operator on a 60-dimensional Lorenz-96 system for 60
epochs on 4 GPUs; this is a 3-state MLP on 2 CPU seeds for 150 epochs,
several orders of magnitude smaller. What the assertions below actually
check, precisely (see them for the exact numbers each run produces):
Sinkhorn beats MSE-only on both metrics' 2-seed average; MMD beats
MSE-only on the correlation-dimension average only -- this script makes
no claim, asserted or otherwise, about MMD's Lyapunov-exponent average
against MSE-only, nor about Sinkhorn against MMD directly. The finer
"Sinkhorn beats MMD" claim is not asserted at all: across the 2 seeds
actually run, Sinkhorn was at least as good as MMD on every individual
metric, and one MMD run's recovered Lyapunov exponent overshot badly
where Sinkhorn's did not -- both are just observations from these two
runs, not a general claim, and 2 seeds is nowhere near enough to call a
finer ranking or a bandwidth-sensitivity explanation established.
"""

import time

import torch
import torchdiffeq

from torchnlts.dimension import correlation_dimension, correlation_sum
from torchnlts.lyapunov import lyapunov_spectrum
from torchnlts.sciml import AttractorLoss, SinkhornLoss

torch.manual_seed(0)
dtype = torch.float64
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
dt = 0.01
window = 30


def lorenz_vector_field(t, x):
    dx = sigma * (x[..., 1] - x[..., 0])
    dy = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
    dz = x[..., 0] * x[..., 1] - beta * x[..., 2]
    return torch.stack([dx, dy, dz], dim=-1)


def integrate(vector_field, x, t_span=(0.0, dt)):
    return torchdiffeq.odeint(vector_field, x, torch.tensor(t_span, dtype=x.dtype), method="rk4")[-1]


def rollout(step_fn, initial_state, num_steps):
    state = initial_state
    states = [state]
    for _ in range(num_steps):
        state = step_fn(state)
        states.append(state)
    return torch.stack(states)


def true_step(x):
    return integrate(lorenz_vector_field, x)


state = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
for _ in range(2000):  # discard transient, settle onto the attractor
    state = true_step(state)
trajectory = rollout(true_step, state, 6000)
_mean, _std = trajectory.mean(dim=0), trajectory.std(dim=0)


class LearnedVectorField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 3),
        ).to(dtype)

    def forward(self, t, x):
        return self.net((x - _mean) / _std) / _std


def train(invariant_loss, invariant_weight: float, num_epochs: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    vector_field = LearnedVectorField()
    optimizer = torch.optim.Adam(vector_field.parameters(), lr=2e-3)
    x_train, y_train = trajectory[:-1], trajectory[1:]
    num_train = x_train.shape[0]
    batch_size = 256

    for epoch in range(num_epochs):
        for indices in torch.randperm(num_train).split(batch_size):
            optimizer.zero_grad()
            predicted = integrate(vector_field, x_train[indices])
            loss = ((predicted - y_train[indices]) ** 2).mean()
            if invariant_loss is not None:
                predicted_start = torch.randint(0, num_train - window, (1,)).item()
                predicted_window = rollout(lambda x: integrate(vector_field, x), trajectory[predicted_start], window)
                target_start = torch.randint(0, num_train - window, (1,)).item()
                target_window = trajectory[target_start : target_start + window + 1]
                loss = loss + invariant_weight * invariant_loss(predicted_window, target_window)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vector_field.parameters(), max_norm=1.0)
            optimizer.step()
    return vector_field


def invariant_errors(vector_field: torch.nn.Module, radius: torch.Tensor, true_lyapunov: float, true_dimension: float):
    with torch.no_grad():
        model_step = lambda x: integrate(vector_field, x)
        lyapunov = lyapunov_spectrum(
            model_step, trajectory[0], num_steps=3000, dt=dt, num_exponents=1, transient_steps=500
        ).item()
        rollout_states = rollout(model_step, trajectory[0], 3000)[500:]
        normalized = (rollout_states - rollout_states.mean(dim=0)) / rollout_states.std(dim=0)
        dimension = correlation_dimension(radius, correlation_sum(normalized, radius, theiler_window=10)).item()
    return abs(lyapunov - true_lyapunov), abs(dimension - true_dimension)


radius = torch.logspace(-1.5, -0.3, 12, dtype=torch.float64)
true_lyapunov = lyapunov_spectrum(true_step, trajectory[0], num_steps=3000, dt=dt, num_exponents=1).item()
true_normalized = (trajectory - trajectory.mean(dim=0)) / trajectory.std(dim=0)
true_dimension = correlation_dimension(radius, correlation_sum(true_normalized, radius, theiler_window=10)).item()
print(f"true top Lyapunov exponent: {true_lyapunov:.4f}, true correlation dimension: {true_dimension:.4f}\n")

configs = [
    ("MSE-only", None, 0.0),
    ("MSE + AttractorLoss (MMD)", AttractorLoss(temperature=10.0), 0.3),
    ("MSE + SinkhornLoss (OT)", SinkhornLoss(reg=15.0, num_iters=50), 0.3),
]

errors = {name: {"le": [], "fd": []} for name, _, _ in configs}
for seed in (1, 2):
    print(f"-- seed {seed} --")
    for name, loss_module, weight in configs:
        t0 = time.time()
        model = train(loss_module, weight, num_epochs=150, seed=seed)
        le_error, fd_error = invariant_errors(model, radius, true_lyapunov, true_dimension)
        print(f"  {name}: LE error={le_error:.4f}, FD error={fd_error:.4f}  ({time.time() - t0:.0f}s)")
        errors[name]["le"].append(le_error)
        errors[name]["fd"].append(fd_error)

print("\naveraged over both seeds:")
averages = {}
for name, _, _ in configs:
    le_mean = sum(errors[name]["le"]) / len(errors[name]["le"])
    fd_mean = sum(errors[name]["fd"]) / len(errors[name]["fd"])
    averages[name] = (le_mean, fd_mean)
    print(f"  {name}: LE error={le_mean:.4f}, FD error={fd_mean:.4f}")

mse_le, mse_fd = averages["MSE-only"]
sinkhorn_le, sinkhorn_fd = averages["MSE + SinkhornLoss (OT)"]
mmd_le, mmd_fd = averages["MSE + AttractorLoss (MMD)"]

assert sinkhorn_le < mse_le, "Sinkhorn-trained model should recover a closer Lyapunov exponent than MSE-only"
assert sinkhorn_fd < mse_fd, "Sinkhorn-trained model should recover a closer correlation dimension than MSE-only"
assert mmd_fd < mse_fd, "MMD-trained model should also recover a closer correlation dimension than MSE-only"
print("\nAll checks passed -- exactly these three comparisons, no others (not MMD's Lyapunov average against")
print("MSE-only, not Sinkhorn against MMD directly): see the module docstring for what these two seeds'")
print("numbers above show about the unasserted Sinkhorn-vs-MMD comparison, and its limits.")
