"""Training a differentiable Lorenz model with `SinkhornLoss` and `AttractorLoss`.

Setting of Jiang, Lu, Orlova & Willett (2023, NeurIPS), "Training neural operators to
preserve invariant measures of chaotic attractors": a model trained on one-step MSE alone can
have the wrong long-term statistics, and a second loss that compares the distribution of the
states visited by the model's own rollouts with the data's distribution is meant to fix that.
Their loss is an entropic optimal transport (Sinkhorn) divergence, which they report beating
maximum mean discrepancy (MMD). `torchnlts.sciml.SinkhornLoss` is that divergence and
`torchnlts.sciml.AttractorLoss` is the MMD.

Three models are trained from the same seeds: a small MLP as the Lorenz vector field, RK4 steps
through torchdiffeq, and 800 Adam steps on MSE only, MSE + `AttractorLoss`, or MSE +
`SinkhornLoss`. At every step the second term compares the pooled states of 8 short model
rollouts (10 steps, started at random data points) with random states of the true trajectory:
a point cloud against a point cloud, not a trajectory against a trajectory. Each model is
scored by the top Lyapunov exponent and the correlation dimension of its own long rollout,
which neither loss optimizes, by its one-step MSE, and by the two second terms evaluated on
fresh rollouts.

Honest result, not the paper's finding: the second term does nothing measurable here. Its value
on fresh rollouts is the same for all three models to within 0.3%, because rollouts of 10 steps
that start at data points stay near the data's distribution whatever the model; rollouts long
enough to reach the model's own invariant measure cost far more than this tutorial allows. The
rows therefore differ only by training noise. No model is chaotic (every exponent is within 0.2
of zero, the true one is 0.86). Over 8 seeds (set `seeds`; about 5 minutes) the mean errors of
the three rows agree to within 0.04, against seed-to-seed standard deviations of 0.05 to 0.10
(exponent) and 0.34 to 0.38 (dimension). The two default seeds happen to show smaller exponent
errors for the second terms, which is not an effect.
"""

import time

import torch
import torchdiffeq

from torchnlts.dimension import correlation_dimension, correlation_sum
from torchnlts.lyapunov import lyapunov_spectrum
from torchnlts.sciml import AttractorLoss, SinkhornLoss

torch.set_num_threads(1)  # tiny tensors: threads only add overhead
torch.manual_seed(0)
dtype = torch.float64
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
dt = 0.01
seeds = (1, 2)
num_steps, batch_size = 800, 256
num_starts, rollout_length = 8, 10
eval_steps, eval_transient, truth_steps = 1500, 300, 3000


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
for _ in range(1000):  # discard transient, settle onto the attractor
    state = true_step(state)
trajectory = rollout(true_step, state, 4000)
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


def train(invariant_loss, invariant_weight: float, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    vector_field = LearnedVectorField()
    optimizer = torch.optim.Adam(vector_field.parameters(), lr=2e-3)
    x_train, y_train = trajectory[:-1], trajectory[1:]
    num_train = x_train.shape[0]

    for _ in range(num_steps):
        indices = torch.randint(0, num_train, (batch_size,))
        optimizer.zero_grad()
        loss = ((integrate(vector_field, x_train[indices]) - y_train[indices]) ** 2).mean()
        if invariant_loss is not None:
            starts = torch.randint(0, num_train, (num_starts,))
            states = rollout(lambda x: integrate(vector_field, x), trajectory[starts], rollout_length)
            predicted_cloud = states.reshape(-1, 3)
            target_cloud = trajectory[torch.randint(0, num_train, (predicted_cloud.shape[0],))]
            loss = loss + invariant_weight * invariant_loss(predicted_cloud, target_cloud)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vector_field.parameters(), max_norm=1.0)
        optimizer.step()
    return vector_field


def scores(vector_field: torch.nn.Module) -> dict:
    with torch.no_grad():
        model_step = lambda x: integrate(vector_field, x)
        lyapunov = lyapunov_spectrum(
            model_step, trajectory[0], num_steps=eval_steps, dt=dt, num_exponents=1, transient_steps=eval_transient
        ).item()
        rollout_states = rollout(model_step, trajectory[0], eval_steps)[eval_transient:]
        normalized = (rollout_states - _mean) / _std
        dimension = correlation_dimension(radius, correlation_sum(normalized, radius, theiler_window=10)).item()
        one_step_mse = ((integrate(vector_field, trajectory[:-1]) - trajectory[1:]) ** 2).mean().item()
        # The two training terms on fresh rollouts (the same draws for every model).
        generator = torch.Generator().manual_seed(123)
        divergences = torch.zeros(2, dtype=dtype)
        for _ in range(50):
            starts = torch.randint(0, trajectory.shape[0], (num_starts,), generator=generator)
            cloud = rollout(model_step, trajectory[starts], rollout_length).reshape(-1, 3)
            target = trajectory[torch.randint(0, trajectory.shape[0], (cloud.shape[0],), generator=generator)]
            divergences += torch.stack([sinkhorn_loss(cloud, target), mmd_loss(cloud, target)]) / 50
    return {
        "LE error": abs(lyapunov - true_lyapunov),
        "FD error": abs(dimension - true_dimension),
        "one-step MSE": one_step_mse,
        "Sinkhorn": divergences[0].item(),
        "MMD": divergences[1].item(),
        "LE": lyapunov,
        "FD": dimension,
    }


radius = torch.logspace(-1.0, -0.3, 12, dtype=dtype)
true_lyapunov = lyapunov_spectrum(true_step, trajectory[0], num_steps=truth_steps, dt=dt, num_exponents=1).item()
true_normalized = (trajectory - _mean) / _std
true_dimension = correlation_dimension(radius, correlation_sum(true_normalized, radius, theiler_window=10)).item()
print(f"true top Lyapunov exponent: {true_lyapunov:.4f}, true correlation dimension: {true_dimension:.4f}\n")

# Sinkhorn values are in squared state units (tens here) and MMD values are below 1, so the weights differ.
mmd_loss = AttractorLoss(temperature=10.0)
sinkhorn_loss = SinkhornLoss(reg=15.0, num_iters=10)
configs = [
    ("MSE-only", None, 0.0),
    ("MSE + AttractorLoss (MMD)", mmd_loss, 0.3),
    ("MSE + SinkhornLoss (OT)", sinkhorn_loss, 0.001),
]

results = {name: [] for name, _, _ in configs}
start_time = time.time()
for seed in seeds:
    print(f"-- seed {seed} --")
    for name, loss_module, weight in configs:
        results[name].append(scores(train(loss_module, weight, seed=seed)))
        row = results[name][-1]
        print(
            f"  {name}: LE={row['LE']:.3f} (error {row['LE error']:.3f}), FD={row['FD']:.3f} (error {row['FD error']:.3f}), "
            f"one-step MSE={row['one-step MSE']:.3f}, Sinkhorn={row['Sinkhorn']:.2f}, MMD={row['MMD']:.4f}"
        )

print(f"\naveraged over {len(seeds)} seeds:")
averages = {}
for name, _, _ in configs:
    averages[name] = {key: sum(run[key] for run in results[name]) / len(seeds) for key in results[name][0]}
    row = averages[name]
    print(
        f"  {name}: LE error={row['LE error']:.3f}, FD error={row['FD error']:.3f}, "
        f"one-step MSE={row['one-step MSE']:.3f}, Sinkhorn={row['Sinkhorn']:.2f}, MMD={row['MMD']:.4f}"
    )
print(f"({time.time() - start_time:.0f}s)")

for name, _, _ in configs[1:]:
    assert averages[name]["one-step MSE"] < 1.1 * averages["MSE-only"]["one-step MSE"], name
print("\nAll checks passed: neither second term costs one-step accuracy (no other claim is asserted).")
