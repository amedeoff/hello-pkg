from __future__ import annotations

import numpy as np
import scipy.special
import torch
import torch.nn.functional as F


K = 4


def categorical_sample(probabilities):
    shape = probabilities.shape[:-1]
    return torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape(shape)


def cosine_alpha_bars(steps: int, device):
    grid = torch.linspace(0, 1, steps + 1, device=device)
    offset = 0.008
    values = torch.cos((grid + offset) / (1 + offset) * torch.pi / 2).square()
    return (values / values[0]).clamp(min=1e-5, max=1.0)


def d3pm_training_batch(tokens, steps: int):
    device = tokens.device
    batch = tokens.shape[0]
    timestep = torch.randint(1, steps + 1, (batch,), device=device)
    alpha_bar = cosine_alpha_bars(steps, device)[timestep]
    clean = F.one_hot(tokens, K).float()
    probabilities = alpha_bar[:, None, None] * clean + (1 - alpha_bar[:, None, None]) / K
    noisy = categorical_sample(probabilities)
    return F.one_hot(noisy, K).float(), timestep.float() / steps


@torch.no_grad()
def sample_d3pm(model, signal, steps: int):
    device = signal.device
    batch, length = signal.shape[:2]
    current = torch.randint(K, (batch, length), device=device)
    alpha_bars = cosine_alpha_bars(steps, device)
    eye = torch.eye(K, device=device)
    for timestep in range(steps, 0, -1):
        time = torch.full((batch,), timestep / steps, device=device)
        logits = model(F.one_hot(current, K).float(), signal, time)
        clean_probabilities = logits.softmax(dim=-1)
        alpha_previous = alpha_bars[timestep - 1]
        alpha_step = (alpha_bars[timestep] / alpha_previous).clamp(0, 1)
        # q(x_{t-1}=i | x_0=k), rows k, columns i.
        q_previous = alpha_previous * eye + (1 - alpha_previous) / K
        # q(x_t=j | x_{t-1}=i) for the observed j.
        observed = F.one_hot(current, K).float()
        likelihood = alpha_step * observed + (1 - alpha_step) / K
        unnormalized = q_previous[None, None, :, :] * likelihood[:, :, None, :]
        posterior_given_clean = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        posterior = (clean_probabilities[..., :, None] * posterior_given_clean).sum(dim=-2)
        current = categorical_sample(posterior)
    return current


def discrete_fm_training_batch(tokens):
    device = tokens.device
    batch = tokens.shape[0]
    time = torch.rand(batch, device=device)
    source = torch.randint(K, tokens.shape, device=device)
    use_target = torch.rand(tokens.shape, device=device) < time[:, None]
    noisy = torch.where(use_target, tokens, source)
    return F.one_hot(noisy, K).float(), time


@torch.no_grad()
def sample_discrete_fm(model, signal, steps: int):
    device = signal.device
    batch, length = signal.shape[:2]
    current = torch.randint(K, (batch, length), device=device)
    for index in range(steps):
        time_value = index / steps
        time = torch.full((batch,), time_value, device=device)
        probabilities = model(F.one_hot(current, K).float(), signal, time).softmax(dim=-1)
        off_diagonal = probabilities.clone()
        off_diagonal.scatter_(-1, current[..., None], 0.0)
        factor = (1.0 / steps) / max(1.0 - time_value, 1.0 / steps)
        transition = off_diagonal * factor
        stay = 1.0 - transition.sum(dim=-1, keepdim=True)
        transition.scatter_add_(-1, current[..., None], stay)
        if transition.min() < -1e-6 or not torch.allclose(
            transition.sum(-1), torch.ones_like(transition[..., 0]), atol=1e-5
        ):
            raise RuntimeError("Invalid discrete FM transition probabilities")
        current = categorical_sample(transition.clamp_min(0))
    return current


def dirichlet_fm_training_batch(tokens, alpha_scale: float, alpha_max: float,
                                prior_pseudocount: float):
    device = tokens.device
    batch, length = tokens.shape
    alpha = 1 + torch.distributions.Exponential(1 / alpha_scale).sample((batch,)).to(device)
    alpha = alpha.clamp(max=alpha_max)
    clean = F.one_hot(tokens, K).float()
    concentrations = torch.ones(batch, length, K, device=device)
    concentrations = concentrations + clean * (alpha[:, None, None] - 1)
    noisy = torch.distributions.Dirichlet(concentrations).sample()
    prior_weight = prior_pseudocount / (alpha + prior_pseudocount - 1)
    expanded = torch.cat([
        noisy * (1 - prior_weight[:, None, None]),
        noisy * prior_weight[:, None, None],
    ], dim=-1)
    normalized_time = (alpha - 1) / (alpha_max - 1)
    return expanded, normalized_time


def simplex_project(values):
    flat = values.reshape(-1, values.shape[-1])
    sorted_values, _ = torch.sort(flat, descending=True, dim=-1)
    cumulative = sorted_values.cumsum(-1) - 1
    divisors = torch.arange(1, flat.shape[-1] + 1, device=flat.device, dtype=flat.dtype)
    support = (sorted_values > cumulative / divisors).sum(-1, keepdim=True)
    row = torch.arange(flat.shape[0], device=flat.device)[:, None]
    threshold = (cumulative / divisors)[row, support - 1]
    return torch.clamp(flat - threshold, min=0).reshape_as(values)


class DirichletConditionalFlow:
    def __init__(self, alpha_max: float, spacing: float = 0.01, grid_size: int = 1000):
        self.alphas = np.arange(1, alpha_max + spacing, spacing)
        self.points = np.linspace(0, 1, grid_size)
        cdfs = np.asarray([
            scipy.special.betainc(alpha, K - 1, self.points) for alpha in self.alphas
        ])
        self.derivative = np.diff(cdfs, axis=0) / spacing

    def c_factor(self, values, alpha: float):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            beta = scipy.special.beta(alpha, K - 1)
            scale = np.where(values < 1, beta / ((1 - values) ** (K - 1)), 0)
            scale = np.where(values ** (alpha - 1) > 0, scale / (values ** (alpha - 1)), 0)
            index = min(np.argmin(np.abs(alpha - self.alphas)), len(self.derivative) - 1)
            interpolated = -np.interp(values, self.points, self.derivative[index])
            result = interpolated * scale
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def sample_dirichlet_fm(model, signal, steps: int, alpha_max: float,
                        prior_pseudocount: float):
    device = signal.device
    batch, length = signal.shape[:2]
    state = torch.distributions.Dirichlet(torch.ones(K, device=device)).sample((batch, length))
    eye = torch.eye(K, device=device)
    conditional_flow = DirichletConditionalFlow(alpha_max)
    repairs = 0
    alpha_grid = torch.linspace(1, alpha_max, steps + 1, device=device)
    for current_alpha, next_alpha in zip(alpha_grid[:-1], alpha_grid[1:]):
        alpha_value = float(current_alpha)
        prior_weight = prior_pseudocount / (current_alpha + prior_pseudocount - 1)
        expanded = torch.cat([state * (1 - prior_weight), state * prior_weight], dim=-1)
        time = torch.full((batch,), (alpha_value - 1) / (alpha_max - 1), device=device)
        probabilities = model(expanded, signal, time).softmax(dim=-1)
        factors = torch.from_numpy(conditional_flow.c_factor(state.cpu().numpy(), alpha_value)).to(state)
        conditional = (eye - state.unsqueeze(-1)) * factors.unsqueeze(-2)
        vector_field = (probabilities.unsqueeze(-2) * conditional).sum(-1)
        state = state + vector_field * (next_alpha - current_alpha)
        invalid = (state < 0).any() or not torch.isfinite(state).all() or not torch.allclose(
            state.sum(-1), torch.ones_like(state[..., 0]), atol=1e-4
        )
        if invalid:
            repairs += 1
            state = simplex_project(torch.nan_to_num(state, nan=0.25, posinf=1.0, neginf=0.0))
    return state.argmax(-1), {"simplex_repairs": repairs}
