"""MAP and diagonal weight Laplace, following fno_laplace.py / torch_laplace.py."""

from statistics import NormalDist

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
import numpy as np


def parameter_l2(model):
    return sum(jnp.sum(x**2) for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))


def map_loss(prediction, target, model, log_noise, total_scalars, prior_precision):
    data_loss = 0.5 * jnp.mean((prediction - target)**2) * jnp.exp(-2 * log_noise) + log_noise
    prior_loss = 0.5 * prior_precision * parameter_l2(model) / total_scalars
    return data_loss + prior_loss, (data_loss, prior_loss)


@eqx.filter_jit
def hutchinson_hessian_diag(loss_fn, theta, batch, key, probes):
    """Average z * (H z) for independent Rademacher vectors z; H is the true Hessian."""
    gradient = jax.grad(lambda weights: loss_fn(weights, batch))

    def probe(i, diagonal):
        vector = jr.rademacher(jr.fold_in(key, i), theta.shape, dtype=theta.dtype)
        hvp = jax.jvp(gradient, (theta,), (vector,))[1]
        return diagonal + vector * hvp

    return jax.lax.fori_loop(0, probes, probe, jnp.zeros_like(theta)) / probes


def tie_complex_diagonal(diagonal_tree, get_pairs):
    """Match torch_laplace: sum real/imag curvature, use it for both components."""
    get_leaves = lambda tree: tuple(value for pair in get_pairs(tree) for value in pair)
    shared = tuple(value for real, imag in get_pairs(diagonal_tree) for value in (real + imag, real + imag))
    return eqx.tree_at(get_leaves, diagonal_tree, shared)


@eqx.filter_jit
def posterior_std(diagonal, prior_precision, scale=1.0, max_std=1e6, damping=1e-6):
    precision = jnp.maximum(diagonal, 0) + prior_precision + damping
    return jnp.minimum(scale / jnp.maximum(jnp.sqrt(precision), 1e-12), max_std)


def rel_l2(prediction, target):
    difference = (prediction - target).reshape(len(target), -1)
    denominator = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    return float(np.mean(np.linalg.norm(difference, axis=1) / np.maximum(denominator, 1e-12)))


def predictive_metrics(mean, variance, target, levels=(0.9, 0.95, 0.99)):
    """Moment-matched Gaussian metrics in original pressure units, including noise."""
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise FloatingPointError("nonfinite predictions or invalid predictive variance")
    std = np.sqrt(variance)
    nll = 0.5 * (np.log(2 * np.pi * variance) + (target - mean)**2 / variance)
    metrics = {"laplace_mnll": float(nll.mean()), "laplace_var_mean": float(variance.mean()),
               "laplace_std_mean": float(std.mean()), "laplace_rel2": rel_l2(mean, target)}
    for level in levels:
        radius = NormalDist().inv_cdf(0.5 + 0.5 * level) * std
        label = int(round(100 * level))
        metrics[f"coverage_{label}"] = float((np.abs(target - mean) <= radius).mean())
        metrics[f"interval_width_{label}"] = float((2 * radius).mean())
    return metrics
