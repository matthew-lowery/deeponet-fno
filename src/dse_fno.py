"""3D point-cloud DSE-FNO; https://arxiv.org/html/2305.19663v4, (5), (8)."""

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr


def fourier_matrix(points, frequencies):
    """V[k,n] = sqrt(3/N) exp(-2 pi i k.x_n), for N three-dimensional points."""
    phase = frequencies @ points.T
    return jnp.sqrt(3.0 / points.shape[0]) * jnp.exp(-2j * jnp.pi * phase)


class SpectralConv(eqx.Module):
    real: jax.Array
    imag: jax.Array

    def __init__(self, width, num_modes, *, key):
        kr, ki = jr.split(key)
        shape = (num_modes, width, width)
        scale = 1 / (width * width * jnp.sqrt(2.0))
        self.real = scale * jr.normal(kr, shape)
        self.imag = scale * jr.normal(ki, shape)

    def __call__(self, values, transform):
        coefficients = transform @ values
        coefficients = jnp.einsum("ki,kio->ko", coefficients, self.real + 1j * self.imag)
        return (transform.conj().T @ coefficients).real


class DSEFNO(eqx.Module):
    frequencies: jax.Array
    lift: eqx.nn.Linear
    spectral: list[SpectralConv]
    pointwise: list[eqx.nn.Linear]
    project: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, modes=8, width=32, depth=4, proj_dim=128, *, key):
        if min(modes, width, depth, proj_dim) < 1:
            raise ValueError("modes, width, depth and proj_dim must be positive")
        axis = jnp.arange(modes)
        self.frequencies = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        keys = jr.split(key, 2 * depth + 3)
        self.lift = eqx.nn.Linear(3, width, key=keys[0])
        self.spectral = [SpectralConv(width, modes**3, key=k) for k in keys[1:depth + 1]]
        self.pointwise = [eqx.nn.Linear(width, width, key=k) for k in keys[depth + 1:2 * depth + 1]]
        self.project = eqx.nn.Linear(width, proj_dim, key=keys[-2])
        self.output = eqx.nn.Linear(proj_dim, 1, key=keys[-1])

    def __call__(self, points):
        """Map one (N,3) normalized car surface to (N,1) normalized pressure.

        Keep the paper's tensor modes {0,...,modes-1}^3 and adjoint
        synthesis, which is not an inverse on a general point cloud.
        All trainable leaves are real, including both parts of spectral weights.
        """
        transform = fourier_matrix(points, self.frequencies)
        values = jax.vmap(self.lift)(points)
        for i, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            values = spectral(values, transform) + jax.vmap(pointwise)(values)
            if i + 1 < len(self.spectral):
                values = jax.nn.gelu(values)
        values = jax.nn.gelu(jax.vmap(self.project)(values))
        return jax.vmap(self.output)(values)
