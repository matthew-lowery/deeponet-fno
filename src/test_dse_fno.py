import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
import numpy as np

from car_data import load_car
from dse_fno import DSEFNO, SpectralConv, fourier_matrix
from jax_laplace import (
    hutchinson_hessian_diag, map_loss, posterior_std, predictive_metrics, tie_complex_diagonal,
)
from run import laplace_eval


class DSETests(unittest.TestCase):
    def test_transform_on_irregular_points(self):
        points = jnp.asarray([[0.1, 0.3, 0.5], [0.9, 0.2, 0.6], [0.3, 0.4, 0.8], [0.6, 0.1, 0.2]])
        frequencies = jnp.asarray([[0, 0, 0], [1, 0, 1], [0, 1, 1]])
        values = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 10
        layer = SpectralConv(2, 3, key=jr.PRNGKey(0))
        weights = np.asarray(layer.real + 1j * layer.imag)
        expected = np.zeros((4, 2), dtype=complex)
        for k, frequency in enumerate(np.asarray(frequencies)):
            forward = np.exp(-2j * np.pi * (np.asarray(points) @ frequency))
            coefficients = sum(forward[n] * np.asarray(values[n]) for n in range(4))
            expected += (3 / 4) * forward.conj()[:, None] * (coefficients @ weights[k])
        actual = layer(values, fourier_matrix(points, frequencies))
        np.testing.assert_allclose(actual, expected.real, rtol=2e-5, atol=2e-6)

    def test_point_order_and_batch_independence(self):
        model = DSEFNO(2, 3, 2, 4, key=jr.PRNGKey(1))
        points = jr.uniform(jr.PRNGKey(2), (7, 3))
        order = jnp.asarray([3, 2, 0, 6, 4, 1, 5])
        np.testing.assert_allclose(model(points[order]), model(points)[order], atol=1e-6)
        batched = jax.vmap(model)(jnp.stack((points, points / 2)))
        np.testing.assert_allclose(batched[0], model(points), atol=1e-6)

    def test_diagonal_curvature_and_prior_scaling(self):
        curvature = jnp.asarray([2.0, 5.0, 9.0])
        loss = lambda theta, batch: 0.5 * jnp.sum(batch * theta**2)
        diagonal = hutchinson_hessian_diag(loss, jnp.ones(3), curvature, jr.PRNGKey(3), 4)
        np.testing.assert_allclose(diagonal, curvature, rtol=1e-6)
        std = posterior_std(jnp.asarray([-2.0, 3.0]), 4.0, 2.0, 0.9, 1.0)
        np.testing.assert_allclose(std, np.minimum(2 / np.sqrt([5.0, 8.0]), 0.9), rtol=1e-6)
        total, (data, prior) = map_loss(jnp.array([2.0]), jnp.array([0.0]), jnp.array([3.0]),
                                      jnp.log(2.0), 10, 4.0)
        self.assertAlmostEqual(float(data), 0.5 + np.log(2), places=6)
        self.assertAlmostEqual(float(prior), 1.8, places=6)
        self.assertAlmostEqual(float(total), float(data + prior), places=6)

    def test_complex_curvature_matches_torch_convention(self):
        real = jnp.asarray([-3.0, 4.0])
        imag = jnp.asarray([2.0, 5.0])
        diagonal = {"real": real, "imag": imag, "bias": jnp.asarray([7.0])}
        tied = tie_complex_diagonal(diagonal, lambda tree: ((tree["real"], tree["imag"]),))
        np.testing.assert_array_equal(tied["real"], [-1.0, 9.0])
        np.testing.assert_array_equal(tied["imag"], [-1.0, 9.0])
        np.testing.assert_array_equal(tied["bias"], [7.0])

    def test_data_split_and_training_only_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "car.npz"
            points = np.arange(72, dtype=np.float32).reshape(6, 4, 3)
            pressure = np.arange(24, dtype=np.float32).reshape(6, 4)
            np.savez(path, grid=points, press=pressure)
            data = load_car(path, 3, 2)
            np.testing.assert_array_equal(data.test_targets[..., 0], pressure[-2:])
            np.testing.assert_array_equal(data.grid_min, points[:3].min(axis=(0, 1)))
            self.assertAlmostEqual(float(data.train_targets.mean()), 0.0, places=6)
            self.assertAlmostEqual(data.pressure_mean, pressure[:3].mean(), places=6)
            self.assertTrue(np.any(data.test_points > 1))
            with self.assertRaises(ValueError):
                load_car(path, 5, 2)

    def test_laplace_full_data_scaling_with_short_final_batch(self):
        class ConstantModel(eqx.Module):
            weight: jax.Array

            def __call__(self, points):
                return jnp.ones((len(points), 1)) * self.weight

        model = ConstantModel(jnp.asarray(0.4))
        data = SimpleNamespace(train_points=np.zeros((5, 3, 3), np.float32),
                               train_targets=np.zeros((5, 3, 1), np.float32),
                               test_points=np.zeros((2, 3, 3), np.float32),
                               test_targets=np.zeros((2, 3, 1), np.float32),
                               pressure_mean=10.0, pressure_scale=2.0)
        args = SimpleNamespace(batch_size=2, hessian_batches=0, hessian_probes=2,
                               prior_precision=4.0, laplace_scale=0.0, laplace_max_std=1e6,
                               laplace_damping=1e-6, laplace_samples=3, test_batch_size=1,
                               laplace_noise=0.5, coverage_levels=[0.95])
        metrics, mean, variance, diagonal = laplace_eval(model, jnp.log(0.5), data, args, jr.PRNGKey(0))
        np.testing.assert_allclose(diagonal, 5 * 3 / 0.5**2, rtol=1e-6)
        np.testing.assert_allclose(mean, 10.8, rtol=1e-6)
        np.testing.assert_allclose(variance, (2 * 0.5)**2 + 0.5**2, rtol=1e-6)
        self.assertEqual(metrics["hessian_cars"], 5)
        args.hessian_batches = 1
        _, _, _, partial = laplace_eval(model, jnp.log(0.5), data, args, jr.PRNGKey(0))
        np.testing.assert_allclose(partial, diagonal, rtol=1e-6)

    def test_gaussian_metrics(self):
        metrics = predictive_metrics(np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3)))
        self.assertAlmostEqual(metrics["laplace_mnll"], 0.5 * np.log(2 * np.pi))
        self.assertEqual(metrics["coverage_95"], 1.0)
        self.assertAlmostEqual(metrics["interval_width_95"], 3.919927969080108, places=6)


if __name__ == "__main__":
    unittest.main()
