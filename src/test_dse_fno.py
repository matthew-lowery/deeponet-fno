import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from car_data import load_car
from dse_fno import DSEFNO, SpectralConv3d_dse, VFT3D
from run import laplace_eval, map_loss
from torch_laplace import hutchinson_hessian_diag, sampled_weights
from vendor.dse.elasticity_fno import FNO_dse, VFT


def regular_points(n):
    axis = torch.arange(n) / n
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(1, -1, 3)


class DSETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)

    def test_upstream_source_and_forward_are_preserved(self):
        root = Path(__file__).resolve().parent / 'vendor/dse'
        source = (root / 'elasticity_fno.py').read_text()
        original = source.replace(
            '    def make_transform(self, x):\n        return VFT(x[:,:,0], x[:,:,1], self.modes1)\n\n', '')
        original = original.replace('transform = self.make_transform(x)',
                                    'transform = VFT(x[:,:,0], x[:,:,1], self.modes1)')
        provenance = json.loads((root / 'source.json').read_text())
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), provenance['original_sha256'])
        self.assertIs(DSEFNO.forward, FNO_dse.forward)
        self.assertIs(VFT3D.forward, VFT.forward)
        self.assertIs(VFT3D.inverse, VFT.inverse)

    def test_spectral_layer_matches_independent_fft(self):
        n, m, channels = 7, 2, 2
        points = regular_points(n)
        values = torch.randn(1, channels, n**3)
        layer = SpectralConv3d_dse(channels, channels, m)
        actual = layer(values, VFT3D(points, m))
        coefficients = torch.fft.rfftn(values.reshape(1, channels, n, n, n), dim=(-3, -2, -1))
        output = torch.zeros_like(coefficients)
        blocks = ((slice(0, m), slice(0, m)), (slice(-m, None), slice(0, m)),
                  (slice(0, m), slice(-m, None)), (slice(-m, None), slice(-m, None)))
        for i, (sx, sy) in enumerate(blocks, 1):
            output[:, :, sx, sy, :m] = torch.einsum(
                'bixyz,ioxyz->boxyz', coefficients[:, :, sx, sy, :m], getattr(layer, f'weights{i}'))
        expected = 3 * torch.fft.irfftn(output, s=(n, n, n), dim=(-3, -2, -1)).reshape_as(actual)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        actual.square().mean().backward()
        actual_gradients = [p.grad.clone() for p in layer.parameters()]
        layer.zero_grad()
        expected.square().mean().backward()
        for gradient, parameter in zip(actual_gradients, layer.parameters()):
            torch.testing.assert_close(gradient, parameter.grad, rtol=5e-5, atol=3e-6)

    def test_mixed_sign_frequencies_are_retained(self):
        points = regular_points(8)
        layer = SpectralConv3d_dse(1, 1, 2)
        with torch.no_grad():
            for weight in layer.parameters():
                weight.fill_(1)
        for frequency in ([1, -1, 0], [-1, 1, 1], [1, -1, -1]):
            field = torch.cos(2 * torch.pi * (points @ torch.tensor(frequency, dtype=torch.float32)))[:, None, :]
            actual = layer(field, VFT3D(points, 2))
            torch.testing.assert_close(actual, 3 * field, rtol=3e-5, atol=1e-5)

    def test_irregular_synthesis_uses_conjugate_frequency_pairs(self):
        points = torch.rand(1, 11, 3)
        transform = VFT3D(points, 2)
        coefficients = torch.randn(1, 32, 2, dtype=torch.cfloat)
        actual = transform.inverse(coefficients * transform.multiplicity[None, :, None]).real
        expected = torch.zeros(11, 2, dtype=torch.cfloat)
        for index, frequency in enumerate(transform.frequencies):
            phase = 2 * torch.pi * (points[0] @ frequency.float())
            term = torch.exp(1j * phase)[:, None] * coefficients[0, index]
            expected += term
            if frequency[2] > 0:
                expected += term.conj()
        expected *= (3 / 11)**0.5
        torch.testing.assert_close(actual[0], expected.real, rtol=2e-5, atol=3e-6)

    def test_point_order_and_batch_independence(self):
        model = DSEFNO(2, 3, proj_dim=4)
        points = torch.rand(2, 7, 3)
        original = points.clone()
        order = torch.tensor([3, 2, 0, 6, 4, 1, 5])
        torch.testing.assert_close(model((None, points[:, order])), model((None, points))[:, order])
        torch.testing.assert_close(model((None, points))[0], model((None, points[:1]))[0])
        torch.testing.assert_close(points, original)

    def test_map_and_laplace_use_existing_torch_helpers(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3)
        total, data, prior = map_loss(torch.tensor([2.0]), torch.tensor([0.0]), model,
                                      torch.tensor(np.log(2.0)), 10, 4.0)
        self.assertAlmostEqual(float(data), 0.5 + np.log(2), places=6)
        self.assertAlmostEqual(float(prior.detach()), 1.8, places=6)
        loss = lambda _: model.weight.square().sum()
        diagonal = hutchinson_hessian_diag(model, loss, [None], 1, 2)
        torch.testing.assert_close(diagonal[0], torch.full_like(model.weight, 2))
        original = model.weight.detach().clone()
        with sampled_weights(model, diagonal, prior_precision=4):
            self.assertFalse(torch.equal(model.weight, original))
        torch.testing.assert_close(model.weight, original)

    def test_laplace_full_data_scaling_with_short_final_batch(self):
        class ConstantModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.4))

            def forward(self, inputs):
                return torch.ones_like(inputs[1][..., :1]) * self.weight

        model = ConstantModel()
        train = DataLoader(TensorDataset(torch.zeros(5, 3, 3), torch.zeros(5, 3, 1)), batch_size=2)
        test = DataLoader(TensorDataset(torch.zeros(2, 3, 3), torch.zeros(2, 3, 1)), batch_size=1)
        data = SimpleNamespace(test_targets=np.zeros((2, 3, 1), np.float32), pressure_mean=10.0, pressure_scale=2.0)
        args = SimpleNamespace(hessian_batches=0, hessian_probes=2, prior_precision=4.0,
                               laplace_scale=0.0, laplace_max_std=1e6, laplace_damping=1e-6,
                               laplace_samples=3, laplace_noise=0.5, coverage_levels=[0.95])
        diagonals = []

        def estimate(*values, **kwargs):
            result = hutchinson_hessian_diag(*values, **kwargs)
            diagonals.append(result[0].clone())
            return result

        with patch('run.hutchinson_hessian_diag', side_effect=estimate):
            metrics = laplace_eval(model, torch.tensor(np.log(0.5)), train, test, data, torch.device('cpu'), args)
            args.hessian_batches = 1
            partial = laplace_eval(model, torch.tensor(np.log(0.5)), train, test, data, torch.device('cpu'), args)
        self.assertEqual(metrics['hessian_cars'], 5)
        self.assertAlmostEqual(metrics['laplace_var_mean'], 1.25)
        self.assertAlmostEqual(metrics['laplace_mnll'], partial['laplace_mnll'])
        for diagonal in diagonals:
            self.assertAlmostEqual(float(diagonal), 5 * 3 / 0.5**2)

    def test_data_split_and_training_only_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'car.npz'
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


if __name__ == '__main__':
    unittest.main()
