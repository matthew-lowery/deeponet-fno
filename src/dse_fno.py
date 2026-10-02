"""Car adapter for the authors' PyTorch DSE-FNO; provenance in vendor/dse/source.json.

The inherited FNO forward pass keeps their four spectral/pointwise layers and GELU
projection. Only the coordinate lift, transform, and spectral blocks become 3D.
"""

import math

import torch
from torch import nn

from vendor.dse.elasticity_fno import FNO_dse, VFT


class VFT3D(VFT):
    def __init__(self, points, modes):
        """points: (batch,N,3), using fixed training-set coordinate normalization.

        Retain kx,ky in {0,...,m-1,-m,...,-1}, kz in {0,...,m-1}.
        Each transform has the paper's sqrt(3/N) normalization. Positive kz
        coefficients contribute twice their real synthesis, accounting for -k.
        """
        self.number_points = points.shape[1]
        self.batch_size = points.shape[0]
        self.modes = modes
        positive = torch.arange(modes, device=points.device)
        signed = torch.cat((positive, torch.arange(-modes, 0, device=points.device)))
        self.frequencies = torch.stack(torch.meshgrid(signed, signed, positive, indexing="ij"), -1).reshape(-1, 3)
        phase = torch.einsum("kd,bnd->bkn", self.frequencies.to(points.dtype), points)
        self.V_fwd = torch.exp(-2j * torch.pi * phase) * math.sqrt(3 / self.number_points)
        self.V_inv = self.V_fwd.conj().transpose(1, 2)
        self.multiplicity = torch.where(self.frequencies[:, 2] == 0, 1, 2)


class SpectralConv3d_dse(nn.Module):
    """The upstream two signed 2D blocks extended to four signed 3D blocks."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.out_channels = out_channels
        self.scale = 1 / (in_channels * out_channels)
        for i in range(1, 5):
            weight = self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat)
            setattr(self, f"weights{i}", nn.Parameter(weight))

    def forward(self, x, transformer):
        batch = x.shape[0]
        m = self.modes
        coefficients = transformer.forward(x.permute(0, 2, 1).to(transformer.V_fwd.dtype))
        coefficients = coefficients.permute(0, 2, 1).reshape(batch, x.shape[1], 2 * m, 2 * m, m)
        output = coefficients.new_zeros(batch, self.out_channels, 2 * m, 2 * m, m)
        blocks = ((slice(0, m), slice(0, m)), (slice(-m, None), slice(0, m)),
                  (slice(0, m), slice(-m, None)), (slice(-m, None), slice(-m, None)))
        for i, (sx, sy) in enumerate(blocks, 1):
            output[:, :, sx, sy, :] = torch.einsum(
                "bixyz,ioxyz->boxyz", coefficients[:, :, sx, sy, :], getattr(self, f"weights{i}"))
        output = output.reshape(batch, self.out_channels, -1).permute(0, 2, 1)
        output = output * transformer.multiplicity[None, :, None]
        return transformer.inverse(output).permute(0, 2, 1).real


class DSEFNO(FNO_dse):
    def __init__(self, modes=8, width=32, depth=4, proj_dim=128):
        if depth != 4:
            raise ValueError("The upstream FNO uses exactly four spectral layers")
        if min(modes, width, proj_dim) < 1:
            raise ValueError("modes, width and proj_dim must be positive")
        super().__init__({"modes1": modes, "modes2": modes, "width": width, "denormalizer": nn.Identity()})
        self.fc0 = nn.Linear(3, width)
        for i in range(4):
            setattr(self, f"conv{i}", SpectralConv3d_dse(width, width, modes))
        if proj_dim != 128:
            self.fc1 = nn.Linear(width, proj_dim)
            self.fc2 = nn.Linear(proj_dim, 1)

    def make_transform(self, points):
        return VFT3D(points, self.modes1)
