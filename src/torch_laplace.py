import contextlib
import math

import torch
import torch.nn.functional as F


def parameter_mb(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / 1024**2


def parameter_l2(model):
    total = None
    for p in model.parameters():
        if not p.requires_grad:
            continue
        value = p.abs().square().sum() if p.is_complex() else p.square().sum()
        total = value if total is None else total + value
    return total


def rel_l2(pred, target):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    return (torch.linalg.norm(pred - target, dim=1) / torch.linalg.norm(target, dim=1).clamp_min(1e-12)).mean()


def gaussian_mnll(pred_samples, target, noise=1e-6):
    mu = pred_samples.mean(dim=0)
    var = pred_samples.var(dim=0, unbiased=False) + noise**2
    nll = 0.5 * (torch.log(2 * math.pi * var) + (target - mu).square() / var)
    return nll.mean(), mu, var


def interval_coverage_metrics(pred_samples, target, noise=1e-6, levels=(0.9, 0.95, 0.99)):
    mu = pred_samples.mean(dim=0)
    std = (pred_samples.var(dim=0, unbiased=False) + noise**2).sqrt()
    normal = torch.distributions.Normal(torch.tensor(0.0, device=target.device), torch.tensor(1.0, device=target.device))
    metrics = {}
    for level in levels:
        z = normal.icdf(torch.tensor(0.5 + 0.5 * level, device=target.device))
        lo = mu - z * std
        hi = mu + z * std
        label = int(round(100 * level))
        metrics[f"coverage_{label}"] = float(((target >= lo) & (target <= hi)).float().mean())
        metrics[f"interval_width_{label}"] = float((hi - lo).mean())
    return metrics


def hutchinson_hessian_diag(model, loss_fn, loader, batches, probes, scale=1.0):
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    diag = [torch.zeros_like(p.real) if p.is_complex() else torch.zeros_like(p) for p in params]
    count = 0

    for batch_id, batch in enumerate(loader):
        if batches > 0 and batch_id >= batches:
            break
        loss = loss_fn(batch)
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        for _ in range(probes):
            vecs = [_rademacher_like(p) for p in params]
            dot = sum(_real_inner(g, v) for g, v in zip(grads, vecs))
            hvps = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
            for d, h, v, p in zip(diag, hvps, vecs, params):
                if h is not None:
                    d.add_(_diag_contribution(h.detach(), v, p))
            count += 1

    if count == 0:
        raise ValueError("no batches used for Hessian diagonal")
    for d in diag:
        d.mul_(scale / count)
    model.train(was_training)
    return diag


@contextlib.contextmanager
def sampled_weights(model, diag, prior_precision=1.0, scale=1.0, max_std=1.0, damping=0.0):
    params = [p for p in model.parameters() if p.requires_grad]
    base = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, b, d in zip(params, base, diag):
            precision = d.clamp_min(0.0) + prior_precision + damping
            std = scale / torch.sqrt(precision).clamp_min(1e-12)
            noise = _normal_like(p, std)
            p.copy_(b + noise * std.clamp_max(max_std))
    try:
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, base):
                p.copy_(b)


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def _rademacher_like(p):
    if p.is_complex():
        real = torch.empty_like(p.real).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        imag = torch.empty_like(p.real).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        return torch.complex(real, imag)
    return torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def _normal_like(p, std):
    if p.is_complex():
        return torch.complex(torch.randn_like(std), torch.randn_like(std))
    return torch.randn_like(p)


def _real_inner(a, b):
    prod = a.conj() * b if a.is_complex() else a * b
    return prod.real.sum()


def _diag_contribution(hvp, vec, param):
    prod = hvp.conj() * vec if param.is_complex() else hvp * vec
    return prod.real
