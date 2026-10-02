import argparse
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb

from car_data import load_car
from dse_fno import DSEFNO
from torch_laplace import (
    parameter_l2, parameter_mb, hutchinson_hessian_diag, sampled_weights,
    gaussian_mnll, interval_coverage_metrics, rel_l2,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="DSE-FNO on DGPO cars with diagonal weight Laplace")
    p.add_argument("--data-path", type=Path, default=ROOT / "datasets/car.npz")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    p.add_argument("--wandb-project", default="laplace_fno")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--name", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ntrain", type=int, default=500)
    p.add_argument("--ntest", type=int, default=111)
    p.add_argument("--num-points", type=int, default=0, help="0 uses all surface points")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--test-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--step-size", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--modes", type=int, default=8, help="four signed blocks of m^3 coefficients")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, choices=[4], default=4)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--prior-precision", type=float, default=100.0)
    p.add_argument("--likelihood-noise", type=float, default=1.0, help="initial normalized noise std")
    p.add_argument("--laplace-samples", type=int, default=50)
    p.add_argument("--hessian-batches", type=int, default=10, help="0 uses all training batches")
    p.add_argument("--hessian-probes", type=int, default=4)
    p.add_argument("--laplace-scale", type=float, default=1.0)
    p.add_argument("--laplace-max-std", type=float, default=1e6)
    p.add_argument("--laplace-damping", type=float, default=1e-6)
    p.add_argument("--laplace-noise", type=float, default=1e-6, help="extra noise std in pressure units")
    p.add_argument("--coverage-levels", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    args = p.parse_args()
    positive = ("ntrain", "ntest", "epochs", "eval_every", "batch_size", "test_batch_size", "lr",
                "step_size", "gamma", "modes", "width", "depth", "proj_dim", "likelihood_noise",
                "hessian_probes", "laplace_max_std")
    for name in positive:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("prior_precision", "laplace_scale", "laplace_damping", "laplace_noise",
                 "hessian_batches", "num_points", "seed"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    if args.laplace_samples < 2:
        p.error("--laplace-samples must be at least 2")
    if args.prior_precision + args.laplace_damping <= 0:
        p.error("prior precision plus damping must be positive")
    if not all(0 < level < 1 for level in args.coverage_levels):
        p.error("coverage levels must be strictly between 0 and 1")
    return args


def map_loss(prediction, target, model, log_noise, total_scalars, prior_precision):
    data_loss = 0.5 * F.mse_loss(prediction, target) * torch.exp(-2 * log_noise) + log_noise
    prior_loss = 0.5 * prior_precision * parameter_l2(model) / total_scalars
    return data_loss + prior_loss, data_loss, prior_loss


@torch.no_grad()
def predict(model, loader, data, device):
    model.eval()
    outputs = [model((None, points.to(device))).cpu() for points, _ in loader]
    return torch.cat(outputs).double() * data.pressure_scale + data.pressure_mean


def laplace_eval(model, log_noise, train_loader, test_loader, data, device, args):
    noise = log_noise.detach().exp()
    selected = list(islice(train_loader, args.hessian_batches or len(train_loader)))
    count = sum(len(points) for points, _ in selected)

    def loss(batch):
        points, targets = (value.to(device) for value in batch)
        return 0.5 * F.mse_loss(model((None, points)), targets, reduction="sum") / noise.square()

    scale = len(train_loader.dataset) * len(selected) / count
    diagonal = hutchinson_hessian_diag(model, loss, selected, 0, args.hessian_probes, scale=scale)
    if not all(torch.isfinite(value).all() for value in diagonal):
        raise FloatingPointError("nonfinite Hessian diagonal")
    predictions = []
    for _ in range(args.laplace_samples):
        with sampled_weights(model, diagonal, args.prior_precision, args.laplace_scale,
                             args.laplace_max_std, args.laplace_damping):
            predictions.append(predict(model, test_loader, data, device))
    samples = torch.stack(predictions)
    target = torch.from_numpy(data.test_targets).double()
    observation_noise = float(noise) * data.pressure_scale
    total_noise = (observation_noise**2 + args.laplace_noise**2)**0.5
    mnll, mean, variance = gaussian_mnll(samples, target, total_noise)
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or not torch.isfinite(mnll):
        raise FloatingPointError("nonfinite Laplace predictions")
    metrics = {"laplace_mnll": float(mnll), "laplace_rel2": float(rel_l2(mean, target)),
               "laplace_var_mean": float(variance.mean()), "laplace_std_mean": float(variance.sqrt().mean()),
               "observation_noise": observation_noise, "hessian_cars": count,
               "hessian_negative_fraction": sum(int((value < 0).sum()) for value in diagonal)
               / sum(value.numel() for value in diagonal)}
    metrics.update(interval_coverage_metrics(samples, target, total_noise, args.coverage_levels))
    return metrics


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda: the spectral layers require complex tensors")
    data = load_car(args.data_path, args.ntrain, args.ntest, args.num_points)
    train_set = TensorDataset(torch.from_numpy(data.train_points), torch.from_numpy(data.train_targets))
    test_set = TensorDataset(torch.from_numpy(data.test_points), torch.from_numpy(data.test_targets))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
    hessian_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(args.seed + 1))
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    model = DSEFNO(args.modes, args.width, args.depth, args.proj_dim).to(device)
    log_noise = nn.Parameter(torch.tensor(np.log(args.likelihood_noise), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([*model.parameters(), log_noise], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["upstream_commit"] = json.loads((ROOT / "vendor/dse/source.json").read_text())["commit"]
    config.update(pressure_mean=data.pressure_mean, pressure_scale=data.pressure_scale,
                  grid_min=data.grid_min.tolist(), grid_scale=data.grid_scale.tolist())
    mb = parameter_mb(model)
    print(f"device={device} train={data.train_points.shape} test={data.test_points.shape} param_mb={mb:.4f}", flush=True)
    with wandb.init(project=args.wandb_project, name=args.name or f"car_dse_fno_seed{args.seed}",
                    mode=args.wandb_mode, config=config, settings=wandb.Settings(disable_git=True)) as run:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            totals = np.zeros(3)
            for points, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                losses = map_loss(model((None, points.to(device))), targets.to(device), model,
                                  log_noise, data.train_targets.size, args.prior_precision)
                losses[0].backward()
                optimizer.step()
                totals += len(points) * np.array([float(value.detach()) for value in losses])
            scheduler.step()
            losses = totals / args.ntrain
            if not np.isfinite(losses).all():
                raise FloatingPointError(f"nonfinite MAP loss at epoch {epoch}")
            train_seconds = time.perf_counter() - started
            metrics = dict(epoch=epoch, train_map_loss=float(losses[0]), train_data_loss=float(losses[1]),
                           train_prior_loss=float(losses[2]), likelihood_noise=float(log_noise.detach().exp()),
                           train_s=train_seconds, param_mb=mb)
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                prediction = predict(model, test_loader, data, device)
                metrics["test_rel2"] = float(rel_l2(prediction, torch.from_numpy(data.test_targets)))
                eval_started = time.perf_counter()
                metrics.update(laplace_eval(model, log_noise, hessian_loader, test_loader, data, device, args))
                metrics["laplace_s"] = time.perf_counter() - eval_started
            print(json.dumps(metrics, allow_nan=False), flush=True)
            run.log(metrics, step=epoch)


if __name__ == "__main__":
    main()
