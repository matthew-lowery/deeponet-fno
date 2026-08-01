import argparse
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch_laplace import gaussian_mnll, hutchinson_hessian_diag, interval_coverage_metrics, parameter_l2, parameter_mb, rel_l2, sampled_weights


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)


def resample_1d(a, n):
    if n <= 0 or a.shape[1] == n:
        return a.astype(np.float32)
    old = np.linspace(0.0, 1.0, a.shape[1], dtype=np.float32)
    new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    flat = a.reshape(a.shape[0], a.shape[1], -1)
    out = np.empty((a.shape[0], n, flat.shape[-1]), dtype=np.float32)
    for i in range(flat.shape[0]):
        for j in range(flat.shape[-1]):
            out[i, :, j] = np.interp(new, old, flat[i, :, j])
    return out.reshape(a.shape[0], n, *a.shape[2:]).astype(np.float32)


def resample_2d(a, n):
    if a.shape[1] == n and a.shape[2] == n:
        return a.astype(np.float32)
    x = torch.as_tensor(a, dtype=torch.float32)
    had_channel = x.ndim == 4
    x = x.permute(0, 3, 1, 2) if had_channel else x[:, None]
    y = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=True)
    return (y.permute(0, 2, 3, 1) if had_channel else y[:, 0]).numpy().astype(np.float32)


def shuffle(xs, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xs[0]))
    return tuple(x[idx] for x in xs)


def load_dataset(name, data_dir, resolution, seed):
    data_dir = os.path.expanduser(data_dir)
    if name == "burgers":
        d = np.load(os.path.join(data_dir, "burgers.npz"))
        x, y = d["x"].astype(np.float32), d["y"].astype(np.float32)
        if resolution:
            x, y = resample_1d(x, resolution), resample_1d(y, resolution)
        grid = np.linspace(0, 1, x.shape[1], dtype=np.float32)[:, None]
        return x[:1000], y[:1000], x[-200:], y[-200:], grid, 1

    if name == "beijing":
        with open(os.path.join(data_dir, "beijing_data.pickle"), "rb") as handle:
            d = pickle.load(handle)
        x, y = d["x"][:6000].astype(np.float32), d["y"][:6000].astype(np.float32)
        x, y = shuffle((x, y), seed)
        if resolution:
            x, y = resample_1d(x, resolution), resample_1d(y, resolution)
        grid = np.linspace(0, 1, x.shape[1], dtype=np.float32)[:, None]
        return x[:5000], y[:5000], x[5000:], y[5000:], grid, 1

    if name == "darcy":
        root = os.path.join(data_dir, "darcy")
        xtr = np.load(os.path.join(root, "darcy_Xtr.npy")).reshape(-1, 29, 29)
        ytr = np.load(os.path.join(root, "darcy_ytr.npy")).reshape(-1, 29, 29)
        xte = np.load(os.path.join(root, "darcy_Xte.npy")).reshape(-1, 29, 29)
        yte = np.load(os.path.join(root, "darcy_yte.npy")).reshape(-1, 29, 29)
        x, y = shuffle((np.concatenate([xtr, xte]), np.concatenate([ytr, yte])), 1)
        n = resolution or 29
        x, y = resample_2d(x, n)[..., None], resample_2d(y, n)[..., None]
        grid_1d = np.linspace(0, 1, n, dtype=np.float32)
        grid = np.stack(np.meshgrid(grid_1d, grid_1d, indexing="xy"), axis=-1)
        return x[:1000], y[:1000], x[1000:1200], y[1000:1200], grid, 2

    raise ValueError("FNO runner only supports burgers, darcy, beijing")


def add_grid(x, grid):
    xt = torch.as_tensor(x, dtype=torch.float32)
    gt = torch.as_tensor(grid, dtype=torch.float32)
    if xt.ndim == 3:
        return torch.cat([xt, gt[None].repeat(xt.shape[0], 1, 1)], dim=-1)
    return torch.cat([xt, gt[None].repeat(xt.shape[0], 1, 1, 1)], dim=-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes, x_ft.shape[-1], self.weights.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.weights.shape[1], x.shape[-1] // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes])
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        mx = min(self.modes, x_ft.shape[-2], self.weights1.shape[-2])
        my = min(self.modes, x_ft.shape[-1], self.weights1.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.weights1.shape[1], x.shape[-2], x.shape[-1] // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mx, :my] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
        out_ft[:, :, -mx:, :my] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))


class FNO1d(nn.Module):
    def __init__(self, in_dim, modes, width, depth, proj_dim):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, width)
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, proj_dim)
        self.fc2 = nn.Linear(proj_dim, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 2, 1)
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x = conv(x) + w(x)
            if i + 1 < len(self.convs):
                x = F.relu(x)
        x = F.relu(self.fc1(x.permute(0, 2, 1)))
        return self.fc2(x)


class FNO2d(nn.Module):
    def __init__(self, in_dim, modes, width, depth, proj_dim):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_dim, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, proj_dim)
        self.fc2 = nn.Linear(proj_dim, 1)

    def forward(self, x):
        b, nx, ny = x.shape[:3]
        x = self.fc0(x).permute(0, 3, 1, 2)
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x = conv(x) + w(x.reshape(b, self.width, -1)).reshape(b, self.width, nx, ny)
            if i + 1 < len(self.convs):
                x = F.relu(x)
        x = F.relu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["burgers", "darcy", "beijing"], required=True)
    p.add_argument("--data-dir", default="/u/mlowery/dgpo/datasets")
    p.add_argument("--resolution", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--test-batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0, help="Deprecated; use --prior-precision for the MAP Gaussian prior.")
    p.add_argument("--step-size", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--laplace-samples", type=int, default=50)
    p.add_argument("--prior-precision", "--laplace-prior-precision", dest="prior_precision", type=float, default=100.0)
    p.add_argument("--laplace-scale", type=float, default=1.0)
    p.add_argument("--laplace-max-std", type=float, default=1e6)
    p.add_argument("--laplace-damping", type=float, default=1e-6)
    p.add_argument("--likelihood-noise", type=float, default=1.0, help="Initial scalar likelihood noise; trained as one log-noise parameter.")
    p.add_argument("--laplace-noise", type=float, default=1e-6)
    p.add_argument("--coverage-levels", type=float, nargs="*", default=[0.9, 0.95, 0.99])
    p.add_argument("--hessian-batches", type=int, default=10)
    p.add_argument("--hessian-probes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="laplace_fno")
    p.add_argument("--name", default="")
    return p.parse_args()


def evaluate(model, loader, y_normalizer, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = y_normalizer.decode(model(x.to(device))).detach().cpu()
            preds.append(out)
            targets.append(y)
    pred, target = torch.cat(preds), torch.cat(targets)
    return float(rel_l2(pred, target)), pred, target


def likelihood_noise(log_noise):
    return torch.exp(log_noise)


def map_loss(pred, target, model, log_noise, total_train_scalars, args):
    noise = likelihood_noise(log_noise)
    data_loss = 0.5 * F.mse_loss(pred, target) / noise.square() + log_noise
    prior_loss = 0.5 * args.prior_precision * parameter_l2(model) / total_train_scalars
    return data_loss + prior_loss, data_loss, prior_loss


def decoded_likelihood_noise(log_noise, y_normalizer):
    noise = likelihood_noise(log_noise).detach().cpu()
    return float(noise * y_normalizer.std.detach().cpu().mean())


def laplace_eval(model, log_noise, train_loader, test_loader, y_normalizer, device, args, num_train_batches):
    noise = likelihood_noise(log_noise).detach()

    def hessian_loss(batch):
        x, y = batch
        return 0.5 * F.mse_loss(model(x.to(device)), y.to(device), reduction="sum") / noise.square()

    diag = hutchinson_hessian_diag(model, hessian_loss, train_loader, args.hessian_batches, args.hessian_probes, scale=num_train_batches)
    samples, target = [], None
    for _ in range(args.laplace_samples):
        with sampled_weights(model, diag, args.prior_precision, args.laplace_scale, args.laplace_max_std, args.laplace_damping):
            _, pred, target = evaluate(model, test_loader, y_normalizer, device)
        samples.append(pred.reshape(-1))
    pred_samples = torch.stack(samples)
    target = target.reshape(-1)
    obs_noise = decoded_likelihood_noise(log_noise, y_normalizer)
    eval_noise = float(torch.sqrt(torch.tensor(obs_noise**2 + args.laplace_noise**2)))
    mnll, _, var = gaussian_mnll(pred_samples, target, eval_noise)
    coverage = interval_coverage_metrics(pred_samples, target, eval_noise, args.coverage_levels)
    return float(mnll), float(var.mean()), float(var.sqrt().mean()), coverage


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.wandb:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.login(key="d612cda26a5690e196d092756d668fc2aee8525b")
    wandb.init(project=args.wandb_project, name=args.name or f"fno_{args.dataset}_seed{args.seed}", config=vars(args), mode=None if args.wandb else "disabled")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    xtr, ytr, xte, yte, grid, dim = load_dataset(args.dataset, args.data_dir, args.resolution, args.seed)
    x_norm = UnitGaussianNormalizer(torch.as_tensor(xtr, dtype=torch.float32))
    y_norm = UnitGaussianNormalizer(torch.as_tensor(ytr, dtype=torch.float32))
    xtr_aug = add_grid(x_norm.encode(torch.as_tensor(xtr, dtype=torch.float32)).numpy(), grid)
    xte_aug = add_grid(x_norm.encode(torch.as_tensor(xte, dtype=torch.float32)).numpy(), grid)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
    yte_t = torch.as_tensor(yte, dtype=torch.float32)
    ytr_encoded = y_norm.encode(ytr_t)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr_aug, ytr_encoded), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xte_aug, yte_t), batch_size=args.test_batch_size, shuffle=False)
    total_train_scalars = ytr_encoded.numel()
    num_train_batches = len(train_loader)
    model_cls = FNO1d if dim == 1 else FNO2d
    model = model_cls(xtr_aug.shape[-1], args.modes, args.width, args.depth, args.proj_dim).to(device)
    log_noise = torch.nn.Parameter(torch.tensor(np.log(args.likelihood_noise), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_noise], lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    mb = parameter_mb(model)
    print(f"param_mb: {mb:.4f}")
    wandb.log({"param_mb": mb}, step=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        losses, data_losses, prior_losses = [], [], []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss, data_loss, prior_loss = map_loss(model(x.to(device)), y.to(device), model, log_noise, total_train_scalars, args)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            data_losses.append(float(data_loss.detach().cpu()))
            prior_losses.append(float(prior_loss.detach().cpu()))
        scheduler.step()
        train_s = time.perf_counter() - t0

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_rel2, _, _ = evaluate(model, test_loader, y_norm, device)
            mnll, var_mean, std_mean, coverage = laplace_eval(model, log_noise, train_loader, test_loader, y_norm, device, args, num_train_batches)
            noise_value = float(likelihood_noise(log_noise).detach().cpu())
            log = {"epoch": epoch, "train_map_loss": float(np.mean(losses)), "train_data_loss": float(np.mean(data_losses)), "train_prior_loss": float(np.mean(prior_losses)), "likelihood_noise": noise_value, "log_likelihood_noise": float(log_noise.detach().cpu()), "test_rel2": test_rel2, "laplace_mnll": mnll, "laplace_var_mean": var_mean, "laplace_std_mean": std_mean, "train_s": train_s, "param_mb": mb}
            log.update(coverage)
            print(log)
            wandb.log(log, step=epoch)


if __name__ == "__main__":
    main()
