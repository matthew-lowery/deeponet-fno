import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
from jax.flatten_util import ravel_pytree
import numpy as np
import optax

from car_data import load_car
from dse_fno import DSEFNO
from jax_laplace import (
    map_loss, hutchinson_hessian_diag, tie_complex_diagonal,
    posterior_std, predictive_metrics, rel_l2,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="DSE-FNO on DGPO cars with diagonal weight Laplace")
    p.add_argument("--data-path", type=Path, default=ROOT / "datasets/car.npz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/car_dse_fno")
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
    p.add_argument("--modes", type=int, default=8, help="m modes per axis, m^3 total")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
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


def batches(points, targets, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        selection = indices[start:start + batch_size]
        yield jnp.asarray(points[selection]), jnp.asarray(targets[selection])


def make_train_step(optimizer, total_scalars, prior_precision):
    @eqx.filter_jit
    def train_step(trainables, state, points, targets):
        def loss(trainables):
            model, log_noise = trainables
            return map_loss(jax.vmap(model)(points), targets, model, log_noise,
                            total_scalars, prior_precision)

        (value, auxiliary), gradients = eqx.filter_value_and_grad(loss, has_aux=True)(trainables)
        updates, state = optimizer.update(gradients, state, eqx.filter(trainables, eqx.is_inexact_array))
        return eqx.apply_updates(trainables, updates), state, jnp.stack((value, *auxiliary))

    return train_step


@eqx.filter_jit
def predict_batch(model, points):
    return jax.vmap(model)(points)


def predict(model, data, batch_size):
    outputs = [np.asarray(predict_batch(model, jnp.asarray(data.test_points[start:start + batch_size])))
               for start in range(0, len(data.test_points), batch_size)]
    return np.concatenate(outputs).astype(np.float64) * data.pressure_scale + data.pressure_mean


def laplace_eval(model, log_noise, data, args, key):
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    theta, unravel = ravel_pytree(parameters)
    likelihood_precision = jnp.exp(-2 * log_noise)

    def loss(weights, batch):
        current = eqx.combine(unravel(weights), static)
        points, targets = batch
        return 0.5 * jnp.sum((jax.vmap(current)(points) - targets)**2) * likelihood_precision

    order_key, hessian_key, sample_key = jr.split(key, 3)
    order = np.asarray(jr.permutation(order_key, len(data.train_points)))
    diagonal = jnp.zeros_like(theta)
    count = 0
    for index, batch in enumerate(batches(data.train_points, data.train_targets, order, args.batch_size)):
        if args.hessian_batches and index >= args.hessian_batches:
            break
        diagonal += hutchinson_hessian_diag(loss, theta, batch, jr.fold_in(hessian_key, index), args.hessian_probes)
        count += len(batch[0])
    diagonal *= len(data.train_points) / count
    if isinstance(model, DSEFNO):
        get_pairs = lambda tree: tuple((layer.real, layer.imag) for layer in tree.spectral)
        diagonal, _ = ravel_pytree(tie_complex_diagonal(unravel(diagonal), get_pairs))
    if not np.isfinite(np.asarray(diagonal)).all():
        raise FloatingPointError("nonfinite Hessian diagonal")
    std = posterior_std(diagonal, args.prior_precision, args.laplace_scale,
                        args.laplace_max_std, args.laplace_damping)
    mean = np.zeros_like(data.test_targets, dtype=np.float64)
    second_moment = np.zeros_like(mean)
    for sample in range(args.laplace_samples):
        weights = theta + std * jr.normal(jr.fold_in(sample_key, sample), theta.shape)
        prediction = predict(eqx.combine(unravel(weights), static), data, args.test_batch_size)
        delta = prediction - mean
        mean += delta / (sample + 1)
        second_moment += delta * (prediction - mean)
    observation_noise = float(jnp.exp(log_noise)) * data.pressure_scale
    variance = np.maximum(second_moment / args.laplace_samples, 0) + observation_noise**2 + args.laplace_noise**2
    metrics = predictive_metrics(mean, variance, data.test_targets, args.coverage_levels)
    metrics.update(hessian_cars=count, hessian_negative_fraction=float(jnp.mean(diagonal < 0)),
                   observation_noise=observation_noise)
    return metrics, mean, variance, diagonal


def main():
    args = parse_args()
    data = load_car(args.data_path, args.ntrain, args.ntest, args.num_points)
    directory = args.output_dir.expanduser() / f"seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    np.savez(directory / "normalization.npz", grid_min=data.grid_min, grid_scale=data.grid_scale,
             pressure_mean=data.pressure_mean, pressure_scale=data.pressure_scale,
             point_indices=data.point_indices)
    model_key, train_key, eval_key = jr.split(jr.PRNGKey(args.seed), 3)
    model = DSEFNO(args.modes, args.width, args.depth, args.proj_dim, key=model_key)
    trainables = (model, jnp.asarray(np.log(args.likelihood_noise), dtype=jnp.float32))
    num_batches = (args.ntrain + args.batch_size - 1) // args.batch_size
    schedule = optax.exponential_decay(args.lr, args.step_size * num_batches, args.gamma, staircase=True)
    optimizer = optax.adam(schedule)
    state = optimizer.init(eqx.filter(trainables, eqx.is_inexact_array))
    train_step = make_train_step(optimizer, data.train_targets.size, args.prior_precision)
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    param_mb = sum(x.size * x.dtype.itemsize for x in leaves) / 1024**2
    print(f"device={jax.devices()[0]} train={data.train_points.shape} test={data.test_points.shape} "
          f"parameters={sum(x.size for x in leaves)} param_mb={param_mb:.4f}", flush=True)
    with (directory / "metrics.jsonl").open("w") as handle:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            order = np.asarray(jr.permutation(jr.fold_in(train_key, epoch), args.ntrain))
            totals = jnp.zeros(3)
            for points, targets in batches(data.train_points, data.train_targets, order, args.batch_size):
                trainables, state, losses = train_step(trainables, state, points, targets)
                totals += len(points) * losses
            losses = np.asarray(totals / args.ntrain)
            if not np.isfinite(losses).all():
                raise FloatingPointError(f"nonfinite MAP loss at epoch {epoch}")
            train_seconds = time.perf_counter() - started
            print(f"epoch={epoch} map_loss={losses[0]:.6g} train_s={train_seconds:.2f}", flush=True)
            if epoch % args.eval_every and epoch != args.epochs:
                continue
            model, log_noise = trainables
            map_prediction = predict(model, data, args.test_batch_size)
            eval_started = time.perf_counter()
            metrics, mean, variance, diagonal = laplace_eval(model, log_noise, data, args, jr.fold_in(eval_key, epoch))
            metrics.update(epoch=epoch, seed=args.seed, test_rel2=rel_l2(map_prediction, data.test_targets),
                           train_map_loss=float(losses[0]), train_data_loss=float(losses[1]),
                           train_prior_loss=float(losses[2]), likelihood_noise=float(jnp.exp(log_noise)),
                           train_s=train_seconds, laplace_s=time.perf_counter() - eval_started, param_mb=param_mb)
            encoded = json.dumps(metrics, allow_nan=False)
            print(encoded, flush=True)
            handle.write(encoded + "\n")
            handle.flush()
            eqx.tree_serialise_leaves(directory / "model.eqx", trainables)
            np.savez(directory / "laplace.npz", hessian_diag=np.asarray(diagonal),
                     log_noise=np.asarray(log_noise), epoch=epoch)
            np.savez(directory / "predictions.npz", map=map_prediction, mean=mean, variance=variance,
                     target=data.test_targets)


if __name__ == "__main__":
    main()
