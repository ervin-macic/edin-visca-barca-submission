"""Train the two neural networks on the converted training matches (cache/ext, other matches only).

    python -m training.train_networks --data-dir path/to/data [--out models]

  mlp48.pt   frame-wise network: 16 epochs, then 32 more at a lower learning rate
  tcn24.pt   temporal convolutional network: 12 epochs, then 12 more

Every training match is passed twice through the tracker simulation (augment.py) followed by the
fused-ball simulation (fill.py), each copy in both team orders. --data-dir is only read to measure
the fill shares of the fused ball (fill.measure_profile, no labels involved). Training data and
checkpoints go to cache/networks/.
"""

import argparse
import gc
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from possession import features
from possession.networks import FrameNet, TemporalNet, normalise

from . import augment
from .common import (CACHE_DIR, MODEL_DIR, X, change_targets, external_matches, load_match,
                     period_runs, swap_teams)
from .fill import make_degrader, measure_profile

DATA_SEED = 2027
SEED = 7105
BASE_EPOCHS = dict(mlp=16, tcn=12)
EXTRA_EPOCHS = dict(mlp=32, tcn=12)
OUTPUT = dict(mlp="mlp48.pt", tcn="tcn24.pt")
# The temporal network reads these 80 named columns (plus their missing-value indicators).
TEMPORAL_NAMES = (
    'seen', 'ball_x', 'ball_absy', 'ball_vx', 'ball_speed', 'ball_accel',
    'ball_margin', 'ball_margin_x', 'ball_margin_y', 'd1_h', 'd1_a', 'd2_h', 'd2_a',
    'd1_diff', 'n3_h', 'n10_h', 'n3_a', 'n10_a', 'rel_speed_h', 'closing_h',
    'rel_speed_a', 'closing_a', 'rel_speed_diff', 'turn_12', 'speed_change_12',
    'turn_25', 'speed_change_25', 'since_kick_h', 'until_kick_h', 'since_kick_a',
    'until_kick_a', 'since_contact_h', 'until_contact_h', 'since_contact_a',
    'until_contact_a', 'since_seen_long', 'until_seen_long', 'last_margin',
    'next_margin', 'last_speed', 'next_speed', 'last_d1_diff', 'next_d1_diff',
    'gap_length', 'gap_position', 'speed_h_1s', 'speed_a_1s', 'speed_all_1s',
    'static_all_1s', 'speed_h_4s', 'speed_a_4s', 'speed_all_4s', 'static_all_4s',
    'cx_h', 'cx_a', 'cy_abs_h', 'cy_abs_a', 'sx_h', 'sx_a', 'sy_h', 'sy_a',
    'back_h', 'back_a', 'front_h', 'front_a', 'cx_gap', 'cx_mid',
    'cvx_h_1s', 'cvx_a_1s', 'player_out_y', 'player_out_x', 'raw_seen',
    'ctx_mean_deceleration_h', 'ctx_mean_deceleration_a', 'ctx_braking_share_h',
    'ctx_braking_share_a', 'ctx_static_share_change_h', 'ctx_static_share_change_a',
    'ctx_motion_valid_share_h', 'ctx_motion_valid_share_a')


def sample_rows(match, rng, stride=10, edge_stride=2, edge=15):
    """Every 10th frame, and every 2nd frame within 15 frames of a label change."""
    truth = match['label']
    offset = int(rng.integers(0, stride))
    take = np.zeros(len(truth), bool)
    take[offset::stride] = True
    change = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    near = np.zeros(len(truth) + 1, np.int32)
    np.add.at(near, np.maximum(change - edge, 0), 1)
    np.add.at(near, np.minimum(change + edge, len(truth)), -1)
    near = near[:-1].cumsum() > 0
    dense = np.zeros(len(truth), bool)
    dense[offset % edge_stride::edge_stride] = True
    take |= near & dense
    return np.flatnonzero(take & (match['period'] > 0) & (truth != X))


def prepare(folder, paths, profile, draws=2):
    """Write the training rows (frame-wise network) and half-long sequences at 12.5 Hz (temporal
    network) to `folder`, plus the training mean and scale of every column."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DATA_SEED)
    degrader = make_degrader(profile)
    shards, sequences, columns = [], [], None
    for name, path in paths.items():
        print('preparing', name, flush=True)
        match = load_match(path)
        for draw in range(draws):
            settings = augment.random_settings(rng)
            for swap in (False, True):
                copy = degrader(match, rng, **settings)
                if swap:
                    copy = swap_teams(copy)
                table, columns = features.compute(copy, deceleration=True)
                index = sample_rows(copy, rng)
                q = change_targets(copy['label'], copy['period'])
                stem = f'{name}_d{draw}_s{int(swap)}'
                np.save(folder / (stem + '_x.npy'), table[index])
                np.save(folder / (stem + '_y.npy'),
                        np.column_stack((copy['label'][index], q[index])).astype(np.int64))
                shards.append(stem)
                selected = [columns.index(n) for n in TEMPORAL_NAMES]
                for half, (a, b, _) in enumerate(period_runs(copy['period'])):
                    grid = np.arange(a, b, 2)
                    sequence = stem + f'_half{half}'
                    np.save(folder / (sequence + '_seq.npy'), table[grid][:, selected])
                    np.save(folder / (sequence + '_target.npy'),
                            np.column_stack((copy['label'][grid], q[grid])).astype(np.int64))
                    sequences.append(sequence)
                del table, copy
                gc.collect()
        del match
        gc.collect()
    count = np.zeros(len(columns), np.float64)
    total = count.copy()
    square = count.copy()
    for stem in shards:
        x = np.load(folder / (stem + '_x.npy'), mmap_mode='r')
        for a in range(0, len(x), 8192):
            batch = np.asarray(x[a:a + 8192], dtype=np.float64)
            finite = np.isfinite(batch)
            batch = np.where(finite, batch, 0)
            count += finite.sum(axis=0)
            total += batch.sum(axis=0)
            square += (batch ** 2).sum(axis=0)
        del x
    mean = total / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean ** 2, 1e-6))
    np.savez(folder / 'scaler.npz', mean=mean.astype(np.float32), scale=scale.astype(np.float32))
    return dict(training_matches=list(paths), columns=columns, shards=shards, sequences=sequences)


def fit(data, folder, info, kind, epochs):
    """Train (or, when folder holds a checkpoint, continue training) one network for `epochs`
    epochs. AdamW, cosine learning-rate schedule, state loss + 0.3 x change loss."""
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (kind + '.pt')
    if destination.exists():
        return
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    with np.load(data / 'scaler.npz') as scaler:
        mean, scale = scaler['mean'], scaler['scale']
    indices = np.array([info['columns'].index(n) for n in TEMPORAL_NAMES])
    if kind == 'tcn':
        mean, scale = mean[indices], scale[indices]
    net = TemporalNet(2 * len(mean)) if kind == 'tcn' else FrameNet(2 * len(mean))
    optimiser = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs, eta_min=.0001)
    history = []
    checkpoint = folder / (kind + '_training.pt')
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
        net.load_state_dict(saved['weights'])
        optimiser.load_state_dict(saved['optimiser'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'])
        rng.bit_generator.state = saved['numpy_rng']
        history = saved['history']
    for epoch in range(len(history), epochs):
        net.train()
        losses, correct, count = [], 0, 0
        if kind != 'tcn':
            for position in rng.permutation(len(info['shards'])):
                stem = info['shards'][position]
                raw = np.load(data / (stem + '_x.npy'), mmap_mode='r')
                y = np.load(data / (stem + '_y.npy'), mmap_mode='r')
                order = rng.permutation(len(y))
                for a in range(0, len(y), 2048):
                    take = order[a:a + 2048]
                    x = torch.from_numpy(normalise(raw[take], mean, scale))
                    labels = torch.from_numpy(np.array(y[take], dtype=np.int64))
                    p, q = net(x)
                    loss = (nn.functional.cross_entropy(p, labels[:, 0])
                            + .3 * nn.functional.cross_entropy(q, labels[:, 1]))
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                    correct += int((p.argmax(1) == labels[:, 0]).sum())
                    count += len(take)
                del raw, y
        else:
            # Chunks of 256 samples within one half; a halo of 30 samples on each side gives full
            # context and is left out of the loss. Frames without a label are masked.
            for position in rng.permutation(len(info['sequences'])):
                stem = info['sequences'][position]
                raw = np.load(data / (stem + '_seq.npy'), mmap_mode='r')
                y = np.load(data / (stem + '_target.npy'), mmap_mode='r')
                starts = rng.permutation(np.arange(0, len(y), 256))
                for a in range(0, len(starts), 8):
                    centres = starts[a:a + 8]
                    grid = centres[:, None] + np.arange(-30, 286)[None, :]
                    clipped = np.clip(grid, 0, len(y) - 1)
                    x = normalise(raw[clipped], mean, scale).transpose(0, 2, 1).copy()
                    targets = np.array(y[clipped[:, 30:-30]], dtype=np.int64)
                    valid = ((grid[:, 30:-30] >= 0) & (grid[:, 30:-30] < len(y))
                             & (targets[:, :, 0] < 3))
                    if not valid.any():
                        continue
                    p, q = net(torch.from_numpy(x))
                    p, q = p[:, :, 30:-30].transpose(1, 2), q[:, :, 30:-30].transpose(1, 2)
                    keep = torch.from_numpy(valid)
                    labels = torch.from_numpy(targets)
                    boundary_weight = torch.where(labels[:, :, 1][keep] > 0, 3., 1.)
                    state_loss = nn.functional.cross_entropy(p[keep], labels[:, :, 0][keep],
                                                             reduction='none')
                    loss = ((state_loss * boundary_weight).mean()
                            + .3 * nn.functional.cross_entropy(q[keep], labels[:, :, 1][keep]))
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                    correct += int((p[keep].argmax(1) == labels[:, :, 0][keep]).sum())
                    count += int(keep.sum())
                del raw, y
        scheduler.step()
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)),
                            training_accuracy=correct / max(count, 1), samples=count))
        print('train', kind, history[-1], flush=True)
        torch.save(dict(weights=net.state_dict(), optimiser=optimiser.state_dict(),
                        scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                        numpy_rng=rng.bit_generator.state, history=history), checkpoint)
    torch.save(dict(kind=kind, weights=net.state_dict(), mean=mean, scale=scale, indices=indices,
                    columns=info['columns'], training_matches=info['training_matches'],
                    history=history), destination)


def continue_from(base, folder, kind):
    """Start a continuation from the base run's last checkpoint: same weights, optimiser moments
    and random streams, a fresh cosine schedule from 0.0003 down to 0.00003."""
    folder.mkdir(parents=True, exist_ok=True)
    saved = torch.load(base / (kind + '_training.pt'), weights_only=False, map_location='cpu')
    for group in saved['optimiser']['param_groups']:
        group['lr'] = group['initial_lr'] = .0003
    prototype = torch.nn.Linear(1, 1)
    optimiser = torch.optim.AdamW(prototype.parameters(), lr=.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EXTRA_EPOCHS[kind],
                                                           eta_min=.00003)
    saved['scheduler'] = scheduler.state_dict()
    saved['history'] = []
    torch.save(saved, folder / (kind + '_training.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR")
                        or os.path.join(os.path.dirname(MODEL_DIR), "data"),
                        help="the match data bundle (default: $MTAH_DATA_DIR, else ./data)")
    parser.add_argument("--out", default=MODEL_DIR, help="folder for the model files")
    args = parser.parse_args()
    torch.set_num_threads(4)
    paths = external_matches()
    if not paths:
        raise SystemExit("no training matches in cache/ext; run training.dfl and training.metrica")
    work = Path(CACHE_DIR) / "networks"
    if work.exists():
        shutil.rmtree(work)
    data = work / "data"
    info = prepare(data, paths, measure_profile(args.data_dir))
    os.makedirs(args.out, exist_ok=True)
    for kind in ("mlp", "tcn"):
        fit(data, work / "base", info, kind, BASE_EPOCHS[kind])
        continue_from(work / "base", work / "continued", kind)
        fit(data, work / "continued", info, kind, EXTRA_EPOCHS[kind])
        shutil.copyfile(work / "continued" / (kind + ".pt"), os.path.join(args.out, OUTPUT[kind]))
    print("saved mlp48.pt and tcn24.pt in " + args.out)


if __name__ == "__main__":
    main()
