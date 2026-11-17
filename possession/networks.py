"""The two neural networks (PyTorch, CPU) and their inference."""

import numpy as np
import torch
from torch import nn

from .data import period_runs


class FrameNet(nn.Module):
    """One frame's features -> state (H/A/D) and change logits."""

    def __init__(self, inputs):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inputs, 192), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(192, 96), nn.ReLU(), nn.Dropout(.10))
        self.state = nn.Linear(96, 3)
        self.change = nn.Linear(96, 4)

    def forward(self, x):
        x = self.body(x)
        return self.state(x), self.change(x)


class Residual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Dropout(.10),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Dropout(.10))

    def forward(self, x):
        return x + self.body(x)


class TemporalNet(nn.Module):
    """Features at 12.5 Hz -> state and change logits. Two convolutions per residual block give a
    receptive field of 61 samples (4.88 s)."""
    halo = 30

    def __init__(self, inputs):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(inputs, 48, 1), nn.ReLU(),
                                  *[Residual(48, d) for d in (1, 2, 4, 8)])
        self.state = nn.Conv1d(48, 3, 1)
        self.change = nn.Conv1d(48, 4, 1)

    def forward(self, x):
        x = self.body(x)
        return self.state(x), self.change(x)


def normalise(x, mean, scale):
    """z-scores with the training mean and scale, clipped to +/-8, missing values 0, plus one
    missing-value indicator per column."""
    finite = np.isfinite(x)
    z = np.where(finite, (x - mean) / scale, 0).astype(np.float32)
    np.clip(z, -8, 8, out=z)
    return np.concatenate((z, (~finite).astype(np.float32)), axis=-1)


def predict_net(saved, table, period):
    """(T, 3) state and (T, 4) change probabilities of one saved network; flat outside the halves.

    The temporal network runs on every second frame within each half (never across halves), and
    its outputs are interpolated back to every frame.
    """
    kind = saved["kind"]
    net = TemporalNet(2 * len(saved["mean"])) if kind == "tcn" else FrameNet(2 * len(saved["mean"]))
    net.load_state_dict(saved["weights"])
    net.eval()
    p = np.full((len(table), 3), 1 / 3, np.float32)
    q = np.zeros((len(table), 4), np.float32)
    q[:, 0] = 1
    with torch.inference_mode():
        for start, stop, _ in period_runs(period):
            if kind == "tcn":
                grid = np.arange(start, stop, 2)
                small_p, small_q = [], []
                for a in range(0, len(grid), 2048):
                    b = min(a + 2048, len(grid))
                    context = np.clip(np.arange(a - 30, b + 30), 0, len(grid) - 1)
                    x = normalise(table[grid[context]][:, saved["indices"]], saved["mean"],
                                  saved["scale"])
                    pp, qq = net(torch.from_numpy(x.T[None].copy()))
                    small_p.append(pp.softmax(1)[0, :, 30:-30].T.numpy())
                    small_q.append(qq.softmax(1)[0, :, 30:-30].T.numpy())
                pp, qq = np.concatenate(small_p), np.concatenate(small_q)
                t = np.arange(start, stop)
                for j in range(3):
                    p[start:stop, j] = np.interp(t, grid, pp[:, j])
                for j in range(4):
                    q[start:stop, j] = np.interp(t, grid, qq[:, j])
            else:
                for a in range(start, stop, 8192):
                    b = min(a + 8192, stop)
                    x = normalise(table[a:b], saved["mean"], saved["scale"])
                    pp, qq = net(torch.from_numpy(x))
                    p[a:b], q[a:b] = pp.softmax(1).numpy(), qq.softmax(1).numpy()
    return p, q
