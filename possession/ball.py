"""The fused ball read by the two neural networks: the tactical ball, gaps filled from the broadcast.

The broadcast recording has its own automatic tracking on its own timeline. While it shows the live
picture it runs in step with the tactical video, shifted by a constant per half (BROADCAST_OFFSET,
from a label-free alignment of the player positions of the two trackings). Fusion rules, all fixed:

  1. Teleports: a ball step faster than TELEPORT is a track swap. A stretch shorter than 1 s
     between two such swaps is the tracker on another object, and is dropped.
  2. The tactical ball is used where it exists; the broadcast ball fills its gaps.
  3. Where both exist but lie more than DISAGREE apart, the episode keeps the source that joins
     the surrounding track with the smaller jump (and is smoother inside).
  4. At a change of source, the incoming source is shifted to meet the outgoing one, the shift
     fading out over SHORT_GAP frames.
  5. Gaps of up to SHORT_GAP frames are bridged by a straight line.
"""

import warnings

import numpy as np

from .data import FPS, N_FRAMES, asset_path, read_atd, read_slots, to_metres

BROADCAST_CUT = 84701             # first broadcast frame of the second half
BROADCAST_OFFSET = (4014, 26920)  # tactical frame = broadcast frame + offset (first, second half)
TELEPORT = 40.0                   # m/s
DISAGREE = 3.0                    # m: each tracker is good to about 1 m, plus a frame of a fast ball
SHORT_GAP = 12                    # frames (about 0.5 s)


def runs(mask):
    """Half-open [start, end) runs of True."""
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def drop_teleports(b):
    b = b.copy()
    step = np.r_[0, np.linalg.norm(np.diff(b, axis=0), axis=1)] * FPS
    jumps = np.flatnonzero(step > TELEPORT)
    for j0, j1 in zip(jumps[:-1], jumps[1:]):
        if j1 - j0 < FPS and np.isfinite(b[j0:j1, 0]).all():
            b[j0:j1] = np.nan
    return b


def roughness(seg):
    if len(seg) < 3:
        return 0.0
    return float(np.nanmean(np.linalg.norm(np.diff(seg, 2, axis=0), axis=1)))


def fuse(tactical, broadcast):
    """(T, 2) float64 ball tracks on the tactical timeline -> the fused (T, 2) track."""
    T, B = drop_teleports(tactical), drop_teleports(broadcast)
    tv, bv = np.isfinite(T[:, 0]), np.isfinite(B[:, 0])
    n = len(T)
    F = np.full((n, 2), np.nan)
    src = np.zeros(n, np.int8)                  # 0 none, 1 tactical, 2 broadcast, 3 line
    F[tv], src[tv] = T[tv], 1
    only_b = bv & ~tv
    F[only_b], src[only_b] = B[only_b], 2

    # disagreement episodes, merged when separated by fewer than 5 frames
    dis = tv & bv & (np.linalg.norm(T - B, axis=1) > DISAGREE)
    s, e = runs(dis)
    eps = []
    for a, b in zip(s, e):
        if eps and a - eps[-1][1] < 5:
            eps[-1][1] = b
        else:
            eps.append([a, b])
    provisional = F.copy()
    for a, b in eps:
        provisional[a:b] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)     # all-NaN roughness
        for a, b in eps:
            cost = {}
            for name, track in (("T", T), ("B", B)):
                c, k = 0.0, 0
                if a > 0 and np.isfinite(provisional[a - 1, 0]) and np.isfinite(track[a, 0]):
                    c += np.linalg.norm(track[a] - provisional[a - 1]); k += 1
                if b < n and np.isfinite(provisional[b, 0]) and np.isfinite(track[b - 1, 0]):
                    c += np.linalg.norm(track[b - 1] - provisional[b]); k += 1
                cost[name] = (c / max(k, 1)) + roughness(track[a:b]) * FPS * 0.1
            if cost["B"] < cost["T"]:
                m = np.isfinite(B[a:b, 0])
                F[a:b][m], src[a:b][m] = B[a:b][m], 2

    # blend source switches
    for f in np.flatnonzero((src[1:] != src[:-1]) & (src[1:] > 0) & (src[:-1] > 0)) + 1:
        off = F[f - 1] - F[f]
        if np.linalg.norm(off) > 5:          # a real jump, not a join offset
            continue
        k = 0
        while k < SHORT_GAP and f + k < n and src[f + k] == src[f]:
            F[f + k] += off * (1 - k / SHORT_GAP)
            k += 1

    # bridge short gaps
    vis = np.isfinite(F[:, 0])
    gs, ge = runs(~vis)
    for a, b in zip(gs, ge):
        if a > 0 and b < n and b - a <= SHORT_GAP:
            t = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
            F[a:b] = F[a - 1] + t[:, None] * (F[b] - F[a - 1])
            src[a:b] = 3
    return F


def with_fused_ball(match, data_dir):
    """A copy of the match whose ball is the fused ball (NaN outside the graded halves)."""
    home, _ = read_slots(asset_path(data_dir, "broadcast", "_atd.xml"))
    _, raw = read_atd(asset_path(data_dir, "broadcast", "_atd.txt"), len(home),
                      N_FRAMES["broadcast"])
    broadcast = to_metres(raw[:, :2])
    shift = match["pitch"] / 2                  # the fusion works in metres from a pitch corner
    tactical = match["ball"] + shift
    b = np.arange(len(broadcast))
    mapped = b + np.where(b < BROADCAST_CUT, BROADCAST_OFFSET[0], BROADCAST_OFFSET[1])
    valid = (mapped >= 0) & (mapped < len(tactical))
    mapped_ball = np.full_like(tactical, np.nan)
    mapped_ball[mapped[valid]] = broadcast[valid] + shift
    fused = fuse(tactical.astype(np.float64), mapped_ball.astype(np.float64)).astype(np.float32)
    out = dict(match)
    out["ball"] = fused - np.asarray(match["pitch"], np.float32) / 2
    out["ball"][match["period"] == 0] = np.nan
    return out
