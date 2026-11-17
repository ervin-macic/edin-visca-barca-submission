"""The fused ball (possession/ball.py), simulated on the training matches for the networks.

At prediction time the networks read the tactical ball with some of its gaps filled from the
broadcast tracking. On a training match the same thing is imitated: inside each ball gap of at least
a second, a contiguous stretch of the true ball position comes back, with an offset of about 1 m
(6 m in one case in ten) and 0.3 m of noise per frame. How much of a gap comes back is drawn from
the shares the fusion achieves on the provided tracking (measure_profile, no labels involved),
separately for gaps during which the players walk and gaps during which they move.
"""

import warnings

import numpy as np

from possession import features
from possession.ball import with_fused_ball
from possession.cleaning import load_tracking

from . import augment
from .common import D, FPS, period_runs


def measure_profile(data_dir):
    """{'walk': [...], 'move': [...]}: per ball gap of at least 1 s in the provided tactical
    tracking, the share of its frames the fused ball fills. A gap is 'walk' when the median mean
    player speed in it is below 1.5 m/s."""
    match = load_tracking(data_dir, clean=False)
    fused = with_fused_ball(match, data_dir)
    shares = dict(walk=[], move=[])
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    for start, stop, _ in period_runs(match["period"]):
        home_speed, _ = features.team_motion(match["home"][start:stop])
        away_speed, _ = features.team_motion(match["away"][start:stop])
        speed = np.nanmean(np.stack((home_speed, away_speed)), axis=0)
        hidden = ~np.isfinite(match["ball"][start:stop, 0])
        edges = np.diff(np.r_[0, hidden.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a < 25:
                continue
            pace = float(np.nanmedian(speed[a:b]))
            kind = "walk" if pace < 1.5 else "move"
            shares[kind].append(float(np.isfinite(fused["ball"][start + a:start + b, 0]).mean()))
    for kind in shares:
        if not shares[kind]:
            raise ValueError("no measured gaps for " + kind)
    return shares


def fill_external(match, degraded, profile):
    """The degraded copy of a training match with parts of its ball gaps filled back in.

    The match's labels only decide which share distribution a gap draws from (dead ball: 'walk').
    """
    ball = degraded["ball"].copy()
    hidden = (match["period"] > 0) & ~np.isfinite(ball[:, 0])
    rng = np.random.default_rng(int(hidden.sum()) * 7919 + len(ball))
    for start, stop, _ in period_runs(match["period"]):
        edges = np.diff(np.r_[0, hidden[start:stop].astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edges == 1) + start, np.flatnonzero(edges == -1) + start):
            n = b - a
            if n < FPS:
                continue
            kind = "walk" if np.mean(match["label"][a:b] == D) >= .5 else "move"
            k = int(round(float(rng.choice(profile[kind])) * n))
            if k < 2:
                continue
            s = a + int(rng.integers(0, n - k + 1))
            sigma = 6.0 if rng.random() < .10 else 1.0
            positions = (match["ball"][s:s + k] + rng.normal(0, sigma, 2)
                         + rng.normal(0, .3, (k, 2)))
            known = np.isfinite(positions).all(axis=1)
            ball[s:s + k][known] = positions[known]
    out = dict(degraded)
    out["ball"] = ball.astype(np.float32)
    return out


def make_degrader(profile):
    """augment.degrade followed by the fill simulation."""

    def degrade_with_fill(match, rng, **settings):
        return fill_external(match, augment.degrade(match, rng, **settings), profile)

    return degrade_with_fill
