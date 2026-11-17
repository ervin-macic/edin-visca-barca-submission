"""Paths, the labelled match format and helpers shared by the training scripts.

A training match is a match (possession/data.py) plus
    label   int8 (T,)   H / A / D codes from the dataset's own annotations, X where unknown
"""

import glob
import os

import numpy as np

from possession import features
from possession.data import A, D, FPS, H, X, period_runs  # noqa: F401  (re-exported)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(ROOT, "external_data")
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
MODEL_DIR = os.path.join(ROOT, "models")
KEYS = ("ball", "home", "away", "label", "period", "pitch")
CHANGE_WIDEN = 2                         # frames either side of a label change that count as "at" it


def cache_path(*parts):
    path = os.path.join(CACHE_DIR, *parts)
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return path


def save_match(path, match):
    np.savez_compressed(path, **{key: match[key] for key in KEYS})


def load_match(path):
    with np.load(path) as data:
        return {key: data[key] for key in KEYS}


def external_matches(sources=("dfl", "metrica")):
    """{name: path} of the converted training matches in cache/ext, in a fixed order."""
    out = {}
    for path in sorted(glob.glob(cache_path("ext", "*.npz"))):
        name = os.path.basename(path)[:-4]
        if name.split("_")[0] in sources:
            out[name] = path
    return out


def swap_teams(match):
    """The same match with home and away exchanged, labels included."""
    label = match["label"].copy()
    home_label, away_label = label == H, label == A
    label[home_label], label[away_label] = A, H
    out = features.swap_teams(match)
    out["label"] = label
    return out


def change_targets(label, period, widen=CHANGE_WIDEN):
    """Per frame: 0 when no label change is within `widen` frames, else 1 + the label that starts
    at that change (1 = home takes over, 2 = away takes over, 3 = dead ball starts)."""
    out = np.zeros(len(label), dtype=np.int32)
    change = np.flatnonzero((label[1:] != label[:-1]) & (period[1:] == period[:-1])
                            & (period[1:] > 0) & (label[1:] != X) & (label[:-1] != X)) + 1
    for offset in sorted(range(-widen, widen + 1), key=abs, reverse=True):
        index = change + offset
        keep = (index >= 0) & (index < len(label))
        out[index[keep]] = 1 + label[change[keep]]
    return out


def label_summary(match):
    """A few numbers that describe a label track: shares, possessions per minute, lengths."""
    label, period = match["label"], match["period"]
    spells = []
    for start, stop, _ in period_runs(period):
        window = label[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in (H, A):
                spells.append((b - a) / float(FPS))
    graded = label[(period != 0) & (label != X)]
    spells = np.array(spells) if spells else np.zeros(1)
    return {
        "minutes": len(graded) / float(FPS) / 60.0,
        "share_H": float(np.mean(graded == H)) if len(graded) else 0.0,
        "share_A": float(np.mean(graded == A)) if len(graded) else 0.0,
        "share_D": float(np.mean(graded == D)) if len(graded) else 0.0,
        "n_spells": int(len(spells)),
        "spell_mean_s": float(spells.mean()),
    }
