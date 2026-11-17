"""Train the two LightGBM models on the converted training matches (cache/ext, other matches only).

    python -m training.train_trees [--out models]

  lightgbm_state.txt    P(H), P(A), P(D) for each frame
  lightgbm_change.txt   P(no change / home takes over / away takes over / dead ball starts)

Every training match is passed twice through the tracker simulation (augment.py, with players
filed under the wrong team now and then), each copy in both team orders. Rows are every 10th frame,
and every 2nd frame within 15 frames of a label change.
"""

import argparse
import os

import numpy as np
import lightgbm as lgb

from possession import features
from possession.models import GROUPS

from . import augment
from .common import MODEL_DIR, X, change_targets, external_matches, load_match, swap_teams

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "verbosity": -1,
    "seed": 7,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 12,          # does not change the result (deterministic=True)
}
ROUNDS = 200
STRIDE, EDGE_STRIDE, EDGE, DRAWS = 10, 2, 15, 2


def build_rows(match, rng, settings, swap, target):
    """Feature rows of one simulated copy of a training match."""
    truth = match["label"]
    match = augment.degrade(match, rng, **settings)
    if swap:
        match = swap_teams(match)
    table, names = features.compute(match, groups=GROUPS)
    offset = int(rng.integers(0, STRIDE))
    take = np.zeros(len(table), dtype=bool)
    take[offset::STRIDE] = True
    change = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    near = np.zeros(len(table) + 1, dtype=np.int32)
    np.add.at(near, np.maximum(change - EDGE, 0), 1)
    np.add.at(near, np.minimum(change + EDGE, len(table)), -1)
    near = np.cumsum(near[:-1]) > 0
    dense = np.zeros(len(table), dtype=bool)
    dense[offset % EDGE_STRIDE::EDGE_STRIDE] = True
    take |= near & dense
    rows = np.flatnonzero(take)
    rows = rows[(match["period"][rows] > 0) & (match["label"][rows] != X)]
    if target == "change":
        return table[rows], change_targets(match["label"], match["period"])[rows], names
    return table[rows], match["label"][rows].astype(np.int32), names


def training_set(matches, seed, target):
    rng = np.random.default_rng(seed)
    xs, ys, names = [], [], None
    for match in matches:
        for _ in range(DRAWS):
            settings = augment.random_settings(rng, flips=True)
            for swap in (False, True):
                x, y, names = build_rows(match, rng, settings, swap, target)
                xs.append(x)
                ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), names


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=MODEL_DIR, help="folder for the model files")
    args = parser.parse_args()
    paths = external_matches()
    if not paths:
        raise SystemExit("no training matches in cache/ext; run training.dfl and training.metrica")
    matches = [load_match(path) for path in paths.values()]
    os.makedirs(args.out, exist_ok=True)
    for target, seed, params in (("state", 2027, None), ("change", 2028, {"num_class": 4})):
        x, y, names = training_set(matches, seed, target)
        print("{} model: {} matches, {} rows, {} features".format(target, len(matches), len(y),
                                                                  len(names)), flush=True)
        data = lgb.Dataset(x, label=y, feature_name=list(names), free_raw_data=True)
        booster = lgb.train(dict(PARAMS, **(params or {})), data, num_boost_round=ROUNDS)
        booster.save_model(os.path.join(args.out, "lightgbm_{}.txt".format(target)))
        del x, y, data
    print("saved lightgbm_state.txt and lightgbm_change.txt in " + args.out)


if __name__ == "__main__":
    main()
