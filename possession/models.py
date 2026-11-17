"""Per-frame probabilities from the three trained models, blended in equal thirds.

    lightgbm   gradient-boosted trees on 311 columns (features.compute with GROUPS), tracked ball:
               lightgbm_state.txt gives P(H / A / D), lightgbm_change.txt gives P(no change /
               home takes over / away takes over / dead ball starts)
    mlp48      frame-wise neural network (192-96 ReLU) on 281 columns, fused ball (ball.py)
    tcn24      temporal convolutional network (4.9 s receptive field) on 80 of the 281 columns,
               fused ball

Each network has a state head and a change head of the same form. All models were trained on
other matches only. Each one also runs on the match with the two teams exchanged, and the two
outputs are averaged.
"""

import os

import numpy as np
import lightgbm as lgb

from . import features
from .data import A, D, H, period_runs

GROUPS = ("keeper", "restart")           # extra feature groups of the LightGBM models
NETWORKS = ("mlp48", "tcn24")
SWAP_STATE = np.array([A, H, D])         # probability columns after exchanging the teams
SWAP_CHANGE = np.array([0, 2, 1, 3])     # the same for the change outputs


def lightgbm_probabilities(match, model_dir):
    """(T, 3) state and (T, 4) change probabilities; flat outside the halves."""
    state = lgb.Booster(model_file=os.path.join(model_dir, "lightgbm_state.txt"))
    change = lgb.Booster(model_file=os.path.join(model_dir, "lightgbm_change.txt"))
    table, names = features.compute(match, groups=GROUPS)
    swapped, _ = features.compute(features.swap_teams(match), groups=GROUPS)
    if names != state.feature_name() or names != change.feature_name():
        raise ValueError("feature columns differ from the ones the LightGBM models were trained on")
    inside = match["period"] > 0
    prob = np.full((len(table), 3), 1.0 / 3.0)
    prob[inside] = 0.5 * (state.predict(table[inside])
                          + state.predict(swapped[inside])[:, SWAP_STATE])
    moves = np.zeros((len(table), 4))
    moves[:, 0] = 1.0
    moves[inside] = 0.5 * (change.predict(table[inside])
                           + change.predict(swapped[inside])[:, SWAP_CHANGE])
    return prob, moves


def network_probabilities(match, model_dir):
    """{network: (T, 3)} state and {network: (T, 4)} change probabilities."""
    os.environ.setdefault("OMP_NUM_THREADS", "4")      # the thread count the networks run with
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    import torch
    from .networks import predict_net
    torch.set_num_threads(4)
    saved = {name: torch.load(os.path.join(model_dir, name + ".pt"), weights_only=False,
                              map_location="cpu") for name in NETWORKS}
    probs, changes = {}, {}
    for swap in (False, True):
        table, columns = features.compute(features.swap_teams(match) if swap else match,
                                          deceleration=True)
        for name, snapshot in saved.items():
            if snapshot["columns"] != columns:
                raise ValueError("feature columns differ from the ones {} was trained on".format(
                    name))
            p, q = predict_net(snapshot, table, match["period"])
            if swap:
                probs[name] = .5 * (probs[name] + p[:, SWAP_STATE])
                changes[name] = .5 * (changes[name] + q[:, SWAP_CHANGE])
            else:
                probs[name], changes[name] = p, q
        del table
    return probs, changes


def blend(tracked, fused, model_dir):
    """Equal-thirds average of the three models' state and change probabilities.

    tracked: the match with the tracked ball (LightGBM); fused: with the fused ball (networks).
    """
    probs, changes = {}, {}
    probs["lightgbm"], changes["lightgbm"] = lightgbm_probabilities(tracked, model_dir)
    p, q = network_probabilities(fused, model_dir)
    probs.update(p)
    changes.update(q)
    members = ("lightgbm",) + NETWORKS
    weight = 1 / len(members)
    prob = sum(weight * probs[k] for k in members)
    change = sum(weight * changes[k] for k in members)
    return prob, change
