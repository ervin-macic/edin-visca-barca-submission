"""Per-frame probabilities -> one label per frame.

A three-state hidden Markov model (H, A, D) decoded with Viterbi: every frame pays the negative
log-probability of its label, every label change pays a fixed penalty, and entering a label at a
frame also pays CHANGE_WEIGHT * -log P(that label starts here) from the change outputs. The
penalties remove flicker without blurring where a change happens.
"""

import numpy as np
from numba import njit

from .data import A, D, H, period_runs

TEAM_SWITCH = 2.0               # penalty of a change H <-> A
TO_DEAD = 24.0                  # penalty of H or A -> D
FROM_DEAD = 24.0                # penalty of D -> H or A
CHANGE_WEIGHT = 3.0
EPS = 1e-6
CHANGE_FLOOR = 1e-3


@njit(cache=True)
def _viterbi(log_prob, switch_cost, entry_cost):
    """Most likely state path. Ties go to staying, then to the lower state index."""
    n, n_states = log_prob.shape
    score = log_prob[0].copy()
    new_score = np.empty(n_states)
    back = np.zeros((n, n_states), dtype=np.int8)
    for t in range(1, n):
        for j in range(n_states):
            best, best_value = j, score[j]
            for i in range(n_states):
                if i != j:
                    value = score[i] - switch_cost[i, j] - entry_cost[t, j]
                    if value > best_value or (value == best_value and i < best):
                        best, best_value = i, value
            back[t, j] = best
            new_score[j] = best_value + log_prob[t, j]
        for j in range(n_states):
            score[j] = new_score[j]
    path = np.empty(n, dtype=np.int8)
    last = 0
    for j in range(1, n_states):
        if score[j] > score[last]:
            last = j
    path[n - 1] = last
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(prob, period, change):
    """Labels for every frame inside the halves; frames outside get D."""
    out = np.full(len(prob), D, dtype=np.int8)
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = TEAM_SWITCH
    cost[H, D] = cost[A, D] = TO_DEAD
    cost[D, H] = cost[D, A] = FROM_DEAD
    for start, stop, _ in period_runs(period):
        log_prob = np.log(np.clip(prob[start:stop], EPS, 1.0))
        entry = -CHANGE_WEIGHT * np.log(np.clip(change[start:stop, 1:], CHANGE_FLOOR, 1.0))
        out[start:stop] = _viterbi(np.ascontiguousarray(log_prob, dtype=np.float64),
                                   np.ascontiguousarray(cost, dtype=np.float64),
                                   np.ascontiguousarray(entry, dtype=np.float64))
    return out


def dead_before_kickoff(match, labels, windows, radius=4.0):
    """Copy of labels with every frame of a half before its kick-off set to dead ball.

    The kick-off is the first frame inside the half's Smart Tagging kick-off window on which the
    ball is tracked within `radius` metres of the centre spot (the tracker picks the ball up as it
    is kicked). If the ball is not seen there, only the frames before the window become dead.
    """
    out = labels.copy()
    ball = match["ball"]
    near_centre = np.hypot(ball[:, 0], ball[:, 1]) < radius
    for start, stop, _ in period_runs(match["period"]):
        inside = [(first, last) for first, last in windows if start <= last and first < stop]
        if not inside:
            continue
        first, last = min(inside)
        kick = max(start, first)
        seen = np.flatnonzero(near_centre[kick:min(last, stop - 1) + 1])
        if len(seen):
            kick += int(seen[0])
        out[start:kick] = D
    return out
