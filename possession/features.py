"""Per-frame features from a match (data.py), computed separately in each half.

Everything here treats a team as an unordered set of points, never uses a slot's identity, and
copes with a missing ball (NaN). Distances are metres and windows are frames at 25 fps, so the same
code runs on the clean tracking of the training matches and on the automatic tracking of this one.

The pitch is oriented per half so that the home team attacks towards +x; swapping the two teams
therefore mirrors the oriented x axis, which keeps the team-swap symmetrisation in models.py exact
(it recomputes the features on the swapped match).
"""

import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from .data import FPS, period_runs

BIG = 1e6
INTERPOLATE_GAP = 12            # ball gaps up to ~0.5 s are bridged linearly
CONTACT_RADIUS = 1.5            # metres: "this team is on the ball"
WINDOWS = (12, 25, 50, 100, 200)
SHIFTS = (-100, -50, -25, -12, -5, 5, 12, 25, 50, 100)
CAP = 250                       # cap for "time since / until" features, frames (10 s)
LONG_CAP = 1500                 # the same for ball-gap lengths (60 s): real gaps are long
KICK = 12.0                     # m/s^2: a peak of ball acceleration above this is a touch.
                                # The automatic tracker smooths the ball, so touches are broad
                                # bumps of 10-25 m/s^2, not spikes; this value gives about 25
                                # events per minute on the ATD and on equally smoothed DFL data.
KICK_PEAK = 6                   # a kick is the highest acceleration within this many frames
MAX_STEP = 0.6                  # metres per frame; larger jumps are track swaps, not movement


def interpolate_short_gaps(ball, max_gap=INTERPOLATE_GAP):
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    index = np.flatnonzero(seen)
    if len(index) < 2:
        return out
    gaps = np.flatnonzero(np.diff(index) > 1)
    for g in gaps:
        a, b = index[g], index[g + 1]
        if b - a - 1 <= max_gap:
            t = (np.arange(a + 1, b) - a) / float(b - a)
            out[a + 1:b] = ball[a] + t[:, None] * (ball[b] - ball[a])
    return out


def sorted_distances(team, ball, k=2):
    """Distances from the ball to the k nearest team members, NaN where unknown. (T, k)."""
    d = np.hypot(team[:, :, 0] - ball[:, None, 0], team[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, BIG)
    d = np.sort(d, axis=1)[:, :k]
    return np.where(d >= BIG, np.nan, d)


def nearest_velocity(team, ball):
    """Velocity (m/s) of the team member nearest to the ball on each frame, NaN where unknown.

    The velocity is taken along that member's own slot, so a slot that has just appeared (or
    jumped to another player) gives NaN rather than a wrong speed.
    """
    d = np.hypot(team[:, :, 0] - ball[:, None, 0], team[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, BIG)
    nearest = np.argmin(d, axis=1)
    velocity = np.full(team.shape, np.nan, dtype=np.float32)
    step = team[4:] - team[:-4]
    jump = np.hypot(step[:, :, 0], step[:, :, 1]) > 4 * MAX_STEP
    velocity[2:-2] = np.where(jump[:, :, None], np.nan, step * (FPS / 4.0))
    out = velocity[np.arange(len(team)), nearest]
    out[d[np.arange(len(team)), nearest] >= BIG] = np.nan
    return out


def past_mean(x, w):
    """Mean of x over the w frames ending at t (inclusive)."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    t = np.arange(len(x))
    lo = np.maximum(t - w + 1, 0)
    return ((c[t + 1] - c[lo]) / (t + 1 - lo)).astype(np.float32)


def future_mean(x, w):
    """Mean of x over the w frames starting at t + 1."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    n = len(x)
    t = np.arange(n)
    lo = np.minimum(t + 1, n)
    hi = np.minimum(t + 1 + w, n)
    count = np.maximum(hi - lo, 1)
    return ((c[hi] - c[lo]) / count).astype(np.float32)


def since_last(flag, cap=CAP):
    """Frames since flag was last True (0 on the frame itself), capped."""
    index = np.where(flag, np.arange(len(flag)), -BIG)
    last = np.maximum.accumulate(index)
    return np.minimum(np.arange(len(flag)) - last, cap).astype(np.float32)


def until_next(flag, cap=CAP):
    return since_last(flag[::-1], cap)[::-1].copy()


def value_at_last(values, flag):
    """values at the most recent frame where flag is True (NaN before the first)."""
    index = np.where(flag, np.arange(len(flag)), -1)
    last = np.maximum.accumulate(index)
    out = values[np.maximum(last, 0)].astype(np.float32)
    out[last < 0] = np.nan
    return out


def value_at_next(values, flag):
    return value_at_last(values[::-1], flag[::-1])[::-1].copy()


def shift(x, k):
    """x(t + k), NaN where that falls outside the array."""
    out = np.full(len(x), np.nan, dtype=np.float32)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def smooth(x, w):
    """Centred NaN-aware moving average over 2w+1 frames."""
    valid = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0), dtype=np.float64)))
    k = np.concatenate(([0], np.cumsum(valid)))
    n = len(x)
    t = np.arange(n)
    lo = np.maximum(t - w, 0)
    hi = np.minimum(t + w + 1, n)
    count = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / count
    out[count == 0] = np.nan
    return out.astype(np.float32)


def team_motion(team):
    """Per-frame mean speed of a team's tracked points (m/s) and the share that is near-static."""
    step = np.hypot(np.diff(team[:, :, 0], axis=0), np.diff(team[:, :, 1], axis=0))
    step = np.where(step > MAX_STEP, np.nan, step) * FPS
    step = np.concatenate((step[:1], step), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_speed = np.nanmean(step, axis=1)
        slow = np.nanmean(np.where(np.isfinite(step), step < 1.0, np.nan), axis=1)
    return mean_speed.astype(np.float32), slow.astype(np.float32)


def team_shape(team, sign):
    """Centroid, spread and line positions of a team, with x oriented by sign."""
    x = team[:, :, 0] * sign
    y = team[:, :, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cx = np.nanmean(x, axis=1)
        cy = np.nanmean(y, axis=1)
        sx = np.nanstd(x, axis=1)
        sy = np.nanstd(y, axis=1)
    filled_low = np.sort(np.where(np.isfinite(x), x, BIG), axis=1)
    filled_high = np.sort(np.where(np.isfinite(x), x, -BIG), axis=1)
    # second-lowest and second-highest x: the back line and the front line without one outlier
    back = np.where(filled_low[:, 1] >= BIG, np.nan, filled_low[:, 1])
    front = np.where(filled_high[:, -2] <= -BIG, np.nan, filled_high[:, -2])
    return [a.astype(np.float32) for a in (cx, cy, sx, sy, back, front)]


def count_within(team, ball, radius):
    d = np.hypot(team[:, :, 0] - ball[:, None, 0], team[:, :, 1] - ball[:, None, 1])
    out = np.sum(d < radius, axis=1).astype(np.float32)
    out[~np.isfinite(ball[:, 0])] = np.nan
    return out


PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16     # metres, Laws of the Game
KEEPER_RANGE = 35.0             # metres from his own goal: the nearest player within it is "the keeper"
GROUPS = ("keeper", "restart")  # optional feature groups (run_features(..., groups=...))


def keeper_and_restart(ball, seen, home, away, length, width, sign, groups):
    """Optional columns, appended after the 273 standard ones.

    keeper:  each team's goalkeeper, found the same way on every match (the team's player nearest
             to its own goal centre, if within KEEPER_RANGE; no roles or slots are used): his
             distance from the goal and from the ball, whether he is on the ball, how long since he
             was, whether the ball is in that team's penalty area.
    restart: the shapes of restarts, readable from the players alone, because the real tracker
             mostly loses the ball while it is dead: opponents inside a team's penalty area (a goal
             kick has none), a player at an attacking corner flag (corner), a player beyond a
             touchline (throw-in), a player at the centre spot and the share of a team in its own
             half (kick-off).
    Columns are per team (_h home, _a away), each oriented so that the team defends -x.
    """
    f = {}
    for name, team, s in (("h", home, sign), ("a", away, -sign)):
        x, y = team[:, :, 0] * s, team[:, :, 1]
        present = np.isfinite(x)
        bx, by = ball[:, 0] * s, ball[:, 1]
        opp = away if name == "h" else home
        ox = opp[:, :, 0] * s
        oy = opp[:, :, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if "keeper" in groups:
                to_goal = np.where(present, np.hypot(x + length / 2.0, y), BIG)
                k = np.argmin(to_goal, axis=1)
                rows = np.arange(len(x))
                has = to_goal[rows, k] < KEEPER_RANGE
                kx = np.where(has, x[rows, k], np.nan)
                ky = np.where(has, y[rows, k], np.nan)
                f["gk_goal_d_" + name] = np.where(has, to_goal[rows, k], np.nan).astype(np.float32)
                d_gk = np.hypot(kx - bx, ky - by)
                f["gk_ball_d_" + name] = d_gk.astype(np.float32)
                everyone = np.concatenate((home, away), axis=1)
                nearest = np.nanmin(np.hypot(everyone[:, :, 0] * s - bx[:, None],
                                             everyone[:, :, 1] - by[:, None]), axis=1)
                on_ball = (d_gk < CONTACT_RADIUS) & (d_gk <= nearest + 1e-6)
                f["gk_on_ball_" + name] = np.where(seen, on_ball, np.nan).astype(np.float32)
                f["since_gk_" + name] = since_last(on_ball & seen)
                f["gk_ball_p50_" + name] = past_mean(on_ball & seen, 50)
                f["gk_ball_f50_" + name] = future_mean(on_ball & seen, 50)
                in_box = (bx < -length / 2.0 + PENALTY_DEPTH) & (np.abs(by) < PENALTY_HALF_WIDTH)
                f["ball_box_" + name] = np.where(seen, in_box, np.nan).astype(np.float32)
            if "restart" in groups:
                box_line = -length / 2.0 + PENALTY_DEPTH
                in_own_box = np.isfinite(ox) & (ox < box_line) & (np.abs(oy) < PENALTY_HALF_WIDTH)
                f["box_opp_" + name] = smooth(in_own_box.sum(1).astype(np.float32), 12)
                own = present & (x < box_line) & (np.abs(y) < PENALTY_HALF_WIDTH)
                f["box_own_" + name] = smooth(own.sum(1).astype(np.float32), 12)
                corner = np.nanmin(np.minimum(np.hypot(x - length / 2.0, y - width / 2.0),
                                              np.hypot(x - length / 2.0, y + width / 2.0)), axis=1)
                f["corner_" + name] = smooth(np.minimum(corner, 30.0).astype(np.float32), 12)
                f["out_y_" + name] = smooth((np.nanmax(np.abs(y), axis=1) - width / 2.0)
                                            .astype(np.float32), 12)
                f["centre_" + name] = smooth(np.minimum(np.nanmin(np.hypot(x, y), axis=1), 30.0)
                                             .astype(np.float32), 12)
                share = present.sum(1)
                f["own_half_" + name] = smooth(np.where(share > 0, (present & (x < 0)).sum(1)
                                                        / np.maximum(share, 1), np.nan)
                                               .astype(np.float32), 12)
    if "restart" in groups:
        for name in ("box_opp_h", "box_opp_a", "corner_h", "corner_a", "out_y_h", "out_y_a"):
            for k in (-50, 50):
                f["{}@{:+d}".format(name, k)] = shift(f[name], k)
    return f


def run_features(ball_raw, home, away, pitch, groups=()):
    """Feature columns for one half. Returns (dict name -> (T,) float32).

    groups: optional extra feature groups (see keeper_and_restart), appended after the 273
    standard columns.
    """
    length, width = float(pitch[0]), float(pitch[1])
    n = len(ball_raw)
    f = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        home_x, away_x = np.nanmean(home[:, :, 0]), np.nanmean(away[:, :, 0])
    sign = 1.0 if not (away_x < home_x) else -1.0      # home attacks towards +x

    seen_raw = np.isfinite(ball_raw[:, 0])
    ball = interpolate_short_gaps(ball_raw)
    seen = np.isfinite(ball[:, 0])
    f["seen"] = seen.astype(np.float32)

    # --- ball kinematics
    bx, by = ball[:, 0] * sign, ball[:, 1]
    vx = np.full(n, np.nan, dtype=np.float32)
    vy = np.full(n, np.nan, dtype=np.float32)
    vx[2:-2] = (bx[4:] - bx[:-4]) * (FPS / 4.0)
    vy[2:-2] = (by[4:] - by[:-4]) * (FPS / 4.0)
    speed = np.hypot(vx, vy)
    accel = np.full(n, np.nan, dtype=np.float32)
    accel[2:-2] = np.hypot(vx[4:] - vx[:-4], vy[4:] - vy[:-4]) * (FPS / 4.0)
    margin = np.minimum(length / 2.0 - np.abs(ball[:, 0]), width / 2.0 - np.abs(ball[:, 1]))
    f["ball_x"] = bx.astype(np.float32)
    f["ball_absy"] = np.abs(by).astype(np.float32)
    f["ball_vx"] = vx
    f["ball_speed"] = speed.astype(np.float32)
    f["ball_accel"] = accel
    f["ball_margin"] = margin.astype(np.float32)
    f["ball_margin_x"] = (length / 2.0 - np.abs(ball[:, 0])).astype(np.float32)
    f["ball_margin_y"] = (width / 2.0 - np.abs(ball[:, 1])).astype(np.float32)

    # --- who is near the ball
    dh = sorted_distances(home, ball)
    da = sorted_distances(away, ball)
    d1h, d1a = dh[:, 0], da[:, 0]
    f["d1_h"], f["d1_a"] = d1h.astype(np.float32), d1a.astype(np.float32)
    f["d2_h"], f["d2_a"] = dh[:, 1].astype(np.float32), da[:, 1].astype(np.float32)
    f["d1_diff"] = (d1a - d1h).astype(np.float32)          # > 0: home is closer
    with np.errstate(invalid="ignore"):
        closer_h = (d1h < d1a)
        closer_a = (d1a < d1h)
        contact_h = closer_h & (d1h < CONTACT_RADIUS)
        contact_a = closer_a & (d1a < CONTACT_RADIUS)
        tight_h = closer_h & (d1h < 0.7)
        tight_a = closer_a & (d1a < 0.7)
        slow_ball = speed < 1.0
        outside = margin < 0.0
    for name, team in (("h", home), ("a", away)):
        f["n3_" + name] = count_within(team, ball, 3.0)
        f["n10_" + name] = count_within(team, ball, 10.0)

    # --- how the ball moves relative to the nearest player of each team. The automatic tracker
    # smooths the ball so much that a touch is a gentle bend, not a kink; a ball under control
    # shows up better as "moving with a player" and a turnover as "leaving one, closing on another".
    ball_velocity = np.stack((np.where(np.isfinite(vx), vx, np.nan) * sign,
                              np.where(np.isfinite(vy), vy, np.nan)), axis=1)
    for name, team, distance in (("h", home, d1h), ("a", away, d1a)):
        relative = ball_velocity - nearest_velocity(team, ball)
        f["rel_speed_" + name] = np.hypot(relative[:, 0], relative[:, 1]).astype(np.float32)
        closing = np.full(n, np.nan, dtype=np.float32)
        closing[5:-5] = (distance[10:] - distance[:-10]) * (FPS / 10.0)      # < 0: getting closer
        f["closing_" + name] = closing
    f["rel_speed_diff"] = f["rel_speed_a"] - f["rel_speed_h"]               # > 0: moves with home
    for w in (12, 25):
        before_x, before_y = shift(vx, -w), shift(vy, -w)
        after_x, after_y = shift(vx, w), shift(vy, w)
        speed_before, speed_after = np.hypot(before_x, before_y), np.hypot(after_x, after_y)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = (before_x * after_x + before_y * after_y) / (speed_before * speed_after)
            turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            turn[(speed_before < 1.0) | (speed_after < 1.0)] = np.nan
            f["turn_{}".format(w)] = turn.astype(np.float32)
            f["speed_change_{}".format(w)] = (speed_after - speed_before).astype(np.float32)

    seen_f = seen.astype(np.float32)
    for w in WINDOWS:
        for tag, mean in (("p", past_mean), ("f", future_mean)):
            share = np.maximum(mean(seen_f, w), 1e-6)
            f["seen_{}{}".format(tag, w)] = mean(seen_f, w)
            f["closer_h_{}{}".format(tag, w)] = mean(closer_h.astype(np.float32), w) / share
            f["contact_h_{}{}".format(tag, w)] = mean(contact_h.astype(np.float32), w) / share
            f["contact_a_{}{}".format(tag, w)] = mean(contact_a.astype(np.float32), w) / share
            f["slow_{}{}".format(tag, w)] = mean(slow_ball.astype(np.float32), w) / share
            f["outside_{}{}".format(tag, w)] = mean(outside.astype(np.float32), w) / share
            for column in ("closer_h_", "contact_h_", "contact_a_", "slow_", "outside_"):
                key = "{}{}{}".format(column, tag, w)
                f[key][f["seen_{}{}".format(tag, w)] == 0] = np.nan

    # --- kicks: a sharp change of ball velocity next to a player is a touch by that player's team
    with np.errstate(invalid="ignore"):
        kick_h = np.where(closer_h & (d1h < 2.5) & np.isfinite(accel), accel, 0.0)
        kick_a = np.where(closer_a & (d1a < 2.5) & np.isfinite(accel), accel, 0.0)
        accel_known = np.where(np.isfinite(accel), accel, 0.0)
        peak = (accel_known >= maximum_filter1d(accel_known, size=2 * KICK_PEAK + 1,
                                                mode="nearest")) & (accel_known > KICK)
        strong_h = peak & (kick_h > 0)
        strong_a = peak & (kick_a > 0)
    f["since_kick_h"], f["until_kick_h"] = since_last(strong_h), until_next(strong_h)
    f["since_kick_a"], f["until_kick_a"] = since_last(strong_a), until_next(strong_a)
    f["kick_order_past"] = f["since_kick_a"] - f["since_kick_h"]
    f["kick_order_next"] = f["until_kick_a"] - f["until_kick_h"]
    for w in (12, 25, 50, 100):
        f["kick_h_p{}".format(w)] = maximum_filter1d(kick_h, size=w, origin=(w - 1) // 2,
                                                     mode="constant").astype(np.float32)
        f["kick_a_p{}".format(w)] = maximum_filter1d(kick_a, size=w, origin=(w - 1) // 2,
                                                     mode="constant").astype(np.float32)
        f["kick_h_f{}".format(w)] = maximum_filter1d(kick_h, size=w, origin=-(w // 2),
                                                     mode="constant").astype(np.float32)
        f["kick_a_f{}".format(w)] = maximum_filter1d(kick_a, size=w, origin=-(w // 2),
                                                     mode="constant").astype(np.float32)

    # --- last / next contact per team, and what the ball was doing when last / next seen
    for name, contact, tight in (("h", contact_h, tight_h), ("a", contact_a, tight_a)):
        f["since_contact_" + name] = since_last(contact)
        f["until_contact_" + name] = until_next(contact)
        f["since_tight_" + name] = since_last(tight)
        f["until_tight_" + name] = until_next(tight)
    f["contact_order_past"] = f["since_contact_a"] - f["since_contact_h"]     # > 0: home later
    f["contact_order_next"] = f["until_contact_a"] - f["until_contact_h"]     # > 0: home sooner
    f["since_seen"] = since_last(seen)
    f["until_seen"] = until_next(seen)
    f["since_seen_long"] = since_last(seen, LONG_CAP)
    f["until_seen_long"] = until_next(seen, LONG_CAP)
    f["since_outside"] = since_last(outside)
    f["until_outside"] = until_next(outside)
    for name, values in (("margin", margin), ("speed", speed), ("d1_diff", f["d1_diff"]),
                         ("x", bx), ("d1_h", d1h), ("d1_a", d1a)):
        ok = seen & np.isfinite(values)
        f["last_" + name] = value_at_last(values, ok)
        f["next_" + name] = value_at_next(values, ok)
    f["gap_dx"] = f["next_x"] - f["last_x"]                # ball travel across a tracking gap
    # who was on the ball in the second before it was lost, and in the second after it came back
    f["before_gap_h"] = value_at_last(f["contact_h_p25"], seen)
    f["before_gap_a"] = value_at_last(f["contact_a_p25"], seen)
    f["after_gap_h"] = value_at_next(f["contact_h_f25"], seen)
    f["after_gap_a"] = value_at_next(f["contact_a_f25"], seen)
    f["gap_length"] = np.where(seen, 0.0, f["since_seen_long"] + f["until_seen_long"]).astype(np.float32)
    f["gap_position"] = np.where(seen, 0.0, f["since_seen_long"]
                                 / np.maximum(f["since_seen_long"] + f["until_seen_long"], 1.0)
                                 ).astype(np.float32)

    # --- ball extremes in short windows (kicks, stops)
    speed_fill_low = np.where(np.isfinite(speed), speed, -BIG)
    speed_fill_high = np.where(np.isfinite(speed), speed, BIG)
    accel_fill = np.where(np.isfinite(accel), accel, -BIG)
    for w in (12, 50):
        peak = maximum_filter1d(speed_fill_low, size=2 * w + 1, mode="nearest")
        low = minimum_filter1d(speed_fill_high, size=2 * w + 1, mode="nearest")
        kick = maximum_filter1d(accel_fill, size=2 * w + 1, mode="nearest")
        f["speed_max_{}".format(w)] = np.where(peak <= -BIG, np.nan, peak).astype(np.float32)
        f["speed_min_{}".format(w)] = np.where(low >= BIG, np.nan, low).astype(np.float32)
        f["accel_max_{}".format(w)] = np.where(kick <= -BIG, np.nan, kick).astype(np.float32)

    # --- players only: available on every frame, ball or no ball
    speed_h, slow_h = team_motion(home)
    speed_a, slow_a = team_motion(away)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        all_speed = np.nanmean(np.stack((speed_h, speed_a)), axis=0)
        all_slow = np.nanmean(np.stack((slow_h, slow_a)), axis=0)
    for w, tag in ((12, "1s"), (50, "4s"), (125, "10s")):
        f["speed_h_" + tag] = smooth(speed_h, w)
        f["speed_a_" + tag] = smooth(speed_a, w)
        f["speed_all_" + tag] = smooth(all_speed, w)
        f["static_all_" + tag] = smooth(all_slow, w)
    cxh, cyh, sxh, syh, back_h, front_h = team_shape(home, sign)
    cxa, cya, sxa, sya, back_a, front_a = team_shape(away, -sign)
    # For the away team x is oriented towards its own attack as well (it attacks -x of home).
    f["cx_h"], f["cx_a"] = smooth(cxh, 5), smooth(cxa, 5)
    f["cy_abs_h"], f["cy_abs_a"] = np.abs(smooth(cyh, 5)), np.abs(smooth(cya, 5))
    f["sx_h"], f["sx_a"] = smooth(sxh, 5), smooth(sxa, 5)
    f["sy_h"], f["sy_a"] = smooth(syh, 5), smooth(sya, 5)
    f["back_h"], f["back_a"] = smooth(back_h, 5), smooth(back_a, 5)
    f["front_h"], f["front_a"] = smooth(front_h, 5), smooth(front_a, 5)
    f["cx_gap"] = f["cx_h"] + f["cx_a"]                     # home centroid ahead of away centroid
    f["cx_mid"] = 0.5 * (f["cx_h"] - f["cx_a"])             # > 0: both teams sit in the away half
    for w, tag in ((25, "1s"), (75, "3s")):
        f["cvx_h_" + tag] = shift(f["cx_h"], w) - shift(f["cx_h"], -w)
        f["cvx_a_" + tag] = shift(f["cx_a"], w) - shift(f["cx_a"], -w)
    both = np.concatenate((home, away), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f["player_out_y"] = np.nanmax(np.abs(both[:, :, 1]), axis=1) - width / 2.0
        f["player_out_x"] = np.nanmax(np.abs(both[:, :, 0]), axis=1) - length / 2.0
    f["player_out_y"] = smooth(f["player_out_y"].astype(np.float32), 5)
    f["player_out_x"] = smooth(f["player_out_x"].astype(np.float32), 5)

    # --- the same core signals a little earlier and later
    core = ("d1_h", "d1_a", "ball_speed", "ball_margin", "seen", "speed_all_1s", "ball_x",
            "rel_speed_h", "rel_speed_a")
    for name in core:
        for k in SHIFTS:
            f["{}@{:+d}".format(name, k)] = shift(f[name], k)
    f["raw_seen"] = seen_raw.astype(np.float32)
    if groups:
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError("unknown feature groups: {}".format(sorted(unknown)))
        f.update(keeper_and_restart(ball, seen, home, away, length, width, sign, groups))
    return f


def compute(match, groups=(), deceleration=False):
    """(features (T, F) float32, names). Rows outside the halves are NaN.

    groups: optional feature groups (GROUPS). deceleration: append the eight player-deceleration
    columns of compute_deceleration.
    """
    period = match["period"]
    names = None
    out = None
    for start, stop, _ in period_runs(period):
        columns = run_features(match["ball"][start:stop], match["home"][start:stop],
                               match["away"][start:stop], match["pitch"], groups)
        if names is None:
            names = list(columns)
            out = np.full((len(period), len(names)), np.nan, dtype=np.float32)
        out[start:stop] = np.stack([columns[name] for name in names], axis=1)
    if deceleration:
        extra, extra_names = compute_deceleration(match)
        out = np.column_stack((out, extra))
        names = names + extra_names
    return out, names


def swap_teams(match):
    """The same match with home and away exchanged."""
    out = dict(match)
    out["home"], out["away"] = match["away"], match["home"]
    return out


# --- player deceleration: how sharply each team brakes, from short uninterrupted slot segments
# (a slot is followed for LAG frames only, never used as an identity)

LAG = 12                        # frames, about half a second
DECELERATION = tuple("ctx_{}_{}".format(signal, team) for signal in
                     ("mean_deceleration", "braking_share", "static_share_change",
                      "motion_valid_share") for team in ("h", "a"))


def mean_valid(x, valid):
    count = valid.sum(axis=1)
    out = np.full(len(x), np.nan, dtype=np.float32)
    np.divide(np.where(valid, x, 0).sum(axis=1), count, out=out, where=count > 0)
    return out


def motion_columns(team):
    n = len(team)
    step = np.full(team.shape[:2], np.nan, dtype=np.float32)
    step[1:] = np.linalg.norm(team[1:] - team[:-1], axis=2)
    speed = np.where(step <= MAX_STEP, step * FPS, np.nan)
    valid = np.isfinite(speed)
    continuous = np.zeros_like(valid)
    if n > LAG:
        cumulative = np.concatenate((np.zeros((1, team.shape[1]), dtype=np.int32),
                                     np.cumsum(valid, axis=0)), axis=0)
        continuous[LAG:] = (cumulative[LAG + 1:] - cumulative[:-LAG - 1]) == LAG + 1
    before = np.full_like(speed, np.nan)
    if n > LAG:
        before[LAG:] = speed[:-LAG]
    deceleration = (before - speed) / (LAG / FPS)
    braking = mean_valid((deceleration > 1).astype(np.float32), continuous)
    mean_deceleration = mean_valid(np.maximum(deceleration, 0), continuous)
    static_change = mean_valid((speed < 1).astype(np.float32) - (before < 1).astype(np.float32),
                               continuous)
    present = np.isfinite(team[:, :, 0]) & np.isfinite(team[:, :, 1])
    available = present.sum(axis=1)
    valid_share = np.full(n, np.nan, dtype=np.float32)
    np.divide(continuous.sum(axis=1), available, out=valid_share, where=available > 0)
    return mean_deceleration, braking, static_change, valid_share


def compute_deceleration(match):
    """The eight deceleration columns, with history reset at each half."""
    table = np.full((len(match["period"]), len(DECELERATION)), np.nan, dtype=np.float32)
    for start, stop, _ in period_runs(match["period"]):
        columns = {}
        for side, key in (("h", "home"), ("a", "away")):
            values = motion_columns(match[key][start:stop])
            for signal, value in zip(("mean_deceleration", "braking_share",
                                      "static_share_change", "motion_valid_share"), values):
                columns["ctx_{}_{}".format(signal, side)] = value
        table[start:stop] = np.column_stack([columns[name] for name in DECELERATION])
    return table, list(DECELERATION)
