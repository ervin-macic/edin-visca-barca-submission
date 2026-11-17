"""Make clean external tracking look like the competition's automatic tracking (ATD).

The external datasets are complete: a ball position on every frame and one unbroken track per
player. The ATD is not, and a model trained on clean data would lean on signals that are absent at
test time. This file imitates what the ATD tracker does. Everything it imitates was measured on
the ATD itself without any labels:

  * the ball is lost for 41% of graded frames, always for more than 5 s at a time, and players
    are at walking pace during about 95% of that time: the tracker loses the ball during
    stoppages, some seconds after play stops, and finds it again on the restart kick;
  * shorter losses never appear as gaps, so the tracker fills them; the filled stretches join
    the observed track smoothly (the track contains no exactly straight segments and no kinks);
  * the last half second before a loss is the tracker coasting to a stop;
  * during roughly a fifth of dead-ball time the ball is tracked after all, and it is then
    moving (median 1.7 m/s, a third of the time within a metre of a player): it is being passed
    or rolled to the place of the restart. External data has nothing like this (DFL freezes the
    ball while it is dead, Metrica's files have no position), so such movement is synthesised;
  * the ball track is heavily smoothed: velocity changes above 30 m/s^2 happen 3 times a minute
    against 40 in DFL's tracking. A Gaussian filter of about 7 frames on DFL's ball reproduces the
    ATD's acceleration profile; player tracks need much less smoothing;
  * tracks are smooth frame to frame; players drop out one by one, and many at once during some
    stoppages.

Labels of the external match are used for one thing only: to know where its stoppages are, so
that ball losses can be placed in them the way the real tracker does.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .common import D, FPS

MIN_GAP = 132               # frames; shorter losses are bridged by the tracker, longer ones are gaps
COAST = 15                  # frames of decaying extrapolation before a gap


def runs_of(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def burst_mask(n, rng, missing_share, mean_missing, short_share=0.5, short_mean=6.0):
    """Boolean (n,) mask, True = dropped, built from alternating kept / dropped runs."""
    mask = np.zeros(n, dtype=bool)
    if missing_share <= 0:
        return mask
    average_missing = short_share * short_mean + (1.0 - short_share) * mean_missing
    mean_kept = average_missing * (1.0 - missing_share) / missing_share
    t = int(rng.integers(0, max(int(mean_kept), 1)))
    while t < n:
        if rng.random() < short_share:
            length = int(rng.geometric(1.0 / short_mean))
        else:
            length = int(rng.geometric(1.0 / mean_missing))
        mask[t:t + length] = True
        t += length + int(rng.geometric(1.0 / max(mean_kept, 1.0)))
    return mask


def nan_gaussian(values, sigma):
    """Gaussian smoothing along time that ignores missing samples and keeps them missing."""
    if sigma <= 0:
        return values
    known = np.isfinite(values)
    filled = gaussian_filter1d(np.where(known, values, 0.0).astype(np.float64), sigma, axis=0,
                               mode="nearest")
    weight = gaussian_filter1d(known.astype(np.float64), sigma, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = filled / weight
    out[~known] = np.nan
    return out.astype(np.float32)


def smooth_noise(shape, rng, sigma, window):
    """Slowly varying position error with std sigma: random offsets every `window` frames,
    joined by straight lines. Per-frame jitter would wreck every speed feature in a way the real
    input (second differences of a centimetre) does not.
    """
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    n = shape[0]
    knots = rng.normal(0.0, sigma, size=(n // window + 2,) + tuple(shape[1:]))
    position = np.arange(n) / float(window)
    low = position.astype(int)
    weight = (position - low).reshape((-1,) + (1,) * (len(shape) - 1))
    return (knots[low] * (1.0 - weight) + knots[low + 1] * weight).astype(np.float32)


def stoppage_ball(ball, home, away, a, b, rng, leg_rate, roll_on, carry):
    """What the ball does, and when the tracker can see it, during the stoppage [a, b).

    The tracker follows a ball that moves and loses one that lies still or is carried. So the
    stoppage is a still (hidden) ball interrupted by a few visible "legs": the ball rolling on
    after play stops, being passed or thrown between people, and finally travelling to the place
    of the restart, where it lies hidden again until play resumes.

    Returns (path (b - a, 2), hidden (b - a,) bool), or (None, None) when the ends are unknown.
    """
    n = len(ball)
    if a < 3 or b >= n or not (np.all(np.isfinite(ball[a - 3:a, 0])) and np.isfinite(ball[b, 0])):
        return None, None
    length = b - a
    path = np.empty((length, 2), dtype=np.float64)
    hidden = np.ones(length, dtype=bool)
    position = ball[a - 1].astype(np.float64)
    restart = ball[b].astype(np.float64)
    t = 0

    if rng.random() < roll_on and length > 60:
        duration = int(rng.integers(12, 50))
        velocity = (ball[a - 1] - ball[a - 3]).astype(np.float64) / 2.0
        path[:duration] = position + np.cumsum(np.linspace(1.0, 0.0, duration))[:, None] * velocity
        hidden[:duration] = False
        position = path[duration - 1].copy()
        t = duration

    players = np.concatenate((home[a:b], away[a:b]), axis=1)
    complete = np.flatnonzero(np.all(np.isfinite(players[:, :, 0]), axis=0))
    n_legs = int(rng.poisson(leg_rate * length / float(FPS)))
    starts = np.sort(rng.integers(t, max(t + 1, length - 20), size=n_legs)) if length - t > 40 else []
    for index, start in enumerate(starts):
        start = max(int(start), t)
        duration = int(rng.integers(20, 125))
        stop = min(start + duration, length)
        if index + 1 < len(starts):
            stop = min(stop, int(starts[index + 1]))
        if stop - start < 8:
            continue
        if index + 1 == len(starts):
            goal = restart
        elif len(complete):
            goal = players[stop - 1, complete[int(rng.integers(0, len(complete)))]].astype(np.float64)
            if np.hypot(*(goal - position)) > 30.0:
                goal = position + (goal - position) * 30.0 / np.hypot(*(goal - position))
        else:
            goal = position + rng.normal(0.0, 8.0, size=2)
        path[t:start] = position
        # a rolling ball slows down: never an exactly straight, constant-speed line (which is what
        # a bridged loss looks like)
        u = (np.arange(stop - start) + 1.0) / float(stop - start)
        w = (1.0 - (1.0 - u) ** rng.uniform(1.3, 2.2))[:, None]
        if len(complete) and rng.random() < carry:
            # the ball is walked along by the player nearest to it, close to their feet
            distance = np.hypot(players[start, complete, 0] - position[0],
                                players[start, complete, 1] - position[1])
            carrier = players[start:stop, complete[int(np.argmin(distance))]].astype(np.float64)
            offset = rng.normal(0.0, 0.4, size=2)
            reach = np.minimum(np.arange(stop - start) / 12.0, 1.0)[:, None]
            path[start:stop] = position * (1.0 - reach) + (carrier + offset) * reach
            goal = path[stop - 1].copy()
        else:
            path[start:stop] = position * (1.0 - w) + goal * w
        hidden[start:stop] = False
        position = goal.copy()
        t = stop
    path[t:] = position
    return path.astype(np.float32), hidden


def undetected_mask(n, rng, occluded, alive_lost):
    """True where the tracker has no detection for reasons unrelated to stoppages."""
    lost = np.zeros(n, dtype=bool)

    # occlusions at any time: short, bridged later by a straight line
    if occluded > 0:
        mean_length = 30.0
        mean_kept = mean_length * (1.0 - occluded) / occluded
        t = int(rng.geometric(1.0 / mean_kept))
        while t < n:
            length = min(MIN_GAP - 2, 2 + int(rng.geometric(1.0 / mean_length)))
            lost[t:t + length] = True
            t += length + int(rng.geometric(1.0 / mean_kept))

    # rare long losses while the ball is in play
    if alive_lost > 0:
        mean_length = 1.6 * MIN_GAP
        mean_kept = mean_length * (1.0 - alive_lost) / alive_lost
        t = int(rng.geometric(1.0 / mean_kept))
        while t < n:
            length = MIN_GAP + int(rng.geometric(1.0 / (0.6 * MIN_GAP)))
            lost[t:t + length] = True
            t += length + int(rng.geometric(1.0 / mean_kept))
    return lost


def track_ball(ball, lost):
    """What the tracker outputs: short losses bridged by a line, long ones empty after coasting."""
    out = ball.copy()
    n = len(ball)
    for a, b in runs_of(lost | ~np.isfinite(ball[:, 0])):
        before, after = a - 1, b
        bridgeable = (b - a < MIN_GAP and before >= 0 and after < n
                      and np.isfinite(ball[before, 0]) and np.isfinite(ball[after, 0]))
        if bridgeable:
            w = ((np.arange(a, b) - before) / float(after - before))[:, None]
            out[a:b] = ball[before] * (1.0 - w) + ball[after] * w
            continue
        out[a:b] = np.nan
        # coasting: the last frames before the gap are a decaying extrapolation, not the ball
        first = a - COAST
        if first >= 2 and np.all(np.isfinite(out[first - 2:a, 0])):
            velocity = (out[first] - out[first - 2]) / 2.0
            factors = np.linspace(1.0, 0.25, COAST)
            out[first:a] = out[first] + np.cumsum(factors)[:, None] * velocity
    return out


def flip_players(home, away, rng, rate, median, sigma):
    """Copies of the two teams in which some players are filed under the other team for a while.

    The real tracker does this to outfield players: on about 11% of graded frames one team shows
    more objects than it has players, because a track restarted under the other team. The
    flipped tracks are not concentrated near the ball and
    last from seconds to minutes. `rate` is episodes per minute; durations are log-normal with
    median `median` seconds. The player is chosen at random among those present.
    """
    n = len(home)
    teams = [home.copy(), away.copy()]
    for _ in range(int(rng.poisson(rate * n / (FPS * 60.0)))):
        src = int(rng.integers(0, 2))
        start = int(rng.integers(0, n))
        length = int(np.clip(np.exp(rng.normal(np.log(median * FPS), sigma)), FPS, 180 * FPS))
        stop = min(n, start + length)
        present = np.flatnonzero(np.isfinite(teams[src][start, :, 0]))
        if not len(present):
            continue
        k = int(rng.choice(present))
        dst = teams[1 - src]
        free = np.flatnonzero(~np.isfinite(dst[start:stop, :, 0]).any(axis=0))
        if len(free):
            col = int(free[0])
        else:
            dst = np.concatenate((dst, np.full((n, 1, 2), np.nan, dtype=dst.dtype)), axis=1)
            col = dst.shape[1] - 1
        dst[start:stop, col] = teams[src][start:stop, k]
        teams[src][start:stop, k] = np.nan
        teams[1 - src] = dst
    return teams[0], teams[1]


FLIP_RATE = 0.75                # episodes per minute: simulated DFL then shows a team above 11
                                # objects on 11% of frames, as the real tracking does
FLIP_MEDIAN = 12.0              # seconds
FLIP_SIGMA = 1.0


def glitch_ball(ball, lost, rng, rate, size, length, far=0.0, players=None):
    """Copy of the ball track with short false detections: now and then, for a few frames, the
    tracker follows something that is not the ball (offset of about `size` metres) and then finds
    the ball again. Applied before the tracker's smoothing, so each glitch becomes a smooth bump.

    Why: on the real tracking, strong accelerations happen away from the players far more often
    than in smoothed external tracking (90th percentile 17.9 against 12.4 m/s^2 more than 5 m from
    any player), while accelerations at touches are lower. `rate` is glitches per minute,
    `length` the mean duration in frames.
    far > 0 limits them to frames on which the ball is more than `far` metres from every player
    (`players`: (T, P, 2)), i.e. to a ball in flight.
    """
    out = ball.copy()
    n = len(ball)
    if far > 0:
        with np.errstate(invalid="ignore"):
            distance = np.hypot(players[:, :, 0] - ball[:, None, 0],
                                players[:, :, 1] - ball[:, None, 1])
        nearest = np.min(np.where(np.isfinite(distance), distance, np.inf), axis=1)
        in_flight = nearest > far
    for _ in range(int(rng.poisson(rate * n / (FPS * 60.0)))):
        start = int(rng.integers(0, n))
        stop = min(n, start + 1 + int(rng.geometric(1.0 / max(length, 1.0))))
        offset = rng.normal(0.0, size, size=2).astype(np.float32)
        keep = ~lost[start:stop]
        if far > 0:
            keep &= in_flight[start:stop]
        out[start:stop][keep] += offset
    return out


# The real tracker's ball texture, from label-free statistics of the provided tracking: ball
# acceleration overall, away from the players and at touch peaks, touches per minute, and the
# distance from the ball to the nearest player. Optional (random_settings(texture=True)); the
# shipped models are trained without it.
TEXTURE = {"smooth_ball": 7.0, "glitch_rate": 30.0, "glitch_size": 1.5, "glitch_len": 6.0}


def degrade(match, rng, leg_rate=0.06, roll_on=0.4, carry=0.35, lag_mean=-15.0, lag_sd=20.0,
            return_max=15,
            occluded=0.09, alive_lost=0.03, jitter_ball=0.15, jitter_player=0.10,
            player_missing=0.05, player_mean_missing=150.0, crowd_loss=0.3, smooth_ball=5.5,
            smooth_player=1.5, wobble_ball=0.015, flip_rate=0.0, flip_median=FLIP_MEDIAN,
            flip_sigma=FLIP_SIGMA, glitch_rate=0.0, glitch_size=1.0, glitch_len=6.0,
            glitch_far=0.0):
    """A copy of a match as the automatic tracker would have delivered it.

    flip_rate > 0 also files players under the wrong team now and then (flip_players).
    glitch_rate > 0 adds short false ball detections before the smoothing (glitch_ball); with
    glitch_far > 0 only while the ball is that far from every player.
    Both are 0 by default and draw no random numbers then, so earlier recipes are reproduced
    exactly.
    """
    out = dict(match)
    label = match["label"]
    n = len(label)

    ball = match["ball"].copy()
    lost = undetected_mask(n, rng, occluded, alive_lost)
    for a, b in runs_of(label == D):
        path, hidden = stoppage_ball(match["ball"], match["home"], match["away"], a, b, rng,
                                     leg_rate, roll_on, carry)
        if path is None:
            lost[a:b] = True
            continue
        ball[a:b] = path
        lost[a:b] |= hidden
        if hidden[0]:
            # no roll-on: the ball was already lost on its way out, a little before play stopped
            lost[max(0, a + min(0, int(round(rng.normal(lag_mean, lag_sd))))):a] = True
        lost[b:b + int(rng.integers(0, return_max + 1))] = True     # found again after the kick
    ball += smooth_noise(ball.shape, rng, jitter_ball, 25)
    if glitch_rate > 0:
        ball = glitch_ball(ball, lost, rng, glitch_rate, glitch_size, glitch_len, glitch_far,
                           np.concatenate((match["home"], match["away"]), axis=1))
    # Order matters: short losses are filled first and the whole run is smoothed afterwards, so
    # a filled stretch joins the observed track without a kink, as it does in the real tracking.
    ball = nan_gaussian(track_ball(ball, lost), smooth_ball)
    if wobble_ball > 0:
        ball += (wobble_ball * gaussian_filter1d(rng.normal(0.0, 1.0, size=ball.shape), 2.0,
                                                 axis=0, mode="nearest")).astype(np.float32)
    out["ball"] = ball

    # stoppages in which the tracker also loses many players at once
    crowd = []
    for a, b in runs_of(~np.isfinite(ball[:, 0])):
        if b - a >= MIN_GAP and rng.random() < crowd_loss:
            start = int(rng.integers(a, b))
            stop = int(rng.integers(start, b)) + 1
            crowd.append((start, stop, float(rng.uniform(0.2, 0.9))))

    for side in ("home", "away"):
        team = nan_gaussian(match[side] + smooth_noise(match[side].shape, rng, jitter_player, 50),
                            smooth_player)
        for slot in range(team.shape[1]):
            team[burst_mask(n, rng, player_missing, player_mean_missing, 0.3), slot] = np.nan
            for start, stop, share in crowd:
                if rng.random() < share:
                    team[start:stop, slot] = np.nan
        out[side] = team
    if flip_rate > 0:
        out["home"], out["away"] = flip_players(out["home"], out["away"], rng, flip_rate,
                                                flip_median, flip_sigma)
    return out


def random_settings(rng, flips=False, texture=False):
    """One draw of tracker behaviour, wide on purpose so no single setting is baked in.

    With flips=True the draw also includes wrong-team episodes (flip_players). With texture=True
    it also includes false ball detections (glitch_ball), and the ball smoothing is drawn again
    around the TEXTURE setting. These numbers are drawn after all the others, so the other
    settings of a draw stay the same.
    """
    settings = {
        "leg_rate": float(rng.uniform(0.02, 0.12)),
        "roll_on": float(rng.uniform(0.0, 0.7)),
        "carry": float(rng.uniform(0.2, 0.8)),
        "lag_mean": float(rng.uniform(-40.0, 0.0)),
        "lag_sd": float(rng.uniform(10.0, 40.0)),
        "return_max": int(rng.integers(5, 30)),
        "occluded": float(rng.uniform(0.03, 0.16)),
        "alive_lost": float(rng.uniform(0.0, 0.07)),
        "jitter_ball": float(rng.uniform(0.0, 0.30)),
        "jitter_player": float(rng.uniform(0.0, 0.30)),
        "player_missing": float(rng.uniform(0.02, 0.10)),
        "player_mean_missing": float(rng.uniform(3.0, 10.0) * FPS),
        "crowd_loss": float(rng.uniform(0.1, 0.5)),
        "smooth_ball": float(rng.uniform(4.0, 8.0)),
        "wobble_ball": float(rng.uniform(0.0, 0.04)),
        "smooth_player": float(rng.uniform(0.0, 3.0)),
    }
    if flips:
        settings["flip_rate"] = float(rng.uniform(0.5, 2.0) * FLIP_RATE)
        settings["flip_median"] = float(rng.uniform(0.5, 2.0) * FLIP_MEDIAN)
        settings["flip_sigma"] = float(rng.uniform(0.6, 1.4))
    if texture:
        # wide around the measured setting: from the default smoothing (5.5) to well above it,
        # and from few to many false detections
        settings["smooth_ball"] = float(rng.uniform(TEXTURE["smooth_ball"] - 2.0,
                                                    TEXTURE["smooth_ball"] + 2.0))
        settings["glitch_rate"] = float(rng.uniform(0.25, 1.75) * TEXTURE["glitch_rate"])
        settings["glitch_size"] = float(rng.uniform(0.5, 1.5) * TEXTURE["glitch_size"])
        settings["glitch_len"] = float(rng.uniform(0.5, 1.5) * TEXTURE["glitch_len"])
    return settings
