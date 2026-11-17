"""The tactical tracking as a match, with its two goalkeeper slots cleaned.

The goalkeeper slots (FIFA/EPTS position_type "Goalkeeper") behave unlike every other slot. Every
other slot only has a position while its player is inside the camera picture. The keeper slots
always have one: when the tracker does not see the keeper it writes an estimate with an empty
speed, and those estimates go stale. The tracker also sometimes files the keeper it does see under
the other team's keeper slot. Three general rules fix this (fix_keeper_*). None of them uses a
frame number, labels or the video picture; the homography only tells which part of the pitch the
camera shows. A fix only drops, moves or swaps whole (x, y, speed) groups between slots, so no
value is ever invented. The provided file is not changed.
"""

import numpy as np

from .data import FPS, HALVES, N_FRAMES, PITCH, asset_path, read_atd, read_slots, to_metres

VIDEO_SIZE = (1920, 1080)       # pixels of the tactical video; the homography maps into it
KEEPER_MAX_SPEED = 12.0         # m/s. Elite sprinters peak just above 12 m/s and goalkeepers move far
                                # slower, so a keeper position that moves faster is not a person.


def last_true_index(flag):
    """Index of the most recent True at or before each position, -1 before the first."""
    return np.maximum.accumulate(np.where(flag, np.arange(len(flag)), -1))


def read_homography(path, n_frames):
    """(T, 3, 3) matrices from normalised pitch coordinates to pixels, and (T,) validity flags."""
    table = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(0, 12))
    if not np.array_equal(table[:, 0], np.arange(n_frames)):
        raise ValueError("homography rows are not one per video frame")
    return table[:, 1:10].reshape(-1, 3, 3), table[:, 11] > 0


def in_picture(homography, frames, x, y):
    """True where normalised pitch point (x, y) on video frame `frames` is inside the picture."""
    matrices, valid = homography
    point = np.stack([x, y, np.ones_like(x)], axis=1)
    image = np.einsum("tij,tj->ti", matrices[frames], point)
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = image[:, 0] / image[:, 2], image[:, 1] / image[:, 2]
    return (valid[frames] & (image[:, 2] > 0) & (u >= 0) & (u <= VIDEO_SIZE[0])
            & (v >= 0) & (v <= VIDEO_SIZE[1]))


class Tracking:
    """The slots in arrays, plus which original slot each slot holds after the fixes.

    xy, speed: (T, S, 2) and (T, S) in the file's units (normalised 0-1 coordinates, m/s).
    source:    (T, S) index of the original slot whose values this slot holds on that frame;
               -1 = empty. Fixes change the data only through drop / move / swap.
    """

    def __init__(self, slots, home, keeper):
        self.n_frames = len(slots)
        self.length, self.width = PITCH
        self.home = home
        self.outfield = ~keeper
        self.keeper_slots = {}
        for is_home in (True, False):
            found = np.flatnonzero(keeper & (home == is_home))
            if len(found) != 1:
                raise ValueError("expected one goalkeeper slot per team in the metadata")
            self.keeper_slots[is_home] = int(found[0])
        self.xy = slots[:, :, :2].copy()
        self.speed = slots[:, :, 2].copy()
        self.source = np.tile(np.arange(len(home), dtype=np.int16), (self.n_frames, 1))
        self.home_left = self.home_defends_left()

    def present(self, k):
        return np.isfinite(self.xy[:, k, 0])

    def seen(self, k):
        """Position and speed: the tracker saw this object on this frame."""
        return self.present(k) & np.isfinite(self.speed[:, k])

    def made_up(self, k):
        """Position without a speed: the tracker's own estimate of an object it did not see."""
        return self.present(k) & ~np.isfinite(self.speed[:, k])

    def metres(self, k):
        return self.xy[:, k, 0] * self.length, self.xy[:, k, 1] * self.width

    def home_defends_left(self):
        """(T,) True where the home team defends the goal at x = 0.

        Per half: a team defends the side its outfield slots stand nearer to on average over the
        half. Frames before a half's graded block take that half's answer.
        """
        out = np.zeros(self.n_frames, dtype=bool)
        home_of, away_of = self.home & self.outfield, ~self.home & self.outfield
        starts = [0] + [first for _, first, _ in HALVES[1:]] + [self.n_frames]
        for (_, first, last), a, b in zip(HALVES, starts[:-1], starts[1:]):
            xs = self.xy[first:last + 1, :, 0]
            out[a:b] = bool(np.nanmean(xs[:, home_of]) < np.nanmean(xs[:, away_of]))
        return out

    def drop(self, frames, k):
        self.xy[frames, k] = np.nan
        self.speed[frames, k] = np.nan
        self.source[frames, k] = -1

    def move(self, frames, src, dst):
        if np.isfinite(self.xy[frames, dst, 0]).any():
            raise ValueError("move into an occupied slot")
        for array in (self.xy, self.speed, self.source):
            array[frames, dst] = array[frames, src]
        self.drop(frames, src)

    def swap(self, frames, a, b):
        for array in (self.xy, self.speed, self.source):
            held = array[frames, a].copy()
            array[frames, a] = array[frames, b]
            array[frames, b] = held


def fix_keeper_ghosts(trk, homography):
    """Drop made-up keeper positions (no speed) that lie inside the camera picture.

    The camera shows that spot and the tracker found nobody there, so the estimate is stale.
    Estimates outside the picture stay: they are the tracker's guess for a keeper the camera does
    not show, usually near his goal.
    """
    for k in trk.keeper_slots.values():
        frames = np.flatnonzero(trk.made_up(k))
        inside = in_picture(homography, frames, trk.xy[frames, k, 0], trk.xy[frames, k, 1])
        trk.drop(frames[inside], k)


def fix_keeper_teams(trk):
    """Give each keeper to the team that defends the half he stands in; at most one per team.

    A goalkeeper does not cross the halfway line in open play, so a keeper in the left half is the
    keeper of the team defending the left goal. When both keeper slots stand in the same half they
    are two versions of one keeper. The one kept is, in order: a sighting over a made-up position;
    the more recent sighting; the one already in that team's slot.
    """
    hk, ak = trk.keeper_slots[True], trk.keeper_slots[False]
    home_left = trk.home_left
    n = trk.n_frames
    present = {k: trk.present(k) for k in (hk, ak)}
    belongs_home = {k: present[k] & ((trk.xy[:, k, 0] < 0.5) == home_left) for k in (hk, ak)}
    belongs_away = {k: present[k] & ~belongs_home[k] for k in (hk, ak)}

    def rank(k, own_slot):
        seen = trk.seen(k)
        last = last_true_index(seen)
        age = np.where(last >= 0, np.arange(n) - last, n + 10 ** 6).astype(np.float64)
        return seen * 1e9 - age + 0.5 * own_slot

    both_home = belongs_home[hk] & belongs_home[ak]
    both_away = belongs_away[hk] & belongs_away[ak]
    away_wins = both_home & (rank(ak, 0) > rank(hk, 1))   # the away slot holds the better home keeper
    home_wins = both_away & (rank(hk, 0) > rank(ak, 1))   # the home slot holds the better away keeper
    swapped = belongs_away[hk] & belongs_home[ak]
    home_slot_only = belongs_away[hk] & ~present[ak]
    away_slot_only = belongs_home[ak] & ~present[hk]

    f = np.flatnonzero
    trk.drop(f(both_home & ~away_wins), ak)
    trk.drop(f(away_wins), hk)
    trk.move(f(away_wins), ak, hk)
    trk.drop(f(both_away & ~home_wins), hk)
    trk.drop(f(home_wins), ak)
    trk.move(f(home_wins), hk, ak)
    trk.swap(f(swapped), hk, ak)
    trk.move(f(home_slot_only), hk, ak)
    trk.move(f(away_slot_only), ak, hk)


def fix_keeper_jumps(trk):
    """Drop keeper positions that move faster than KEEPER_MAX_SPEED from the previous frame.

    A sighting right after a made-up position is kept even when it is far from it: the estimate
    was stale, the sighting is not.
    """
    limit = KEEPER_MAX_SPEED / FPS
    for k in trk.keeper_slots.values():
        x, y = trk.metres(k)
        present, seen = trk.present(k), trk.seen(k)
        step = np.hypot(np.diff(x), np.diff(y))
        bad = np.zeros(trk.n_frames, dtype=bool)
        with np.errstate(invalid="ignore"):
            bad[1:] = (step > limit) & present[1:] & present[:-1]
        bad[1:] &= ~(seen[1:] & ~seen[:-1])
        trk.drop(np.flatnonzero(bad), k)


def load_tracking(data_dir, clean=True):
    """The tactical tracking (ATD) as a match, goalkeeper slots cleaned unless clean=False."""
    home, keeper = read_slots(asset_path(data_dir, "tactical", "_atd.xml"))
    n = N_FRAMES["tactical"]
    slots, ball = read_atd(asset_path(data_dir, "tactical", "_atd.txt"), len(home), n)
    if clean:
        trk = Tracking(slots, home, keeper)
        homography = read_homography(asset_path(data_dir, "tactical", "_homography.csv"), n)
        fix_keeper_ghosts(trk, homography)
        fix_keeper_teams(trk)
        fix_keeper_jumps(trk)
        slots = slots[np.arange(n)[:, None], np.maximum(trk.source, 0)]
        slots[trk.source < 0] = np.nan
    period = np.zeros(n, dtype=np.int8)
    for half, first, last in HALVES:
        period[first:last + 1] = half
    return {
        "ball": to_metres(ball[:, :2]),
        "home": to_metres(slots[:, home, :2]),
        "away": to_metres(slots[:, ~home, :2]),
        "period": period,
        "pitch": np.array(PITCH, dtype=np.float32),
    }
