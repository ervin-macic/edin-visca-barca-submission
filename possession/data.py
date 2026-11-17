"""Constants of the match and readers for the provided files.

A match is a dict of numpy arrays over the frames of the tactical video (25 fps):

    ball    float32 (T, 2)     ball x, y in metres, pitch centre at (0, 0), x along the length;
                               NaN where the ball is not tracked
    home    float32 (T, P, 2)  home-team track slots (Barcelona), NaN where a slot is empty
    away    float32 (T, Q, 2)  away-team track slots (Athletic Club)
    period  int8    (T,)       1 or 2 on the graded frames of each half, 0 elsewhere
    pitch   float32 (2,)       pitch length and width in metres

Track slots carry a team but no player identity, so every feature treats a team as an unordered
set of points.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

GAME = "BAR-ATH"
FPS = 25
PITCH = (105.3, 68.0)                                   # metres
N_FRAMES = {"tactical": 199319, "broadcast": 164488}    # video frames of each camera
HALVES = ((1, 11185, 90651), (2, 110808, 190197))       # graded tactical frames: (half, first, last)
H, A, D, X = 0, 1, 2, 3                                 # label codes
LETTERS = np.array(["H", "A", "D", "X"])


def asset_path(data_dir, source, suffix):
    """data_dir/<source>/BAR-ATH_<source><suffix>, the layout of the data bundle."""
    return os.path.join(data_dir, source, "{}_{}{}".format(GAME, source, suffix))


def period_runs(period):
    """[(start, stop, period)] half-open spans of constant non-zero period."""
    edges = np.flatnonzero(np.diff(period.astype(np.int16)) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(period)]))
    return [(int(a), int(b), int(period[a])) for a, b in zip(bounds[:-1], bounds[1:])
            if period[a] != 0]


def read_slots(meta_path):
    """Per track slot, in data order: home flag and goalkeeper flag (FIFA/EPTS metadata).

    The first Team element is the home side.
    """
    root = ET.parse(meta_path).getroot()
    home_team = next(root.iter("Team")).get("id")
    channels = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    order = []
    for ref in root.iter("PlayerChannelRef"):
        player = channels[ref.get("playerChannelId")]
        if not order or order[-1] != player:
            order.append(player)
    flags = {}
    for player in root.iter("Player"):
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        flags[player.get("id")] = (player.get("teamId") == home_team,
                                   params.get("position_type") == "Goalkeeper")
    home = np.array([flags[p][0] for p in order])
    keeper = np.array([flags[p][1] for p in order])
    return home, keeper


def read_atd(raw_path, n_slots, n_frames):
    """FIFA/EPTS raw tracking -> (slots (T, S, 3), ball (T, 3)) float64, one row per video frame.

    Each line is "<frame_id>:<slot>;<slot>;...:<ball>", every group "x,y,speed" in normalised 0-1
    pitch units with NaN for a missing value. frame_id counts from 1.
    """
    width = 1 + 3 * n_slots + 3
    # Stream the lines instead of holding millions of Python strings at once.
    with open(raw_path) as handle:
        table = np.loadtxt((line.replace(":", ",").replace(";", ",") for line in handle),
                           delimiter=",", dtype=np.float64, ndmin=2)
    if table.shape != (n_frames, width) or not np.array_equal(table[:, 0],
                                                              np.arange(1, n_frames + 1)):
        raise ValueError("{}: expected frames 1..{} with {} columns".format(raw_path, n_frames,
                                                                          width))
    return table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3), table[:, -3:]


def to_metres(values):
    """Normalised 0-1 pitch coordinates (..., 2) -> float32 metres with the centre at (0, 0)."""
    return ((values - 0.5) * np.array(PITCH)).astype(np.float32)


def kickoff_windows(data_dir):
    """[(first, last)] tactical frames of the Smart Tagging 'Kick Off (Start)' set pieces.

    Smart Tagging times are seconds from the start of the tactical video. Each of these tags is a
    window of about 5 s that contains the kick-off of a half.
    """
    path = asset_path(data_dir, "tactical", "_smart_tagging.xml")
    out = []
    for instance in ET.parse(path).getroot().iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        tags = {(label.findtext("group") or "").strip().lower():
                (label.findtext("text") or "").strip().lower()
                for label in instance.findall("label")}
        if code == "set pieces" and tags.get("type") == "kick off (start)":
            out.append((int(round(float(instance.findtext("start")) * FPS)),
                        int(round(float(instance.findtext("end")) * FPS))))
    return sorted(out)
