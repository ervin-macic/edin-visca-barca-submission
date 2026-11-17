"""DFL open tracking data (Bassek et al. 2025) -> the common match format.

The positions file stores, for every half, one <FrameSet> per person and one for the ball. The
ball frames carry two league-official flags, which are exactly the label this challenge asks for:

    BallStatus      1 = ball in play, 0 = ball dead
    BallPossession  1 = home team, 2 = guest team

So the label needs no modelling: D where BallStatus is 0, otherwise the possessing team.

    python -m training.dfl        # converts every downloaded match into cache/ext/dfl_*.npz
"""

import glob
import os
import re

import numpy as np

from .common import A, D, EXTERNAL_DIR, H, X, cache_path, label_summary, save_match

SECTIONS = {"firstHalf": 1, "secondHalf": 2}
ATTR = re.compile(r'(\w+)="([^"]*)"')


def match_ids():
    pattern = os.path.join(EXTERNAL_DIR, "dfl", "DFL_04_03_positions_raw_observed_*.xml")
    return sorted(os.path.basename(p)[len("DFL_04_03_positions_raw_observed_"):-4]
                  for p in glob.glob(pattern))


def home_team_id(match_id):
    path = os.path.join(EXTERNAL_DIR, "dfl", "DFL_02_01_matchinformation_{}.xml".format(match_id))
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return re.search(r'HomeTeamId="([^"]+)"', text).group(1)


def load(match_id):
    home_id = home_team_id(match_id)
    path = os.path.join(EXTERNAL_DIR, "dfl",
                        "DFL_04_03_positions_raw_observed_{}.xml".format(match_id))

    # First pass over the file collects each frame set as (section, team, person, frames, xy,
    # flags); the arrays are assembled afterwards, once the frame range of each half is known.
    sets = []
    pitch = (105.0, 68.0)
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                parts = line.split('"')
                # N, T, X, Y are always the first four attributes.
                current["n"].append(int(parts[1]))
                current["x"].append(float(parts[5]))
                current["y"].append(float(parts[7]))
                if current["ball"]:
                    attrs = dict(ATTR.findall(line))
                    current["poss"].append(int(attrs["BallPossession"]))
                    current["status"].append(int(attrs["BallStatus"]))
            elif line.startswith("<FrameSet "):
                attrs = dict(ATTR.findall(line))
                current = {"section": SECTIONS.get(attrs["GameSection"], 0),
                           "team": attrs["TeamId"], "ball": attrs["TeamId"] == "BALL",
                           "n": [], "x": [], "y": [], "poss": [], "status": []}
                sets.append(current)
            elif line.startswith("<PitchSize "):
                attrs = dict(ATTR.findall(line))
                pitch = (float(attrs["X"]), float(attrs["Y"]))

    out = {key: [] for key in ("ball", "home", "away", "label", "period")}
    n_home = max(sum(1 for s in sets if s["section"] == sec and s["team"] == home_id)
                 for sec in (1, 2))
    n_away = max(sum(1 for s in sets if s["section"] == sec
                     and s["team"] not in (home_id, "BALL", "referee"))
                 for sec in (1, 2))
    for section in (1, 2):
        here = [s for s in sets if s["section"] == section]
        ball_set = [s for s in here if s["ball"]][0]
        first, last = min(ball_set["n"]), max(ball_set["n"])
        length = last - first + 1

        ball = np.full((length, 2), np.nan, dtype=np.float32)
        index = np.array(ball_set["n"]) - first
        ball[index, 0] = ball_set["x"]
        ball[index, 1] = ball_set["y"]
        label = np.full(length, X, dtype=np.int8)
        poss = np.array(ball_set["poss"])
        status = np.array(ball_set["status"])
        label[index] = np.where(status == 0, D, np.where(poss == 1, H, A))

        teams = {"home": np.full((length, n_home, 2), np.nan, dtype=np.float32),
                 "away": np.full((length, n_away, 2), np.nan, dtype=np.float32)}
        slot = {"home": 0, "away": 0}
        for s in here:
            if s["ball"] or s["team"] == "referee":
                continue
            side = "home" if s["team"] == home_id else "away"
            index = np.array(s["n"]) - first
            keep = (index >= 0) & (index < length)
            teams[side][index[keep], slot[side], 0] = np.array(s["x"])[keep]
            teams[side][index[keep], slot[side], 1] = np.array(s["y"])[keep]
            slot[side] += 1

        out["ball"].append(ball)
        out["home"].append(teams["home"])
        out["away"].append(teams["away"])
        out["label"].append(label)
        out["period"].append(np.full(length, section, dtype=np.int8))

    match = {key: np.concatenate(value) for key, value in out.items()}
    match["pitch"] = np.array(pitch, dtype=np.float32)
    return match


if __name__ == "__main__":
    for match_id in match_ids():
        match = load(match_id)
        save_match(cache_path("ext", "dfl_{}.npz".format(match_id.split("_")[-1])), match)
        summary = label_summary(match)
        print(match_id, {k: round(v, 3) for k, v in summary.items()}, flush=True)
